
"""
@file        auth.py
@author      ParisNeo
@description API-key authentication, quotas, and live user administration.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from fastapi import HTTPException

@dataclass(frozen=True)
class UserContext:
    user_id: str
    is_admin: bool = False
    api_key: Optional[str] = None

@dataclass(frozen=True)
class _UserRecord:
    user_id: str
    api_key: str
    is_admin: bool
    max_concurrent_jobs: int
    quota_per_hour: int

class _QuotaWindow:
    def __init__(self, limit: int, window_seconds: int = 3600) -> None:
        self.limit = max(1, limit)
        self.window = window_seconds
        self.events: List[float] = []
        self.lock = threading.Lock()

    def try_consume(self) -> bool:
        now = time.time()
        with self.lock:
            self.events = [event for event in self.events if event >= now - self.window]
            if len(self.events) >= self.limit:
                return False
            self.events.append(now)
            return True

class AuthManager:
    """Thread-safe mutable key index backed by the persisted service config."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.quotas: Dict[str, _QuotaWindow] = {}
        self.records: Dict[str, _UserRecord] = {}
        self.reload(config)

    def reload(self, config: Dict) -> None:
        auth = config.get("auth", {})
        records: Dict[str, _UserRecord] = {}
        for user_id, settings in (auth.get("users", {}) or {}).items():
            key = settings.get("api_key", "")
            if not key:
                continue
            records[user_id] = _UserRecord(
                user_id=user_id,
                api_key=key,
                is_admin=bool(settings.get("is_admin", False)),
                max_concurrent_jobs=max(1, int(settings.get("max_concurrent_jobs", 2))),
                quota_per_hour=max(1, int(settings.get("quota_per_hour", 200))),
            )
        with self.lock:
            self.config = config
            self.enabled = bool(auth.get("enabled", False))
            self.allow_anonymous = bool(auth.get("allow_anonymous", True))
            self.records = records
            self.quotas = {
                user_id: self.quotas.get(user_id) or _QuotaWindow(record.quota_per_hour)
                for user_id, record in records.items()
            }
            for user_id, record in records.items():
                self.quotas[user_id].limit = record.quota_per_hour

    def list_users(self) -> List[str]:
        with self.lock:
            return list(self.records)

    def public_users(self) -> List[Dict]:
        with self.lock:
            return [{
                "user_id": record.user_id,
                "is_admin": record.is_admin,
                "max_concurrent_jobs": record.max_concurrent_jobs,
                "quota_per_hour": record.quota_per_hour,
                "key_hint": f"••••{record.api_key[-4:]}",
            } for record in self.records.values()]

    def resolve(self, api_key: Optional[str]) -> UserContext:
        with self.lock:
            if not self.enabled:
                return UserContext("anonymous", True)
            if api_key:
                for record in self.records.values():
                    if hmac.compare_digest(record.api_key, api_key):
                        return UserContext(record.user_id, record.is_admin)
                raise HTTPException(status_code=401, detail="Invalid API key.")
            if self.allow_anonymous:
                return UserContext("anonymous")
            raise HTTPException(status_code=401, detail="Send an X-API-Key or Bearer token.")

    def create_user(
        self,
        user_id: str,
        is_admin: bool = False,
        max_concurrent_jobs: int = 2,
        quota_per_hour: int = 200,
    ) -> str:
        with self.lock:
            if user_id in self.records or user_id == "anonymous":
                raise ValueError("User already exists or is reserved.")
            api_key = "oig_" + secrets.token_urlsafe(32)
            self.config.setdefault("auth", {}).setdefault("users", {})[user_id] = {
                "api_key": api_key,
                "is_admin": is_admin,
                "max_concurrent_jobs": max_concurrent_jobs,
                "quota_per_hour": quota_per_hour,
            }
            self.reload(self.config)
            return api_key

    def rotate_key(self, user_id: str) -> str:
        with self.lock:
            if user_id not in self.records:
                raise KeyError(user_id)
            api_key = "oig_" + secrets.token_urlsafe(32)
            self.config["auth"]["users"][user_id]["api_key"] = api_key
            self.reload(self.config)
            return api_key

    def delete_user(self, user_id: str) -> None:
        with self.lock:
            if user_id not in self.records:
                raise KeyError(user_id)
            if self.records[user_id].is_admin and sum(
                record.is_admin for record in self.records.values()
            ) <= 1:
                raise ValueError("Cannot delete the last administrator.")
            del self.config["auth"]["users"][user_id]
            self.reload(self.config)

    def enforce_quota(self, user_id: str) -> None:
        with self.lock:
            quota = self.quotas.get(user_id)
        if quota and not quota.try_consume():
            raise HTTPException(status_code=429, detail=f"Hourly quota exceeded for '{user_id}'.")

    def max_concurrent(self, user_id: str) -> int:
        with self.lock:
            record = self.records.get(user_id)
            return record.max_concurrent_jobs if record else 4

    def can_access(self, user: UserContext, owner: Optional[str]) -> bool:
        return not self.enabled or user.is_admin or owner == user.user_id
