"""Job storage abstraction for OpenImageGen.

`JobStore` is an interface; `InMemoryJobStore` is the default implementation.
Swap in a persistent implementation (SQLite, Redis, etc.) by passing to the
service constructor.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .schemas import JobStatus

logger = logging.getLogger("OpenImageGen.jobs")


class JobStore(ABC):
    """Abstract interface for storing generation jobs."""

    @abstractmethod
    def add(self, job: JobStatus) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[JobStatus]: ...

    @abstractmethod
    def update(self, job: JobStatus) -> None: ...

    @abstractmethod
    def delete(self, job_id: str) -> None: ...

    @abstractmethod
    def list_ids(self) -> List[str]: ...

    @abstractmethod
    def list_expired(self, current_time: Optional[float] = None) -> List[str]: ...

    @abstractmethod
    def count(self) -> int: ...


class InMemoryJobStore(JobStore):
    """Thread-safe in-memory job store. Data is lost on process restart."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobStatus] = {}
        self._lock = threading.RLock()

    def add(self, job: JobStatus) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: JobStatus) -> None:
        with self._lock:
            if job.job_id in self._jobs:
                self._jobs[job.job_id] = job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._jobs.keys())

    def list_expired(self, current_time: Optional[float] = None) -> List[str]:
        now = current_time if current_time is not None else time.time()
        with self._lock:
            return [job_id for job_id, job in self._jobs.items() if now > job.expires_at]

    def count(self) -> int:
        with self._lock:
            return len(self._jobs)
