"""Tests for the InMemoryJobStore."""

from __future__ import annotations

import threading
import time

import pytest

from openimagegen.jobs import InMemoryJobStore, JobStore
from openimagegen.schemas import JobStatus


def _make_job(job_id: str, expires_at: float, status: str = "pending") -> JobStatus:
    return JobStatus(
        job_id=job_id, status=status,
        created_at=expires_at - 100, expires_at=expires_at,
    )


class TestBasic:
    def test_implements_interface(self):
        assert isinstance(InMemoryJobStore(), JobStore)

    def test_add_and_get(self):
        store = InMemoryJobStore()
        job = _make_job("j1", expires_at=time.time() + 100)
        store.add(job)
        assert store.get("j1") is job
        assert store.count() == 1

    def test_get_missing_returns_none(self):
        assert InMemoryJobStore().get("nope") is None

    def test_update_existing(self):
        store = InMemoryJobStore()
        job = _make_job("j1", expires_at=time.time() + 100)
        store.add(job)
        job.status = "completed"
        store.update(job)
        assert store.get("j1").status == "completed"

    def test_update_nonexistent_is_noop(self):
        store = InMemoryJobStore()
        store.update(_make_job("ghost", expires_at=time.time() + 100))
        assert store.get("ghost") is None
        assert store.count() == 0

    def test_delete(self):
        store = InMemoryJobStore()
        store.add(_make_job("j1", expires_at=time.time() + 100))
        store.delete("j1")
        assert store.get("j1") is None
        assert store.count() == 0

    def test_delete_missing_is_noop(self):
        store = InMemoryJobStore()
        store.delete("ghost")
        assert store.count() == 0

    def test_list_ids(self):
        store = InMemoryJobStore()
        for jid in ("a", "b", "c"):
            store.add(_make_job(jid, expires_at=time.time() + 100))
        assert set(store.list_ids()) == {"a", "b", "c"}


class TestExpiry:
    def test_list_expired_returns_only_expired(self):
        store = InMemoryJobStore()
        now = time.time()
        store.add(_make_job("live", expires_at=now + 1000))
        store.add(_make_job("dead", expires_at=now - 100))
        store.add(_make_job("borderline", expires_at=now - 1))
        assert set(store.list_expired(current_time=now)) == {"dead", "borderline"}

    def test_list_expired_no_time_uses_wall_clock(self):
        store = InMemoryJobStore()
        store.add(_make_job("live", expires_at=time.time() + 10000))
        assert store.list_expired() == []


class TestThreadSafety:
    def test_concurrent_adds(self):
        store = InMemoryJobStore()
        n = 100

        def add_range(start: int):
            for i in range(start, start + n):
                store.add(_make_job(f"j{i}", expires_at=time.time() + 1000))

        t1 = threading.Thread(target=add_range, args=(0,))
        t2 = threading.Thread(target=add_range, args=(n,))
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert store.count() == 2 * n

    def test_concurrent_reads_no_errors(self):
        store = InMemoryJobStore()
        store.add(_make_job("shared", expires_at=time.time() + 1000))
        errors = []

        def reader():
            for _ in range(50):
                try:
                    _ = store.get("shared")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert errors == []
