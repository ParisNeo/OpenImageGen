
"""
@file        download_jobs.py
@author      ParisNeo
@description Persistent, observable background jobs for model downloads.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

ProgressReporter = Callable[[Optional[int], str], None]
DownloadWorker = Callable[[ProgressReporter], Dict[str, bool]]

class DownloadJobManager:
    """Runs downloads serially and exposes thread-safe, JSON-serializable status."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oig-download")

    def submit(
        self,
        label: str,
        model_keys: Sequence[str],
        worker: DownloadWorker,
    ) -> Dict[str, Any]:
        keys = list(model_keys)
        with self._lock:
            for job in self._jobs.values():
                if job["status"] in {"pending", "running"} and job["model_keys"] == keys:
                    return dict(job)

            job_id = str(uuid.uuid4())
            job = {
                "job_id": job_id,
                "label": label,
                "model_keys": keys,
                "status": "pending",
                "progress": 0,
                "message": "Waiting for download worker…",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "results": None,
                "error": None,
            }
            self._jobs[job_id] = job

        self._executor.submit(self._run, job_id, worker)
        return self.get(job_id) or {}

    def _run(self, job_id: str, worker: DownloadWorker) -> None:
        self._update(
            job_id,
            status="running",
            progress=0,
            message="Starting download…",
            started_at=time.time(),
        )

        def report(progress: Optional[int], message: str) -> None:
            normalized = None if progress is None else max(0, min(100, int(progress)))
            self._update(job_id, progress=normalized, message=message)

        try:
            results = worker(report)
            succeeded = bool(results) and all(results.values())
            failed_models = [key for key, success in results.items() if not success]
            self._update(
                job_id,
                status="completed" if succeeded else "failed",
                progress=100 if succeeded else None,
                message=(
                    "All model files are ready."
                    if succeeded
                    else f"Download failed for: {', '.join(failed_models) or 'unknown model'}"
                ),
                results=results,
                finished_at=time.time(),
            )
        except Exception as error:
            self._update(
                job_id,
                status="failed",
                progress=None,
                message=str(error),
                error=str(error),
                finished_at=time.time(),
            )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values()]
        return sorted(jobs, key=lambda job: job["created_at"], reverse=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
