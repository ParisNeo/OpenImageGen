"""
@file        websocket_manager.py
@author      ParisNeo
@description WebSocket connection manager for real-time job progress streaming.
             Manages per-job WebSocket connections and broadcasts progress updates
             from background generation threads to connected clients.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger("OpenImageGen.websocket")


class WebSocketManager:
    """Manages WebSocket connections keyed by job_id for targeted broadcasting."""

    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the main event loop for cross-thread coroutine scheduling."""
        self._loop = loop

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        """Accept a WebSocket connection and register it under a job_id."""
        await ws.accept()
        async with self._lock:
            if job_id not in self._connections:
                self._connections[job_id] = set()
            self._connections[job_id].add(ws)
        logger.debug(f"WebSocket connected for job {job_id}")

    async def disconnect(self, job_id: str, ws: WebSocket) -> None:
        """Remove a WebSocket connection from the job's connection set."""
        async with self._lock:
            conns = self._connections.get(job_id)
            if conns:
                conns.discard(ws)
                if not conns:
                    del self._connections[job_id]

    async def broadcast(self, job_id: str, data: dict) -> None:
        """Send a JSON message to all WebSocket clients watching a given job."""
        async with self._lock:
            conns = set(self._connections.get(job_id, set()))
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                pass

    def notify_sync(self, job_id: str, data: dict) -> None:
        """Thread-safe notification from sync context (progress callbacks).

        The generation pipeline runs in a thread pool executor. This method
        schedules the async broadcast on the main event loop using
        ``asyncio.run_coroutine_threadsafe``.
        """
        if not self._connections.get(job_id) or self._loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(job_id, data), self._loop)
        except Exception as e:
            logger.debug(f"WebSocket notify_sync failed: {e}")

    def has_connections(self, job_id: str) -> bool:
        """Return True if any WebSocket client is connected for the given job."""
        return bool(self._connections.get(job_id))