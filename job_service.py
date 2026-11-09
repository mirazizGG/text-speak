"""Durable speech scheduling with bounded inference and an exclusive admin barrier."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from dataclasses import dataclass
import logging
from pathlib import Path
import time

from fastapi import HTTPException

from engine import ProtectedVoiceError, VoiceDeletionError, VoiceNotFoundError


logger = logging.getLogger("text_speak")


@dataclass
class _AdminRequest:
    operation: str
    payload: dict
    future: asyncio.Future
    owned_upload: Path | None = None


def _consume_exception(future):
    # A disconnected HTTP caller must not leave an unobserved exception warning.
    if not future.cancelled():
        future.exception()


class JobService:
    """Only the scheduler may dispatch GPU work, including catalog mutations.

    Queued speech lives in SQLite rather than executor queues. At most ``slots``
    backend calls exist, and each completed call independently admits the next
    FIFO job. Client cancellation never owns or releases a backend slot.
    """

    def __init__(self, engine, admin_worker, backend, store, request_timeout=0):
        self.engine = engine
        self.admin_worker = admin_worker
        self.backend = backend
        self.store = store
        self.request_timeout = request_timeout
        self.catalog_lock = asyncio.Lock()
        self.ready = False
        self.failed = False
        self.task = None
        self._stopping = False
        self._wake = asyncio.Event()
        self._active = {}
        self._admins = deque()
        self._admin_active = False
        self._recover_backend = False
        self._waiters = {}
        self._last_cleanup = None
        self._cleanup_lock = asyncio.Lock()
        self._storage_slots = asyncio.Semaphore(2)
        self._storage_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-storage")
        self._storage_closed = False
        self._speech_turn = False

    @property
    def busy(self):
        return bool(self._active) or self._admin_active

    def stats(self):
        return {"slots": self.backend.slots, "active": len(self._active), "admin_pending": len(self._admins)}

    async def io(self, operation, *args):
        """Keep both running and executor-queued storage calls bounded at two.

        Once submitted, an operation must settle before cancellation releases its
        caller's catalog lock or result lease. Filesystem threads cannot be killed.
        """
        async with self._storage_slots:
            future = asyncio.get_running_loop().run_in_executor(self._storage_executor, operation, *args)
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(future)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                # Opening a result on behalf of a disconnected caller transfers
                # no ownership, so release its lease here before propagating.
                if hasattr(result, "stream") and hasattr(result, "close"):
                    close = asyncio.get_running_loop().run_in_executor(self._storage_executor, result.close)
                    while not close.done():
                        with suppress(asyncio.CancelledError):
                            await asyncio.shield(close)
                    close.result()
                raise asyncio.CancelledError
            return result

    async def storage(self, method, *args):
        return await self.io(getattr(self.store, method), *args)

    async def close_result(self, lease):
        operation = (asyncio.to_thread(lease.close) if self._storage_closed else self.io(lease.close))
        closing = asyncio.create_task(operation)
        while not closing.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(closing)
        closing.result()

    async def cleanup_if_due(self):
        async with self._cleanup_lock:
            now = time.monotonic()
            if self._last_cleanup is None or now - self._last_cleanup >= 60:
                await self.storage("cleanup")
                self._last_cleanup = now

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name="tts-job-scheduler")

    def _require_ready(self):
        if not self.ready or self._stopping:
            raise HTTPException(503, "Model is not ready", headers={"Retry-After": "10"})

    async def enqueue(self, items):
        self._require_ready()
        async with self.catalog_lock:
            self._require_ready()
            await self.cleanup_if_due()
            voices = {voice["id"]: voice for voice in self.engine.voices}
            snapshots = {} if self.backend.requires_snapshots else None
            for item in items:
                if item.get("mode", "preset") != "preset":
                    continue
                voice_id = item.get("voice")
                voice = voices.get(voice_id)
                if voice is None:
                    raise HTTPException(422, "Unknown voice; use GET /v1/voices")
                if item.get("gender") is not None and item["gender"] != voice["gender"]:
                    raise HTTPException(422, "Gender does not match selected voice")
                if snapshots is not None and voice_id not in snapshots:
                    descriptor = await self.io(self.engine.profile_snapshot_descriptor, voice_id)
                    snapshots[voice_id] = await self.storage("pin_voice",
                        voice_id, descriptor["gender"], descriptor["name"],
                        Path(descriptor["path"]), descriptor["sha256"],
                    )
            try:
                group = await self.storage("enqueue", items, snapshots)
            finally:
                self._wake.set()
        return group

    async def speech(self, body):
        group = await self.enqueue([body.model_dump(exclude_none=True)])
        job = await self.wait_job(group["jobs"][0]["id"])
        if job["state"] == "failed":
            self._raise_job_error(job["error"])
        try:
            lease = await self.storage("acquire_result", job["id"])
            try:
                return await self.io(lease.stream.read)
            finally:
                await self.close_result(lease)
        except (KeyError, OSError):
            raise HTTPException(503, "Speech result is unavailable; retry later") from None

    async def wait_job(self, job_id):
        """Wait only for callers that explicitly need a synchronous response."""
        while True:
            if self._stopping or self.failed:
                raise HTTPException(503, "Speech service is stopping; poll the job after restart")
            future = asyncio.get_running_loop().create_future()
            waiters = self._waiters.setdefault(job_id, set())
            waiters.add(future)
            try:
                job = await self.storage("get_job", job_id)
                if job["state"] in {"succeeded", "failed"}:
                    return job
                if self._stopping or self.failed:
                    raise HTTPException(503, "Speech service is stopping; poll the job after restart")
                await future
            finally:
                waiters.discard(future)
                if not waiters:
                    self._waiters.pop(job_id, None)

    async def discard_upload(self, path):
        if path is None:
            return
        operation = (asyncio.to_thread(path.unlink, missing_ok=True) if self._storage_closed
                     else self.io(path.unlink, True))
        cleanup = asyncio.create_task(operation)
        while not cleanup.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(cleanup)
        cleanup.result()

    async def submit_admin(self, operation, payload, *, owned_upload=None):
        # Admission owns the staged upload immediately. Once queued, the actual
        # model operation owns it even if its HTTP caller disconnects or expires.
        try:
            self._require_ready()
            queue_limit = getattr(getattr(self.admin_worker, "settings", None), "queue_size", 0)
            if queue_limit and len(self._admins) >= queue_limit:
                raise HTTPException(429, "Admin queue is full; retry later", headers={"Retry-After": "5"})
            future = asyncio.get_running_loop().create_future()
            future.add_done_callback(_consume_exception)
            self._admins.append(_AdminRequest(operation, payload, future, owned_upload))
            self._wake.set()
        except BaseException:
            await self.discard_upload(owned_upload)
            raise
        # Persist the barrier even when its caller disconnects or times out.
        return await asyncio.shield(future)

    async def stop(self):
        self.ready = False
        self._stopping = True
        self._wake.set()
        self._notify_all()
        if self.task is not None:
            await asyncio.shield(self.task)
        if not self._storage_closed:
            self._storage_closed = True
            # No blocking executor shutdown runs on the ASGI event loop.
            await asyncio.to_thread(self._storage_executor.shutdown, wait=True)

    def _notify(self, job_id):
        for future in tuple(self._waiters.get(job_id, ())):
            if not future.done():
                future.set_result(None)

    def _notify_all(self):
        for job_id in tuple(self._waiters):
            self._notify(job_id)

    async def _run(self):
        backend_started = False
        try:
            while not self.admin_worker.ready:
                if self.admin_worker.failed:
                    raise RuntimeError("Parent model initialization failed")
                if self._stopping:
                    return
                await asyncio.sleep(0.02)
            backend_started = True
            await self.backend.start()
            await self.storage("recover")
            await self.cleanup_if_due()
            if self.backend.slots < 1:
                raise RuntimeError("Speech backend has no available slots")
            self.ready = not self._stopping
            while not self._stopping:
                self._wake.clear()
                for slot, task in tuple(self._active.items()):
                    if task.done():
                        task.result()
                        del self._active[slot]
                if self._recover_backend:
                    if not self._active:
                        await self.backend.recover()
                        self._recover_backend = False
                        self._wake.set()
                elif self._admins and not self._speech_turn:
                    if not self._active:
                        await self._run_admin(self._admins.popleft())
                        self._speech_turn = True
                        self._wake.set()
                else:
                    for slot in range(self.backend.slots):
                        if slot in self._active:
                            continue
                        job = await self.storage("claim", slot)
                        if job is None:
                            break
                        if self._stopping:
                            await self.storage("requeue", job["id"])
                            break
                        self._active[slot] = asyncio.create_task(
                            self._execute_job(job), name=f"tts-job-{job['id']}",
                        )
                    self._speech_turn = False
                    if self._admins and not self._active:
                        self._wake.set()
                await self._wake.wait()
        except Exception:
            self.failed = True
            self.ready = False
            # Never log exception text or submitted payloads.
            logger.error("Speech scheduler is unavailable", exc_info=False)
            self._notify_all()
        finally:
            self.ready = False
            while self._admins:
                request = self._admins.popleft()
                if not request.future.done():
                    request.future.set_exception(HTTPException(503, "Speech service is stopping"))
                await self.discard_upload(request.owned_upload)
            if self._active:
                # Give active work its execution window, then let the backend
                # terminate its processes before cancelling our coroutine shells.
                await asyncio.wait(tuple(self._active.values()), timeout=self.request_timeout or 30)
            if backend_started:
                await self.backend.stop()
            for task in self._active.values():
                if not task.done():
                    task.cancel()
            if self._active:
                await asyncio.gather(*self._active.values(), return_exceptions=True)
            self._active.clear()
            self._notify_all()

    async def _run_admin(self, request):
        self._admin_active = True
        try:
            async with self.catalog_lock:
                try:
                    result = await self.admin_worker.submit(request.operation, request.payload)
                except Exception as error:
                    if not request.future.done():
                        request.future.set_exception(self._admin_error(error))
                else:
                    if not request.future.done():
                        request.future.set_result(result)
                finally:
                    # SpeechWorker.submit can time out while its executor thread
                    # still owns the GPU. Its queue owns the real completion signal.
                    completion = asyncio.create_task(self.admin_worker.queue.join())
                    cancelled = False
                    while not completion.done():
                        try:
                            await asyncio.shield(completion)
                        except asyncio.CancelledError:
                            cancelled = True
                    completion.result()
                    if cancelled:
                        raise asyncio.CancelledError
        finally:
            await self.discard_upload(request.owned_upload)
            self._admin_active = False

    async def _execute_job(self, job):
        execution = asyncio.create_task(self.backend.run(job))
        try:
            try:
                if getattr(self.backend, "handles_timeout", False):
                    result = await asyncio.shield(execution)
                else:
                    result = await asyncio.wait_for(asyncio.shield(execution), self.request_timeout or None)
            except TimeoutError:
                await self.storage("fail", job["id"], "execution_timeout", "Speech execution timed out")
                self._notify(job["id"])
                # Timeout changes the public state, never the actual slot owner.
                with suppress(Exception):
                    await asyncio.shield(execution)
            except Exception as error:
                if getattr(error, "code", None) == 504 or getattr(error, "status_code", None) == 504:
                    self._recover_backend = self._recover_backend or bool(getattr(error, "recoverable", False))
                    await self.storage("fail", job["id"], "execution_timeout", "Speech execution timed out")
                    self._notify(job["id"])
                elif isinstance(error, BrokenProcessPool) or getattr(error, "recoverable", False):
                    self._recover_backend = True
                    if job["attempts"] < 2:
                        await self.storage("requeue", job["id"])
                        self._notify(job["id"])
                    else:
                        await self.storage("fail", job["id"], "worker_failed", "Speech worker failed after retry")
                        self._notify(job["id"])
                else:
                    code, detail = self._failure(error)
                    await self.storage("fail", job["id"], code, detail)
                    self._notify(job["id"])
            else:
                try:
                    await self.storage("complete", job["id"], result)
                except ValueError:
                    await self.storage("fail", job["id"], "invalid_artifact", "Speech worker returned an invalid audio result")
                self._notify(job["id"])
        finally:
            if not execution.done():
                execution.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await execution
            self._wake.set()

    @staticmethod
    def _failure(error):
        if isinstance(error, VoiceNotFoundError):
            return "voice_not_found", "Voice was not found"
        if isinstance(error, ValueError):
            return "invalid_generation", "Invalid reference audio or generation settings; see /v1/capabilities"
        return "generation_failed", "Speech generation failed; retry later"

    @staticmethod
    def _raise_job_error(error):
        statuses = {"execution_timeout": 504, "voice_not_found": 404, "invalid_generation": 422}
        raise HTTPException(statuses.get(error["code"], 503), error["detail"])

    @staticmethod
    def _admin_error(error):
        if isinstance(error, HTTPException):
            return error
        if isinstance(error, VoiceNotFoundError):
            return HTTPException(404, "Voice was not found")
        if isinstance(error, ProtectedVoiceError):
            return HTTPException(403, "Built-in voices cannot be deleted")
        if isinstance(error, VoiceDeletionError):
            return HTTPException(503, "Voice deletion could not complete; retry DELETE with the same voice ID")
        if isinstance(error, TimeoutError):
            return HTTPException(504, "Speech request timed out; retry later")
        if isinstance(error, ValueError):
            return HTTPException(422, "Invalid reference audio or generation settings; see /v1/capabilities")
        return HTTPException(503, "Speech generation failed; retry later")
