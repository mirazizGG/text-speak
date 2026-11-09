"""Durable scheduling tests with controllable inference, without model weights."""

import asyncio
from concurrent.futures.process import BrokenProcessPool
import hashlib
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import wave

from fastapi import HTTPException
import pytest

from job_service import JobService
from jobs import JobStore


class FakeEngine:
    voices = [{"id": "female_1", "gender": "female", "name": "Test voice"}]


class FakeAdmin:
    ready = True
    failed = False

    def __init__(self, backend, timeout=None):
        self.backend = backend
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.actual = None
        self.calls = []

    async def submit(self, operation, payload):
        self.calls.append((operation, payload))
        self.queue.put_nowait((operation, payload))
        self.actual = asyncio.create_task(self._execute())
        if self.timeout is None:
            return await asyncio.shield(self.actual)
        try:
            return await asyncio.wait_for(asyncio.shield(self.actual), self.timeout)
        except TimeoutError:
            raise HTTPException(504, "Admin request timed out") from None

    async def _execute(self):
        try:
            assert self.backend.active == 0, "Admin overlapped speech inference"
            self.started.set()
            await self.release.wait()
            return {"ok": True}
        finally:
            self.queue.task_done()


def make_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * 240)
    return output.getvalue()


class FakeBackend:
    requires_snapshots = False
    handles_timeout = False

    def __init__(self, slots=2):
        self.slots = slots
        self.active = 0
        self.maximum = 0
        self.started = []
        self.jobs = []
        self.finished = []
        self.gates = {}
        self.errors = {}
        self.recoveries = 0
        self.closed = False
        self.loads = 0

    async def start(self):
        self.loads += 1

    async def stop(self):
        self.closed = True

    async def recover(self):
        assert self.active == 0
        self.recoveries += 1

    async def run(self, job):
        text = job["payload"]["text"]
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.started.append(text)
        self.jobs.append(job)
        try:
            if text in self.gates:
                await self.gates[text].wait()
            if text in self.errors:
                raise self.errors[text]()
            data = make_wav()
            path = Path(job["artifact_path"])
            path.write_bytes(data)
            self.finished.append(text)
            return {"path": str(path), "bytes": len(data), "sample_rate": 24000,
                    "duration": 0.01, "wav_sha256": hashlib.sha256(data).hexdigest()}
        finally:
            self.active -= 1


def items(count):
    return [{"text": str(index), "mode": "preset", "voice": "female_1", "language": "uz",
             "generation": {"num_step": 32}} for index in range(count)]


async def until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def service_for(tmp_path, backend=None, timeout=3):
    backend = backend or FakeBackend()
    store = JobStore(tmp_path, min_free_bytes=0)
    admin = FakeAdmin(backend)
    service = JobService(FakeEngine(), admin, backend, store, request_timeout=timeout)
    await service.start()
    await until(lambda: service.ready or service.failed)
    assert service.ready
    return service, store, admin, backend


def test_accepts_100_fifo_jobs_and_refills_each_slot_independently(tmp_path):
    async def scenario():
        backend = FakeBackend()
        backend.gates = {"0": asyncio.Event(), "1": asyncio.Event(), "2": asyncio.Event()}
        service, store, _, _ = await service_for(tmp_path, backend)
        try:
            group = await service.enqueue(items(100))
            assert len(group["jobs"]) == 100
            assert store.stats()["queued"] == 100
            await until(lambda: len(backend.started) == 2)
            assert backend.started == ["0", "1"]
            assert backend.maximum == 2
            backend.gates["0"].set()
            await until(lambda: len(backend.started) == 3)
            assert backend.started == ["0", "1", "2"]
            first, second, third = [store.get_job(job["id"]) for job in group["jobs"][:3]]
            assert first["state"] == "succeeded"
            assert second["state"] == third["state"] == "running"
            assert store.get_result_path(first["id"]).read_bytes() == make_wav()
            assert "download_url" in first
            assert "download_url" not in second or second["download_url"] is None
            backend.gates["1"].set()
            backend.gates["2"].set()
            await until(lambda: store.stats()["succeeded"] == 100)
            assert backend.started == [str(index) for index in range(100)]
            assert backend.maximum == 2
            assert all(job["payload"]["generation"]["num_step"] == 32 for job in backend.jobs)
            assert all(job["state"] == "succeeded" for job in store.get_group(group["id"])["jobs"])
        finally:
            for gate in backend.gates.values():
                gate.set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_admin_has_priority_and_timeout_retains_gpu_barrier(tmp_path):
    async def scenario():
        backend = FakeBackend()
        backend.gates = {"0": asyncio.Event(), "1": asyncio.Event()}
        service, store, admin, _ = await service_for(tmp_path, backend)
        admin.timeout = 0.03
        try:
            group = await service.enqueue(items(3))
            await until(lambda: len(backend.started) == 2)
            request = asyncio.create_task(service.submit_admin("save_voice", {}))
            await until(lambda: service.stats()["admin_pending"] == 1)
            backend.gates["0"].set()
            await until(lambda: backend.finished == ["0"])
            assert backend.started == ["0", "1"]
            assert not admin.started.is_set()
            backend.gates["1"].set()
            await admin.started.wait()
            with pytest.raises(HTTPException) as raised:
                await request
            assert raised.value.status_code == 504
            assert backend.started == ["0", "1"]
            assert service.catalog_lock.locked()
            assert not admin.actual.done()
            admin.release.set()
            await until(lambda: store.stats()["succeeded"] == 3)
            assert backend.started == ["0", "1", "2"]
            assert store.get_group(group["id"])["state"] == "succeeded"
        finally:
            for gate in backend.gates.values():
                gate.set()
            admin.release.set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_admin_waiting_limit_does_not_limit_durable_speech_admission(tmp_path):
    async def scenario():
        backend = FakeBackend()
        backend.gates = {"0": asyncio.Event(), "1": asyncio.Event()}
        service, store, admin, _ = await service_for(tmp_path, backend)
        admin.settings = SimpleNamespace(queue_size=4)
        requests = []
        try:
            await service.enqueue(items(2))
            await until(lambda: len(backend.started) == 2)
            requests = [asyncio.create_task(service.submit_admin("save_voice", {"index": index}))
                        for index in range(4)]
            await until(lambda: service.stats()["admin_pending"] == 4)
            rejected_upload = tmp_path / "rejected-upload"
            rejected_upload.write_bytes(b"reference")
            with pytest.raises(HTTPException) as raised:
                await service.submit_admin("save_voice", {"index": 4}, owned_upload=rejected_upload)
            assert raised.value.status_code == 429
            assert raised.value.headers["Retry-After"] == "5"
            assert not rejected_upload.exists()
            group = await service.enqueue(items(50))
            assert len(group["jobs"]) == 50
            assert store.stats()["queued"] == 50
            assert backend.started == ["0", "1"]
            assert not admin.started.is_set()
            for gate in backend.gates.values():
                gate.set()
            admin.release.set()
            assert await asyncio.gather(*requests) == [{"ok": True}] * 4
            await until(lambda: store.stats()["succeeded"] == 52)
            assert backend.maximum == 2
        finally:
            for gate in backend.gates.values():
                gate.set()
            admin.release.set()
            await asyncio.gather(*requests, return_exceptions=True)
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_disabled_admin_queue_limit_accepts_all_waiters_without_overlapping_gpu_work(tmp_path):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.gates = {"0": asyncio.Event()}
        service, store, admin, _ = await service_for(tmp_path, backend, timeout=0)
        admin.settings = SimpleNamespace(queue_size=0)
        requests = []
        try:
            await service.enqueue(items(1))
            await until(lambda: backend.active == 1)
            requests = [asyncio.create_task(service.submit_admin("save_voice", {"index": index}))
                        for index in range(12)]
            await until(lambda: service.stats()["admin_pending"] == 12)
            assert not admin.started.is_set()
            backend.gates["0"].set()
            admin.release.set()
            assert await asyncio.gather(*requests) == [{"ok": True}] * 12
            assert len(admin.calls) == 12
            assert backend.maximum == 1
        finally:
            backend.gates["0"].set()
            admin.release.set()
            await asyncio.gather(*requests, return_exceptions=True)
            await service.stop()
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_caller", [True, False])
def test_owned_upload_survives_disconnect_or_timeout_until_actual_admin_completion(tmp_path, cancel_caller):
    async def scenario():
        service, store, admin, _ = await service_for(tmp_path)
        if not cancel_caller:
            admin.timeout = 0.02
        upload = tmp_path / "active-upload"
        upload.write_bytes(b"reference")
        request = asyncio.create_task(service.submit_admin("save_voice", {"audio": upload}, owned_upload=upload))
        try:
            await until(admin.started.is_set)
            if cancel_caller:
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
            else:
                with pytest.raises(HTTPException) as raised:
                    await request
                assert raised.value.status_code == 504
            assert upload.read_bytes() == b"reference"
            assert not admin.actual.done()
            admin.release.set()
            await until(lambda: not upload.exists())
            assert admin.actual.done()
        finally:
            admin.release.set()
            await asyncio.gather(request, return_exceptions=True)
            await service.stop()
            store.close()

    asyncio.run(scenario())


def test_shutdown_discards_queued_owned_upload_before_backend_finishes(tmp_path):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.gates = {"0": asyncio.Event()}
        service, store, admin, _ = await service_for(tmp_path, backend)
        upload = tmp_path / "queued-upload"
        upload.write_bytes(b"reference")
        request = None
        stopping = None
        try:
            await service.enqueue(items(1))
            await until(lambda: backend.active == 1)
            request = asyncio.create_task(service.submit_admin("save_voice", {"audio": upload}, owned_upload=upload))
            await until(lambda: service.stats()["admin_pending"] == 1)
            stopping = asyncio.create_task(service.stop())
            with pytest.raises(HTTPException) as raised:
                await request
            assert raised.value.status_code == 503
            await until(lambda: not upload.exists())
            assert not admin.started.is_set()
            assert backend.active == 1
            backend.gates["0"].set()
            await stopping
        finally:
            backend.gates["0"].set()
            admin.release.set()
            if request is not None:
                await asyncio.gather(request, return_exceptions=True)
            if stopping is not None:
                await stopping
            await service.stop()
            store.close()

    asyncio.run(scenario())


def test_stopped_service_rejects_and_removes_owned_upload_after_storage_shutdown(tmp_path):
    async def scenario():
        service, store, _, _ = await service_for(tmp_path)
        await service.stop()
        upload = tmp_path / "rejected-upload"
        upload.write_bytes(b"reference")
        try:
            with pytest.raises(HTTPException) as raised:
                await service.submit_admin("save_voice", {"audio": upload}, owned_upload=upload)
            assert raised.value.status_code == 503
            assert not upload.exists()
        finally:
            store.close()

    asyncio.run(scenario())


def test_disabled_execution_timeout_preserves_running_and_queued_jobs(tmp_path):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.gates = {"0": asyncio.Event()}
        service, store, _, _ = await service_for(tmp_path, backend, timeout=0)
        try:
            group = await service.enqueue(items(2))
            await until(lambda: backend.started == ["0"])
            await asyncio.sleep(0.05)
            assert store.get_job(group["jobs"][0]["id"])["state"] == "running"
            assert store.get_job(group["jobs"][1]["id"])["state"] == "queued"
            backend.gates["0"].set()
            await until(lambda: store.stats()["succeeded"] == 2)
            assert backend.maximum == 1
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()

    asyncio.run(scenario())


def test_disabled_execution_timeout_keeps_a_bounded_shutdown_grace(tmp_path, monkeypatch):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.gates = {"0": asyncio.Event()}
        service, store, _, _ = await service_for(tmp_path, backend, timeout=0)
        original_wait = asyncio.wait
        timeouts = []

        async def shutdown_wait(tasks, *, timeout):
            timeouts.append(timeout)
            return await original_wait(tasks, timeout=0)

        try:
            await service.enqueue(items(1))
            await until(lambda: backend.started == ["0"])
            monkeypatch.setattr(asyncio, "wait", shutdown_wait)
            await service.stop()
            assert timeouts == [30]
            assert backend.closed
            assert backend.active == 0
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()

    asyncio.run(scenario())


def test_cancelling_legacy_caller_leaves_accepted_queued_job_durable(tmp_path):
    async def scenario():
        from api import SpeechRequest

        backend = FakeBackend(slots=1)
        backend.gates["0"] = asyncio.Event()
        service, store, _, _ = await service_for(tmp_path, backend)
        try:
            await service.enqueue(items(1))
            await until(lambda: backend.started == ["0"])
            caller = asyncio.create_task(service.speech(SpeechRequest(**items(2)[1])))
            await until(lambda: store.stats()["queued"] == 1)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert store.stats()["running"] == 1
            assert store.stats()["queued"] == 1
            assert service._waiters == {}
            backend.gates["0"].set()
            await until(lambda: store.stats()["succeeded"] == 2)
            assert backend.finished == ["0", "1"]
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_execution_timeout_does_not_release_running_thread_slot(tmp_path):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.gates["0"] = asyncio.Event()
        service, store, _, _ = await service_for(tmp_path, backend, timeout=0.05)
        try:
            group = await service.enqueue(items(2))
            await until(lambda: store.stats()["failed"] == 1)
            failed = store.get_job(group["jobs"][0]["id"])
            assert failed["error"]["code"] == "execution_timeout"
            assert backend.started == ["0"]
            assert backend.active == 1
            backend.gates["0"].set()
            await until(lambda: store.stats()["succeeded"] == 1)
            assert backend.started == ["0", "1"]
            assert backend.maximum == 1
            assert store.get_job(failed["id"])["state"] == "failed"
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("error_kind", ["process", "tensorrt"])
def test_broken_worker_retries_twice_and_other_jobs_continue(tmp_path, error_kind):
    from engine import WorkerRestartRequired

    async def scenario():
        backend = FakeBackend()
        error_class = BrokenProcessPool if error_kind == "process" else WorkerRestartRequired
        backend.errors["0"] = lambda: error_class("PRIVATE submitted input")
        backend.gates["1"] = asyncio.Event()
        service, store, _, _ = await service_for(tmp_path, backend)
        try:
            group = await service.enqueue(items(4))
            await until(lambda: service._recover_backend and len(backend.started) == 2)
            assert backend.recoveries == 0
            assert backend.started == ["0", "1"]
            backend.gates["1"].set()
            await until(lambda: store.stats()["failed"] == 1 and store.stats()["succeeded"] == 3)
            bad = store.get_job(group["jobs"][0]["id"])
            assert bad["error"] == {"code": "worker_failed", "detail": "Speech worker failed after retry"}
            assert backend.started.count("0") == 2
            assert backend.recoveries == 2
            assert backend.maximum <= 2
            assert not service.failed
        finally:
            backend.gates["1"].set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_queued_jobs_survive_clean_stop_and_restart(tmp_path):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.gates["0"] = asyncio.Event()
        service, store, _, _ = await service_for(tmp_path, backend)
        group = await service.enqueue(items(3))
        await until(lambda: backend.started == ["0"])
        stopping = asyncio.create_task(service.stop())
        await until(lambda: not service.ready)
        backend.gates["0"].set()
        await stopping
        assert store.stats()["queued"] == 2
        store.close()
        second, reopened, _, replacement = await service_for(tmp_path, FakeBackend(slots=1))
        try:
            await until(lambda: reopened.stats()["succeeded"] == 3)
            assert replacement.started == ["1", "2"]
            assert reopened.get_group(group["id"])["state"] == "succeeded"
        finally:
            await second.stop()
            reopened.close()
    asyncio.run(scenario())


def test_crash_recovery_requeues_inflight_job(tmp_path):
    store = JobStore(tmp_path, min_free_bytes=0)
    group = store.enqueue(items(2), None)
    interrupted = store.claim(0)
    assert interrupted["attempts"] == 1
    store.close()

    async def scenario():
        service, reopened, _, backend = await service_for(tmp_path)
        try:
            await until(lambda: reopened.stats()["succeeded"] == 2)
            assert backend.started == ["0", "1"]
            assert reopened.get_group(group["id"])["state"] == "succeeded"
        finally:
            await service.stop()
            reopened.close()
    asyncio.run(scenario())


def test_backend_watchdog_owns_execution_timeout_and_recovery(tmp_path):
    class ExecutionTimeout(RuntimeError):
        recoverable = True
        code = 504

    async def scenario():
        backend = FakeBackend(slots=1)
        backend.handles_timeout = True
        backend.gates["0"] = asyncio.Event()
        backend.errors["0"] = ExecutionTimeout
        service, store, _, _ = await service_for(tmp_path, backend, timeout=0.02)
        try:
            group = await service.enqueue(items(2))
            await until(lambda: backend.started == ["0"])
            await asyncio.sleep(0.04)
            assert store.stats()["running"] == 1
            backend.gates["0"].set()
            await until(lambda: store.stats()["failed"] == 1 and store.stats()["succeeded"] == 1)
            assert store.get_job(group["jobs"][0]["id"])["error"]["code"] == "execution_timeout"
            assert backend.started == ["0", "1"]
            assert backend.recoveries == 1
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_waiting_jobs_have_no_queue_expiry(tmp_path):
    async def scenario():
        backend = FakeBackend(slots=1)
        backend.handles_timeout = True
        backend.gates["0"] = asyncio.Event()
        service, store, _, _ = await service_for(tmp_path, backend, timeout=0.02)
        try:
            group = await service.enqueue(items(2))
            await until(lambda: backend.started == ["0"])
            await asyncio.sleep(0.04)
            assert store.get_job(group["jobs"][1]["id"])["state"] == "queued"
            backend.gates["0"].set()
            await until(lambda: store.stats()["succeeded"] == 2)
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_voice_profile_is_pinned_before_catalog_can_change(tmp_path):
    async def scenario():
        source = tmp_path / "prompt.pt"
        source.write_bytes(b"original voice prompt")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()

        class SnapshotEngine(FakeEngine):
            def profile_snapshot_descriptor(self, voice_id):
                return {"path": source, "sha256": digest, "gender": "female", "name": "Test voice"}

        backend = FakeBackend(slots=1)
        backend.requires_snapshots = True
        backend.gates["0"] = asyncio.Event()
        service, store, _, _ = await service_for(tmp_path / "jobs", backend)
        service.engine = SnapshotEngine()
        try:
            group = await service.enqueue(items(1))
            source.unlink()
            service.engine.voices = []
            await until(lambda: backend.started == ["0"])
            assert backend.jobs[0]["snapshot"] == {
                "voice_id": "female_1", "gender": "female", "name": "Test voice", "sha256": digest,
            }
            assert (store.snapshots_dir / f"{digest}.pt").read_bytes() == b"original voice prompt"
            backend.gates["0"].set()
            await until(lambda: store.get_group(group["id"])["state"] == "succeeded")
            with pytest.raises(HTTPException) as error:
                await service.enqueue(items(1))
            assert error.value.status_code == 422
        finally:
            backend.gates["0"].set()
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_start_is_nonblocking_and_waits_for_parent_load(tmp_path):
    async def scenario():
        backend = FakeBackend()
        admin = FakeAdmin(backend)
        admin.ready = False
        store = JobStore(tmp_path, min_free_bytes=0)
        service = JobService(FakeEngine(), admin, backend, store)
        try:
            await service.start()
            assert not service.ready
            assert backend.loads == 0
            await asyncio.sleep(0.025)
            assert backend.loads == 0
            admin.ready = True
            await until(lambda: service.ready)
            assert backend.loads == 1
        finally:
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_one_admin_barrier_is_followed_by_speech_before_next_admin(tmp_path):
    async def scenario():
        backend = FakeBackend()
        backend.gates = {str(index): asyncio.Event() for index in range(4)}
        service, store, admin, _ = await service_for(tmp_path, backend)
        requests = []
        try:
            await service.enqueue(items(4))
            await until(lambda: backend.started == ["0", "1"])
            requests = [asyncio.create_task(service.submit_admin("save_voice", {"index": index}))
                        for index in range(4)]
            await until(lambda: service.stats()["admin_pending"] == 4)
            admin.release.set()
            backend.gates["0"].set()
            backend.gates["1"].set()
            await until(lambda: backend.started == ["0", "1", "2", "3"])
            assert len(admin.calls) == 1
            assert service.stats()["admin_pending"] == 3
            backend.gates["2"].set()
            backend.gates["3"].set()
            await asyncio.gather(*requests)
            assert len(admin.calls) == 4
            await until(lambda: store.stats()["succeeded"] == 4)
        finally:
            for gate in backend.gates.values():
                gate.set()
            admin.release.set()
            await asyncio.gather(*requests, return_exceptions=True)
            await service.stop()
            store.close()
    asyncio.run(scenario())


def test_slow_metadata_storage_keeps_loop_and_other_worker_refill_responsive(tmp_path, monkeypatch):
    async def scenario():
        backend = FakeBackend()
        backend.gates = {str(index): asyncio.Event() for index in range(3)}
        service, store, _, _ = await service_for(tmp_path, backend)
        entered = threading.Event()
        release = threading.Event()
        reader = None
        original = store.get_group

        def slow_group(group_id):
            entered.set()
            assert release.wait(timeout=3)
            return original(group_id)

        try:
            group = await service.enqueue(items(3))
            await until(lambda: backend.started == ["0", "1"])
            monkeypatch.setattr(store, "get_group", slow_group)
            reader = asyncio.create_task(service.storage("get_group", group["id"]))
            await until(entered.is_set)
            assert not reader.done()
            backend.gates["0"].set()
            await until(lambda: backend.started == ["0", "1", "2"], timeout=0.5)
            assert not reader.done()
            assert backend.active == 2
        finally:
            release.set()
            for gate in backend.gates.values():
                gate.set()
            if reader:
                await reader
            await service.stop()
            store.close()
    asyncio.run(scenario())
