"""Atomic results and resource boundaries without loading a GPU model."""

import asyncio
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
import io
from pathlib import Path
import sys
import types
import wave

import pytest

from inference_pool import ExecutionTimeout, ProcessSpeechBackend, _write_result
from jobs import JobStore


def audio(rate=24000, seconds=0.1):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x10\x01" * round(rate * seconds))
    return output.getvalue()


def test_atomic_worker_result_is_accepted_and_recovered_by_store(tmp_path):
    store = JobStore(tmp_path)
    group = store.enqueue([{"text": "Salom", "voice": "female_1", "key": "assistant.welcome"}])
    job = store.claim(0)
    result = _write_result(audio(), job["id"], store.results_dir)
    assert not list(store.results_dir.glob("*.tmp"))
    # A crash after rename and before the SQL commit publishes the one valid file.
    store.close()
    restored = JobStore(tmp_path)
    assert restored.recover()["succeeded"] == 1
    assert restored.get_job(group["id"], job["id"])["state"] == "succeeded"
    assert restored.get_result_path(job["id"]).read_bytes() == audio()
    assert result["worker_pid"] > 0
    restored.close()


@pytest.mark.parametrize("identity", ["../escape", "a" * 31, "A" * 32, "b" * 33])
def test_worker_rejects_result_path_injection_before_creating_storage(tmp_path, identity):
    directory = tmp_path / "results"
    with pytest.raises(ValueError, match="identity"):
        _write_result(audio(), identity, directory)
    assert not directory.exists()


def test_invalid_audio_cannot_replace_completed_result(tmp_path):
    job_id = "a" * 32
    expected = _write_result(audio(), job_id, tmp_path)
    with pytest.raises(ValueError, match="format"):
        _write_result(audio(rate=16000), job_id, tmp_path)
    assert Path(expected["path"]).read_bytes() == audio()


def test_worker_can_publish_audio_beyond_the_old_ten_minute_cap(tmp_path):
    expected = _write_result(audio(seconds=601), "a" * 32, tmp_path)
    assert expected["duration"] == 601
    assert expected["bytes"] > 28800044


@pytest.mark.parametrize("timeout", [0, -1])
def test_disabled_watchdog_waits_for_the_actual_worker_result(tmp_path, timeout):
    async def check():
        backend = ProcessSpeechBackend({"device": "cpu"}, tmp_path, execution_timeout=timeout)
        pending = Future()
        backend._pool = types.SimpleNamespace(submit=lambda *args: pending)
        execution = asyncio.create_task(backend.run({"id": "a" * 32}))
        await asyncio.sleep(0.02)
        assert not execution.done()
        pending.set_result({"path": "result.wav"})
        assert await execution == {"path": "result.wav"}

    asyncio.run(check())


def test_worker_partial_filename_is_owned_and_cleaned_by_recovery(tmp_path):
    store = JobStore(tmp_path)
    store.enqueue([{"text": "Salom", "voice": "female_1"}])
    job = store.claim(0)
    partial = store.results_dir / f"{job['id']}.wav.123.tmp"
    partial.write_bytes(b"incomplete")
    store.requeue(job["id"])
    assert not partial.exists() and store.get_job(job["id"])["state"] == "queued"
    store.close()


def test_host_and_gpu_reserves_reduce_effective_workers(tmp_path, monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.as_posix() == "/sys/fs/cgroup/memory.max":
            return str(16 * 1024 ** 3)
        if path.as_posix() == "/sys/fs/cgroup/memory.current":
            return str(4 * 1024 ** 3)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(
        mem_get_info=lambda device: (24 * 1024 ** 3, 32 * 1024 ** 3))))
    backend = ProcessSpeechBackend({"device": "cuda:0"}, tmp_path, slots=3)
    backend._memory_budget()
    assert backend.requested_slots == 3 and backend.slots == 2
    assert backend.capacity_reason == "container memory reserve"
    assert backend._pool is None


def test_insufficient_memory_never_starts_a_process(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(
        mem_get_info=lambda device: (3 * 1024 ** 3, 32 * 1024 ** 3))))
    backend = ProcessSpeechBackend({"device": "cuda:0"}, tmp_path)
    with pytest.raises(RuntimeError, match="reserved memory"):
        backend._memory_budget()
    assert backend._pool is None


def test_execution_watchdog_consumes_failed_future_and_stops_owned_pool(tmp_path, monkeypatch):
    async def check():
        backend = ProcessSpeechBackend({"device": "cpu"}, tmp_path, execution_timeout=0.01)
        pending = Future()
        pool = types.SimpleNamespace(submit=lambda *args: pending)
        backend._pool = pool
        disposed = []

        def dispose():
            disposed.append(True)
            backend._pool = None
            pending.set_exception(BrokenProcessPool("terminated"))

        monkeypatch.setattr(backend, "_dispose_pool", dispose)
        loop = asyncio.get_running_loop()
        errors = []
        loop.set_exception_handler(lambda *args: errors.append(args))
        with pytest.raises(ExecutionTimeout) as caught:
            await backend.run({"id": "a" * 32})
        assert caught.value.code == 504 and caught.value.recoverable
        await asyncio.sleep(0.01)
        assert disposed == [True] and backend._pool is None and not errors

    asyncio.run(check())
