"""Private spawned speech engines; the coordinator remains the profile owner."""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import hashlib
import io
import multiprocessing
import os
from pathlib import Path
import re
import time
import wave


_MODEL = None
_SNAPSHOTS = None
_RESULTS = None
_GIB = 1024 ** 3


class ExecutionTimeout(RuntimeError):
    code = 504
    recoverable = True


def _write_result(data, job_id, results_dir):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("Invalid result identity")
    directory = Path(results_dir)
    if directory.is_symlink():
        raise ValueError("Invalid result storage")
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{job_id}.wav"
    temporary = directory / f"{job_id}.wav.{os.getpid()}.tmp"
    if final.is_symlink() or temporary.is_symlink():
        raise ValueError("Invalid result storage")
    with wave.open(io.BytesIO(data), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (24000, 1, 2):
            raise ValueError("Unexpected generated audio format")
        duration = audio.getnframes() / audio.getframerate()
        if duration <= 0:
            raise ValueError("Generated audio is empty")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(final), "bytes": len(data), "sample_rate": 24000,
            "duration": duration, "wav_sha256": hashlib.sha256(data).hexdigest(),
            "worker_pid": os.getpid()}


def _initialize_process(config, snapshots_dir, results_dir, ready, gate, load_lock):
    global _MODEL, _SNAPSHOTS, _RESULTS
    try:
        # CUDA is initialized only inside a fresh spawned interpreter. Loading
        # models sequentially avoids multiplying temporary startup allocations.
        with load_lock:
            from engine import OmniVoiceEngine

            _MODEL = OmniVoiceEngine(**{**config, "profile_writer": False})
            _MODEL.load()
            _SNAPSHOTS, _RESULTS = Path(snapshots_dir), Path(results_dir)
        ready.put({"pid": os.getpid(), "ready": True})
        if not gate.wait(180):
            raise RuntimeError("Speech process startup was not acknowledged")
    except Exception:
        ready.put({"pid": os.getpid(), "ready": False})
        raise RuntimeError("Speech process initialization failed") from None


def _identity():
    return os.getpid()


def _synthesize_job(job):
    if _MODEL is None:
        raise RuntimeError("Speech process is not initialized")
    snapshot = job.get("snapshot")
    snapshot_path = None
    if snapshot is not None:
        digest = snapshot.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid queued voice identity")
        snapshot_path = _SNAPSHOTS / f"{digest}.pt"
    try:
        data = _MODEL.execute_pinned_speech(job["payload"], snapshot_path, snapshot)
    except Exception as exc:
        # A failed CUDA RNG restore can replace the model's original exception.
        # Preserve recovery at the process boundary, outside all RNG contexts.
        if (getattr(getattr(_MODEL, "_trt_backend", None), "failed", False)
                or getattr(getattr(_MODEL, "_flashinfer_backend", None), "failed", False)):
            from engine import WorkerRestartRequired

            raise WorkerRestartRequired("OmniVoice inference failed.") from exc
        raise
    return _write_result(data, job["id"], _RESULTS)


class ProcessSpeechBackend:
    """No more than ``slots`` jobs are submitted to this executor at once."""

    requires_snapshots = True
    handles_timeout = True

    def __init__(self, engine_config, jobs_root, slots=2, execution_timeout=0):
        if not 1 <= slots <= 8:
            raise ValueError("Speech process count must be between 1 and 8")
        self.config = dict(engine_config)
        self.root = Path(jobs_root)
        self.requested_slots = slots
        self.slots = slots
        self.execution_timeout = execution_timeout
        self.capacity_reason = None
        self.worker_pids = []
        self._pool = None
        self._reset_lock = asyncio.Lock()
        self._ready_queue = None

    def _memory_budget(self):
        # Conservative admission estimates include long-output CPU buffers and
        # headroom for the profile owner's lazily loaded ASR. Real deployment
        # measurements still determine the configured process count.
        effective = self.requested_slots
        reasons = []
        try:
            limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            try:
                memory_stat = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
                # Clean inactive file cache is reclaimable, as in Docker's own
                # usage reporting; never count it as live model/ASR memory.
                used -= min(used, int(memory_stat.get("inactive_file", "0")))
            except (OSError, ValueError):
                pass
            if limit != "max":
                per_worker = 4.5 if self.config.get("acceleration") == "tensorrt" else 3.25
                host_slots = max(0, (int(limit) - used - 4 * _GIB) // int(per_worker * _GIB))
                effective = min(effective, host_slots)
                if host_slots < self.requested_slots:
                    reasons.append("container memory reserve")
        except (OSError, ValueError):
            pass
        if self.config.get("device", "cuda:0").startswith("cuda"):
            import torch

            free, _ = torch.cuda.mem_get_info(self.config.get("device", "cuda:0"))
            per_worker = (6 if self.config.get("acceleration") == "tensorrt" else
                          5 if self.config.get("flashinfer_graphs") else 4)
            gpu_slots = max(0, (free - 4 * _GIB) // (per_worker * _GIB))
            effective = min(effective, gpu_slots)
            if gpu_slots < self.requested_slots:
                reasons.append("GPU memory reserve")
        if effective < 1:
            raise RuntimeError("Not enough reserved memory to start a speech process")
        self.slots = int(effective)
        self.capacity_reason = ", ".join(reasons) or None

    def _create_pool(self):
        self._memory_budget()
        context = multiprocessing.get_context("spawn")
        self._ready_queue = context.Queue()
        gate, load_lock = context.Event(), context.Semaphore(1)
        pool = ProcessPoolExecutor(
            max_workers=self.slots, mp_context=context,
            initializer=_initialize_process,
            initargs=(self.config, str(self.root / "voices"), str(self.root / "results"),
                      self._ready_queue, gate, load_lock),
        )
        self._pool = pool
        try:
            probes = [pool.submit(_identity) for _ in range(self.slots)]
            ready = set()
            deadline = time.monotonic() + 180
            while len(ready) < self.slots:
                message = self._ready_queue.get(timeout=max(0.1, deadline - time.monotonic()))
                if not message["ready"]:
                    raise RuntimeError("Speech process initialization failed")
                ready.add(message["pid"])
            gate.set()
            for probe in probes:
                probe.result(timeout=15)
            self.worker_pids = sorted(ready)
        except Exception:
            gate.set()
            self._dispose_pool()
            raise

    def _dispose_pool(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            # Python 3.11 has no public terminate_workers API. These Process
            # objects belong exclusively to this pool; never touch other PIDs.
            processes = list((getattr(pool, "_processes", None) or {}).values())
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            pool.shutdown(wait=True, cancel_futures=True)
        if self._ready_queue is not None:
            self._ready_queue.close()
            self._ready_queue = None
        self.worker_pids = []

    async def start(self):
        await asyncio.to_thread(self._create_pool)

    async def run(self, job):
        pool = self._pool
        if pool is None:
            raise BrokenProcessPool("Speech processes are restarting")
        future = asyncio.wrap_future(pool.submit(_synthesize_job, job))
        # Pool termination also completes the shielded original future. Retrieve
        # its exception even when the watchdog already returned a timeout.
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            return await asyncio.wait_for(asyncio.shield(future),
                                          self.execution_timeout if self.execution_timeout > 0 else None)
        except TimeoutError:
            async with self._reset_lock:
                if self._pool is pool:
                    await asyncio.to_thread(self._dispose_pool)
            raise ExecutionTimeout("Speech execution timed out") from None

    async def recover(self):
        async with self._reset_lock:
            await asyncio.to_thread(self._dispose_pool)
            await asyncio.to_thread(self._create_pool)

    async def stop(self):
        async with self._reset_lock:
            await asyncio.to_thread(self._dispose_pool)


class LocalSpeechBackend:
    """Injected engines keep their original single owner for local tests."""

    requires_snapshots = False
    handles_timeout = False
    slots = 1
    requested_slots = 1
    capacity_reason = None

    def __init__(self, engine, executor, jobs_root):
        self.engine = engine
        self.executor = executor
        self.root = Path(jobs_root)

    async def start(self):
        pass

    async def run(self, job):
        def execute():
            data = self.engine.execute("speech", job["payload"])
            return _write_result(data, job["id"], self.root / "results")

        return await asyncio.get_running_loop().run_in_executor(self.executor, execute)

    async def recover(self):
        pass

    async def stop(self):
        pass
