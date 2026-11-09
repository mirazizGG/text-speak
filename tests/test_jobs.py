"""Durable queue, publication, recovery and private snapshot boundaries."""

from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import io
from pathlib import Path
import sqlite3
import struct
import threading
from types import SimpleNamespace
import uuid
import wave

import pytest

import jobs
from jobs import JobNotReadyError, JobStore


def request(number=0, **changes):
    return {"text": f"Salom. Sinov matni {number}.", "voice": "female_1",
            "mode": "preset", "language": "uz", "key": f"assistant.item{number}", **changes}


@pytest.fixture
def store(tmp_path):
    value = JobStore(tmp_path / "jobs")
    yield value
    value.close()


def pinned(store, tmp_path, contents=b"trusted prompt tensor data"):
    source = tmp_path / "source.pt"
    source.write_bytes(contents)
    digest = hashlib.sha256(contents).hexdigest()
    return source, store.pin_voice("female_1", "female", "Dilnoza", source, digest)


def publish_file(job, *, rate=24000, channels=1, width=2, seconds=0.1):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setframerate(rate)
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.writeframes(struct.pack("<h", 1000) * round(rate * seconds * channels * width / 2))
    path = Path(job["artifact_path"])
    path.write_bytes(data.getvalue())
    return {"path": str(path), "sample_rate": rate, "bytes": len(data.getvalue()),
            "wav_sha256": hashlib.sha256(data.getvalue()).hexdigest(), "duration": seconds}


@pytest.mark.parametrize("publication", ["complete", "requeue", "recover"])
def test_every_success_syncs_result_directory_before_committing(store, monkeypatch, publication):
    store.enqueue([request()])
    job = store.claim(0)
    result = publish_file(job)
    observed = []

    def sync(path):
        if path == store.results_dir:
            observed.append(store.get_job(job["id"])["state"])

    monkeypatch.setattr(jobs, "_fsync_directory", sync)
    if publication == "complete":
        store.complete(job["id"], result)
    elif publication == "requeue":
        store.requeue(job["id"])
    else:
        store.recover()
    assert observed == ["running"]
    assert store.get_job(job["id"])["state"] == "succeeded"


def test_hundred_requests_are_durable_fifo_without_four_item_limit(store):
    group = store.enqueue([request(i) for i in range(100)])
    assert group["counts"] == {"queued": 100, "running": 0, "succeeded": 0, "failed": 0}
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    for number, public in enumerate(group["jobs"]):
        internal = store.claim(number % 3)
        assert internal["id"] == public["id"]
        assert internal["payload"]["text"] == request(number)["text"]
        assert "key" not in internal["payload"]
        assert internal["filename"] == f"assistant.item{number}.wav"
        assert internal["attempts"] == 1
    assert store.claim(0) is None
    assert store.stats()["running"] == 100


def test_claim_is_atomic_across_connections(tmp_path):
    first = JobStore(tmp_path / "jobs")
    second = JobStore(tmp_path / "jobs")
    try:
        group = first.enqueue([request(i) for i in range(100)])
        def claim_many(slot):
            connection = first if slot % 2 else second
            claimed = []
            while (job := connection.claim(slot)) is not None:
                claimed.append(job["id"])
            return claimed
        with ThreadPoolExecutor(max_workers=8) as pool:
            assigned = [item for batch in pool.map(claim_many, range(8)) for item in batch]
        assert len(assigned) == len(set(assigned)) == 100
        assert set(assigned) == {job["id"] for job in group["jobs"]}
    finally:
        first.close()
        second.close()


def test_fifo_uses_admission_order_when_wall_clock_moves_backwards(store, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    earlier = store.enqueue([request(0)])
    clock[0] = 900.0
    later = store.enqueue([request(1)])
    assert store.claim(0)["group_id"] == earlier["id"]
    assert store.claim(1)["group_id"] == later["id"]


def test_pinned_voice_survives_original_deletion_without_path_disclosure(store, tmp_path):
    source, snapshot = pinned(store, tmp_path)
    original = source.read_bytes()
    group = store.enqueue([request()], {"female_1": snapshot})
    source.unlink()
    internal = store.claim(0)
    assert internal["snapshot"] == snapshot
    assert (store.snapshots_dir / f"{snapshot['sha256']}.pt").read_bytes() == original
    assert "snapshot" not in group["jobs"][0]
    assert "artifact_path" not in group["jobs"][0]
    assert str(store.root) not in str(group)
    assert group["jobs"][0]["voice_name"] == "Dilnoza"


def test_changed_source_cannot_replace_pinned_identity(store, tmp_path):
    source, snapshot = pinned(store, tmp_path)
    target = store.snapshots_dir / f"{snapshot['sha256']}.pt"
    original = target.read_bytes()
    source.write_bytes(b"different prompt")
    with pytest.raises(ValueError, match="checksum"):
        store.pin_voice("female_1", "female", "Dilnoza", source, snapshot["sha256"])
    assert target.read_bytes() == original
    assert list(store.snapshots_dir.glob("*.tmp")) == []


def test_snapshot_hash_and_size_are_checked_before_publication(store, tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"reference")
    with pytest.raises(ValueError, match="checksum"):
        store.pin_voice("female_1", "female", "Voice", source, "a" * 64)
    with source.open("wb") as stream:
        stream.truncate(jobs._MAX_SNAPSHOT_BYTES + 1)
    with pytest.raises(ValueError):
        store.pin_voice("female_1", "female", "Voice", source, "a" * 64)
    assert list(store.snapshots_dir.iterdir()) == []


def test_enqueue_requires_matching_snapshot_when_mapping_is_supplied(store, tmp_path):
    _, snapshot = pinned(store, tmp_path)
    with pytest.raises(ValueError):
        store.enqueue([request()], {})
    with pytest.raises(ValueError):
        store.enqueue([request()], {"female_1": dict(snapshot, voice_id="male_1")})
    assert store.list_groups() == []
    assert store.enqueue([request(mode="auto", voice=None)], {})["counts"]["queued"] == 1


def test_enqueue_is_one_transaction_even_if_sql_fails_mid_group(store):
    store._db.execute("""CREATE TRIGGER force_second_insert_failure BEFORE INSERT ON jobs
        WHEN NEW.ordinal=1 BEGIN SELECT RAISE(ABORT,'injected failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        store.enqueue([request(0), request(1), request(2)])
    assert store.list_groups() == []
    assert sum(store.stats().values()) == 0


def test_invalid_item_does_not_partially_admit_group(store):
    with pytest.raises(ValueError):
        store.enqueue([request(1), request(2, key="../unsafe")])
    assert store.list_groups() == []


def test_disk_admission_reserves_all_outstanding_audio(tmp_path, monkeypatch):
    value = JobStore(tmp_path / "jobs", min_free_bytes=100, max_audio_bytes=1000)
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda _: SimpleNamespace(free=2100))
    try:
        value.enqueue([request(0), request(1)])
        with pytest.raises(OSError) as error:
            value.enqueue([request(2)])
        assert error.value.errno == errno.ENOSPC
        assert len(value.list_groups()) == 1
        current = value.claim(0)
        value.fail(current["id"], "invalid_generation", "Generation failed")
        value.enqueue([request(3)])
        assert value.stats()["queued"] == 2
    finally:
        value.close()


def test_unlimited_audio_admission_estimates_pending_text_against_actual_free_disk(tmp_path, monkeypatch):
    value = JobStore(tmp_path / "jobs", min_free_bytes=100)
    free = [100 + value._estimated_audio_bytes(request(text="a")) * 2]
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda _: SimpleNamespace(free=free[0]))
    try:
        assert value.max_audio_bytes == 0
        value.enqueue([request(0, text="a"), request(1, text="a")])
        with pytest.raises(OSError) as error:
            value.enqueue([request(2, text="a")])
        assert error.value.errno == errno.ENOSPC
        free[0] += value._estimated_audio_bytes(request(text="a"))
        value.enqueue([request(2, text="a")])
        assert value.stats()["queued"] == 3
        assert value._estimated_audio_bytes(request(text="a" * 2000)) > value._estimated_audio_bytes(request(text="a" * 1000))
        assert value._estimated_audio_bytes(request(text="a" * 1000, speed=0.5)) > value._estimated_audio_bytes(request(text="a" * 1000))
        assert value._estimated_audio_bytes(request(text="a" * 1000, duration=2)) == 96044
    finally:
        value.close()


def test_store_accepts_more_than_a_thousand_jobs_and_long_text(store, monkeypatch):
    monkeypatch.setattr(jobs.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**63))
    group = store.enqueue([request(i, text="a" * 1200) for i in range(1001)])
    assert len(group["jobs"]) == 1001
    assert group["counts"]["queued"] == 1001


def test_long_wav_can_complete_recover_and_stream_without_the_old_ten_minute_cap(store):
    store.enqueue([request(text="Uzun matn")])
    job = store.claim(0)
    result = publish_file(job, seconds=601)
    assert result["bytes"] > 28800044
    store.complete(job["id"], result)
    assert store.recover()["missing_results"] == 0
    with store.acquire_result(job["id"]) as lease:
        assert lease.size == result["bytes"]
        assert lease.stream.read(4) == b"RIFF"
    assert store.get_job(job["id"])["duration"] == 601


def test_an_explicit_operator_audio_cap_remains_optional(tmp_path):
    value = JobStore(tmp_path / "jobs", max_audio_bytes=1000)
    try:
        value.enqueue([request()])
        job = value.claim(0)
        result = publish_file(job)
        with pytest.raises(ValueError):
            value.complete(job["id"], result)
    finally:
        value.close()


def test_default_retention_preserves_completed_and_failed_jobs_past_seven_days(store, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    group = store.enqueue([request(0), request(1)])
    ready, failed = store.claim(0), store.claim(1)
    store.complete(ready["id"], publish_file(ready))
    store.fail(failed["id"], "generation_failed", "Failed")
    clock[0] += 31 * 24 * 60 * 60
    assert store.retention_seconds == 0
    assert store.cleanup()["jobs"] == 0
    assert store.get_group(group["id"])["counts"] == {
        "queued": 0, "running": 0, "succeeded": 1, "failed": 1}
    assert store.get_result_path(ready["id"]).is_file()


def test_audio_publication_and_group_scoped_download(store):
    group = store.enqueue([request()])
    job = store.claim(0)
    with pytest.raises(JobNotReadyError):
        store.get_result_path(group["id"], job["id"])
    result = publish_file(job)
    store.complete(job["id"], result)
    store.complete(job["id"], result)  # A duplicate completion is idempotent.
    public = store.get_job(group["id"], job["id"])
    assert public["state"] == "succeeded"
    assert public["download_url"] == f"/v1/audio/jobs/{group['id']}/{job['id']}/audio"
    assert store.get_result_path(job["id"]) == Path(result["path"])
    assert store.get_job(job["id"]) == public
    with pytest.raises(KeyError):
        store.get_result_path(uuid.uuid4().hex, job["id"])
    assert "path" not in public and "wav_sha256" not in public


@pytest.mark.parametrize("field,value", [
    ("path", "outside.wav"), ("sample_rate", 16000), ("bytes", 1),
    ("wav_sha256", "f" * 64), ("duration", 100), ("duration", float("nan")),
])
def test_worker_metadata_cannot_spoof_published_audio(store, field, value):
    store.enqueue([request()])
    job = store.claim(0)
    result = publish_file(job)
    with pytest.raises(ValueError):
        store.complete(job["id"], dict(result, **{field: value}))
    assert store.get_job(job["id"])["state"] == "running"


@pytest.mark.parametrize("updates", [{"rate": 16000}, {"channels": 2}, {"width": 1}])
def test_only_expected_pcm_format_can_be_published(store, updates):
    store.enqueue([request()])
    job = store.claim(0)
    result = publish_file(job, **updates)
    with pytest.raises(ValueError):
        store.complete(job["id"], result)


def test_truncated_wav_and_partial_file_are_never_downloadable(store):
    group = store.enqueue([request()])
    job = store.claim(0)
    result = publish_file(job)
    path = Path(result["path"])
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError):
        store.complete(job["id"], result)
    path.rename(str(path) + ".tmp")
    with pytest.raises(JobNotReadyError):
        store.get_result_path(group["id"], job["id"])


def test_recovery_reconciles_rename_before_commit_and_requeues_partial(store, tmp_path):
    group = store.enqueue([request(0), request(1)])
    completed = store.claim(0)
    partial = store.claim(1)
    publish_file(completed)
    partial_path = Path(partial["artifact_path"] + ".tmp")
    partial_path.write_bytes(b"unfinished")
    keep = store.results_dir / "unrelated.txt"
    keep.write_text("keep")
    store.close()
    reopened = JobStore(tmp_path / "jobs")
    try:
        recovered = reopened.recover()
        assert recovered["succeeded"] == 1 and recovered["requeued"] == 1
        assert reopened.get_job(completed["id"])["state"] == "succeeded"
        assert reopened.get_result_path(group["id"], completed["id"]).is_file()
        assert not partial_path.exists() and keep.read_text() == "keep"
        retry = reopened.claim(0)
        assert retry["id"] == partial["id"] and retry["attempts"] == 2
        assert reopened.claim(1) is None
    finally:
        reopened.close()


def test_requeue_does_not_repeat_work_after_final_audio_was_published(store):
    store.enqueue([request()])
    job = store.claim(0)
    publish_file(job)
    store.requeue(job["id"])
    assert store.get_job(job["id"])["state"] == "succeeded"
    assert store.claim(0) is None


def test_retry_removes_only_its_interrupted_partial(store):
    store.enqueue([request(0), request(1)])
    first, other = store.claim(0), store.claim(1)
    partial = Path(first["artifact_path"] + ".tmp")
    protected = Path(other["artifact_path"] + ".tmp")
    partial.write_bytes(b"interrupted output")
    protected.write_bytes(b"another worker is writing")
    store.requeue(first["id"])
    assert not partial.exists() and protected.exists()
    assert store.claim(0)["id"] == first["id"]


def test_recovery_detects_tampered_completed_audio(store):
    store.enqueue([request()])
    job = store.claim(0)
    result = publish_file(job)
    store.complete(job["id"], result)
    Path(job["artifact_path"]).write_bytes(b"broken")
    with pytest.raises(KeyError):
        store.get_result_path(job["id"])
    assert store.recover()["missing_results"] == 1
    assert store.get_job(job["id"])["state"] == "failed"


@pytest.mark.parametrize("unsafe", ["../outside", "a/../../x", "A" * 32, "a" * 31, "a" * 33, "", None])
def test_job_and_group_ids_cannot_address_arbitrary_paths(store, unsafe):
    with pytest.raises(KeyError):
        store.get_group(unsafe)
    with pytest.raises(KeyError):
        store.get_job(unsafe)
    with pytest.raises(KeyError):
        store.get_result_path(unsafe)
    with pytest.raises(KeyError):
        store.fail(unsafe, "worker_failed", "Failure")


def test_failed_job_is_terminal_and_error_is_plain_text(store):
    group = store.enqueue([request()])
    job = store.claim(0)
    store.fail(job["id"], "worker_failed", "A failure\r\n" + "x" * 1000)
    store.requeue(job["id"])
    public = store.get_job(job["id"])
    assert public["state"] == "failed"
    assert public["error"]["code"] == "worker_failed"
    assert "\n" not in public["error"]["detail"] and len(public["error"]["detail"]) == 500
    assert store.get_group(group["id"])["state"] == "failed"
    assert store.claim(0) is None


def test_cleanup_preserves_live_jobs_and_discards_finished_voice_embeddings(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    value = JobStore(tmp_path / "jobs", retention_seconds=60)
    try:
        _, snapshot = pinned(value, tmp_path)
        prompt = value.snapshots_dir / f"{snapshot['sha256']}.pt"
        jobs.os.utime(prompt, (900, 900))
        old = value.enqueue([request(0)], {"female_1": snapshot})
        old_job = value.claim(0)
        old_result = publish_file(old_job)
        value.complete(old_job["id"], old_result)
        waiting = value.enqueue([request(1)], {"female_1": snapshot})
        clock[0] = 1061
        removed = value.cleanup()
        assert removed["jobs"] == 1 and not Path(old_result["path"]).exists()
        with pytest.raises(KeyError):
            value.get_group(old["id"])
        assert value.get_group(waiting["id"])["state"] == "queued" and prompt.exists()
        active = value.claim(0)
        value.complete(active["id"], publish_file(active))
        assert value.cleanup()["snapshots"] == 1
        assert not prompt.exists()
        assert value.get_result_path(active["id"]).is_file()
        assert value.get_group(waiting["id"])["state"] == "succeeded"
    finally:
        value.close()


def test_newly_pinned_orphan_snapshot_has_enqueue_grace(store, tmp_path):
    _, snapshot = pinned(store, tmp_path)
    store.cleanup()
    assert (store.snapshots_dir / f"{snapshot['sha256']}.pt").exists()
    assert store.enqueue([request()], {"female_1": snapshot})["counts"]["queued"] == 1


def test_cleanup_removes_stale_owned_snapshot_partial_and_preserves_unrelated_files(store):
    old = store.snapshots_dir / f".{hashlib.sha256(b'old').hexdigest()}.{uuid.uuid4().hex}.tmp"
    old.write_bytes(b"interrupted prompt copy")
    jobs.os.utime(old, (1, 1))
    recent = store.snapshots_dir / f".{hashlib.sha256(b'new').hexdigest()}.{uuid.uuid4().hex}.tmp"
    recent.write_bytes(b"new copy in progress")
    unrelated = store.snapshots_dir / "unrelated.tmp"
    unrelated.write_bytes(b"keep")
    store.cleanup()
    assert not old.exists() and recent.exists() and unrelated.exists()


def test_symlink_inputs_and_results_are_rejected(store, tmp_path):
    source = tmp_path / "source.pt"
    source.write_bytes(b"prompt")
    link = tmp_path / "link.pt"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this host")
    with pytest.raises(ValueError):
        store.pin_voice("female_1", "female", "Voice", link, hashlib.sha256(b"prompt").hexdigest())
    store.enqueue([request()])
    job = store.claim(0)
    result = publish_file(job)
    path = Path(result["path"])
    outside = tmp_path / "external.wav"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        store.complete(job["id"], result)
    assert outside.is_file()


def test_all_fifty_pending_groups_remain_discoverable_after_reload(tmp_path):
    value = JobStore(tmp_path / "jobs")
    groups = [value.enqueue([request(number)]) for number in range(50)]
    value.claim(0)
    value.claim(1)
    assert len(value.list_groups()) == 50
    assert not value.history_has_more()
    value.close()
    reopened = JobStore(tmp_path / "jobs")
    try:
        reopened.recover()
        listed = reopened.list_groups()
        assert {group["id"] for group in listed} == {group["id"] for group in groups}
        assert all(group["counts"]["queued"] == 1 for group in listed)
    finally:
        reopened.close()


def test_terminal_history_pages_do_not_hide_older_active_groups(store, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    active = [store.enqueue([request(number)]) for number in range(3)]
    history = []
    clock[0] = 2000
    for number in range(50):
        group = store.enqueue([request(number + 3)])
        store.fail(group["jobs"][0]["id"], "invalid_generation", "Generation failed")
        history.append(group)
    pages = [store.list_groups(20, offset, include_active=False) for offset in (0, 20, 40)]
    assert [len(page) for page in pages] == [20, 20, 10]
    assert [group["id"] for page in pages for group in page] == [group["id"] for group in reversed(history)]
    assert store.history_has_more(20, 0) and store.history_has_more(20, 20)
    assert not store.history_has_more(20, 40)
    merged = store.list_groups(20, 20)
    assert len(merged) == len({group["id"] for group in merged}) == 23
    assert {group["id"] for group in active} <= {group["id"] for group in merged}
    assert [group["id"] for group in merged[:20]] == [group["id"] for group in pages[1]]


def test_partial_group_is_active_until_every_job_finishes(store):
    group = store.enqueue([request(0), request(1)])
    store.fail(group["jobs"][0]["id"], "worker_failed", "Failure")
    assert store.list_groups(1, 100) == [store.get_group(group["id"])]
    assert store.list_groups(include_active=False) == []
    store.fail(group["jobs"][1]["id"], "worker_failed", "Failure")
    assert len(store.list_groups(include_active=False)) == 1


def test_result_lease_protects_expired_group_and_artifact_until_stream_closes(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    value = JobStore(tmp_path / "jobs", retention_seconds=10)
    try:
        group = value.enqueue([request()])
        job = value.claim(0)
        result = publish_file(job)
        original = Path(result["path"]).read_bytes()
        value.complete(job["id"], result)
        lease = value.acquire_result(group["id"], job["id"])
        assert (lease.filename, lease.size, lease.sample_rate) == ("assistant.item0.wav", len(original), 24000)
        clock[0] = 1011
        assert value.cleanup()["jobs"] == 0
        assert value.get_group(group["id"])["state"] == "succeeded"
        assert lease.stream.read() == original
        lease.close()
        lease.close()
        assert lease.stream.closed and not value._lease_counts
        assert value.cleanup()["jobs"] == 1
        assert not Path(result["path"]).exists()
        with pytest.raises(KeyError):
            value.get_group(group["id"])
    finally:
        value.close()


def test_cleanup_waits_for_every_download_lease(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    value = JobStore(tmp_path / "jobs", retention_seconds=10)
    try:
        value.enqueue([request()])
        job = value.claim(0)
        value.complete(job["id"], publish_file(job))
        first, second = value.acquire_result(job["id"]), value.acquire_result(job["id"])
        clock[0] = 1011
        first.close()
        first.close()
        assert value.cleanup()["jobs"] == 0
        assert second.stream.read(4) == b"RIFF"
        assert value._lease_counts[job["id"]] == 1
        second.close()
        assert value.cleanup()["jobs"] == 1
    finally:
        value.close()


def test_partial_stream_disconnect_releases_lease_for_retention_cleanup(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    value = JobStore(tmp_path / "jobs", retention_seconds=10)
    try:
        value.enqueue([request()])
        job = value.claim(0)
        value.complete(job["id"], publish_file(job))
        lease = value.acquire_result(job["id"])
        def chunks():
            try:
                while block := lease.stream.read(64):
                    yield block
            finally:
                lease.close()
        stream = chunks()
        assert next(stream).startswith(b"RIFF")
        clock[0] = 1011
        assert value.cleanup()["jobs"] == 0
        stream.close()  # The client disconnected before the final chunk.
        assert lease.stream.closed and not value._lease_counts
        assert value.cleanup()["jobs"] == 1
    finally:
        value.close()


def test_lease_can_be_closed_after_store_shutdown(store):
    store.enqueue([request()])
    job = store.claim(0)
    store.complete(job["id"], publish_file(job))
    lease = store.acquire_result(job["id"])
    store.close()
    assert lease.stream.read(4) == b"RIFF"
    lease.close()
    lease.close()
    assert lease.stream.closed and not store._lease_counts


def test_not_ready_or_corrupted_result_cannot_acquire_a_lease(store):
    group = store.enqueue([request()])
    job = store.claim(0)
    with pytest.raises(JobNotReadyError):
        store.acquire_result(job["id"])
    store.complete(job["id"], publish_file(job))
    with pytest.raises(KeyError):
        store.acquire_result(uuid.uuid4().hex, job["id"])
    Path(job["artifact_path"]).write_bytes(b"invalid audio")
    with pytest.raises(KeyError):
        store.acquire_result(group["id"], job["id"])
    assert not store._lease_counts


def test_acquisition_opens_and_leases_under_same_lock_as_cleanup(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(jobs.time, "time", lambda: clock[0])
    value = JobStore(tmp_path / "jobs", retention_seconds=10)
    opened, release, cleanup_entered = threading.Event(), threading.Event(), threading.Event()
    try:
        value.enqueue([request()])
        job = value.claim(0)
        value.complete(job["id"], publish_file(job))
        original_open = jobs._open_regular
        def paused_open(path, maximum):
            result = original_open(path, maximum)
            opened.set()
            release.wait(5)
            return result
        monkeypatch.setattr(jobs, "_open_regular", paused_open)
        with ThreadPoolExecutor(max_workers=2) as executor:
            acquiring = executor.submit(value.acquire_result, job["id"])
            assert opened.wait(2)
            clock[0] = 1011
            def cleanup():
                cleanup_entered.set()
                return value.cleanup()
            removing = executor.submit(cleanup)
            try:
                assert cleanup_entered.wait(2) and not removing.done()
            finally:
                release.set()
            lease = acquiring.result(timeout=2)
            assert removing.result(timeout=2)["jobs"] == 0
        assert lease.stream.read(4) == b"RIFF"
        lease.close()
        assert value.cleanup()["jobs"] == 1
    finally:
        release.set()
        value.close()
