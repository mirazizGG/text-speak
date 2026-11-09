"""Durable speech jobs and private, immutable voice prompt snapshots."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import threading
import time
import uuid
import wave

from engine import speech_filename


_ID = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VOICE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_PARTIAL = re.compile(r"[0-9a-f]{32}\.wav(?:\.[A-Za-z0-9_-]+)?\.tmp\Z")
_SNAPSHOT_PARTIAL = re.compile(r"\.[0-9a-f]{64}\.[0-9a-f]{32}\.tmp\Z")
_STATES = ("queued", "running", "succeeded", "failed")
_MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
_SNAPSHOT_GRACE_SECONDS = 60


class JobNotReadyError(LookupError):
    """The job exists but has no completed audio yet."""


class ResultLease:
    """An opened result that retention cleanup cannot remove while streaming."""

    def __init__(self, store, job_id, stream, filename, size, sample_rate):
        self._store = store
        self._job_id = job_id
        self.stream = stream
        self.filename = filename
        self.size = size
        self.sample_rate = sample_rate
        self._closed = False
        self._registered = False

    def close(self):
        # The store lock also protects cleanup and acquisition. No SQLite calls
        # are needed here, so closing a response after store shutdown is safe.
        with self._store._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.stream.close()
            finally:
                if self._registered:
                    remaining = self._store._lease_counts.get(self._job_id, 0) - 1
                    if remaining > 0:
                        self._store._lease_counts[self._job_id] = remaining
                    else:
                        self._store._lease_counts.pop(self._job_id, None)
                    self._registered = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise KeyError("Job or group was not found")
    return value


def _no_symlinks(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        try:
            attributes = part.lstat()
        except FileNotFoundError:
            continue
        if part.is_symlink() or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Job storage paths must not contain symbolic links")
    return path


def _fsync_directory(path):
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_regular(path, maximum):
    path = _no_symlinks(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    stream = os.fdopen(descriptor, "rb")
    try:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_size <= 0
                or maximum > 0 and info.st_size > maximum):
            raise ValueError("Job artifact is not a valid regular file")
        return stream, info
    except BaseException:
        stream.close()
        raise


@contextmanager
def _read_regular(path, maximum):
    stream, info = _open_regular(path, maximum)
    with stream:
        yield stream, info


def _stream_digest(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


class JobStore:
    def __init__(self, root: Path, retention_seconds=0,
                 min_free_bytes=512 * 1024 * 1024, max_audio_bytes=0):
        if (isinstance(retention_seconds, bool) or not isinstance(retention_seconds, (int, float))
                or not math.isfinite(retention_seconds) or retention_seconds < 0):
            raise ValueError("Job retention must be nonnegative; zero keeps results")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (min_free_bytes, max_audio_bytes)) or 0 < max_audio_bytes < 46:
            raise ValueError("Invalid job storage limits")
        self.root = _no_symlinks(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.root / "voices"
        self.results_dir = self.root / "results"
        for directory in (self.root, self.snapshots_dir, self.results_dir):
            _no_symlinks(directory)
            directory.mkdir(exist_ok=True)
            if not directory.is_dir():
                raise ValueError("Job storage must use directories")
        self.retention_seconds = float(retention_seconds)
        self.min_free_bytes = min_free_bytes
        self.max_audio_bytes = max_audio_bytes
        self._lock = threading.RLock()
        self._lease_counts = {}
        self._closed = False
        database = _no_symlinks(self.root / "jobs.sqlite3")
        self._db = sqlite3.connect(database, timeout=30, isolation_level=None,
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS groups (
                id TEXT PRIMARY KEY,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('queued','running','succeeded','failed')),
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                payload_json TEXT NOT NULL,
                snapshot_json TEXT,
                text TEXT NOT NULL,
                download_key TEXT,
                voice TEXT,
                voice_name TEXT,
                filename TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                slot INTEGER,
                error_code TEXT,
                error_detail TEXT,
                result_json TEXT,
                UNIQUE(group_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created_at);
            CREATE INDEX IF NOT EXISTS jobs_group ON jobs(group_id, ordinal);
        """)

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Job store is closed")
            _no_symlinks(self.snapshots_dir)
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def _row(self, job_id, group_id=None):
        _identifier(job_id)
        if group_id is None:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        else:
            _identifier(group_id)
            row = self._db.execute("SELECT * FROM jobs WHERE id=? AND group_id=?",
                                   (job_id, group_id)).fetchone()
        if row is None:
            raise KeyError("Job or group was not found")
        return row

    def _snapshot(self, value, voice_id):
        if not isinstance(value, dict) or value.get("voice_id") != voice_id:
            raise ValueError("The voice snapshot does not match the selected voice")
        digest = value.get("sha256")
        gender, name = value.get("gender"), value.get("name")
        if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
                or gender not in ("female", "male")
                or not isinstance(name, str) or not 1 <= len(name.strip()) <= 80):
            raise ValueError("Invalid voice snapshot metadata")
        path = self.snapshots_dir / f"{digest}.pt"
        with _read_regular(path, _MAX_SNAPSHOT_BYTES) as (stream, _):
            if _stream_digest(stream) != digest:
                raise ValueError("Voice snapshot checksum does not match")
        return {"voice_id": voice_id, "gender": gender, "name": name, "sha256": digest}

    def pin_voice(self, voice_id, gender, name, source: Path, expected_digest):
        if (not isinstance(voice_id, str) or not _VOICE.fullmatch(voice_id)
                or gender not in ("female", "male") or not isinstance(name, str)
                or not 1 <= len(name.strip()) <= 80
                or not isinstance(expected_digest, str) or not _DIGEST.fullmatch(expected_digest)):
            raise ValueError("Invalid voice snapshot metadata")
        snapshot = {"voice_id": voice_id, "gender": gender, "name": name,
                    "sha256": expected_digest}
        destination = self.snapshots_dir / f"{expected_digest}.pt"
        temporary = self.snapshots_dir / f".{expected_digest}.{uuid.uuid4().hex}.tmp"
        with self._lock:
            if self._closed:
                raise RuntimeError("Job store is closed")
            _no_symlinks(self.snapshots_dir)
            try:
                with _read_regular(source, _MAX_SNAPSHOT_BYTES) as (stream, info):
                    if destination.exists() or destination.is_symlink():
                        self._snapshot(snapshot, voice_id)
                        if _stream_digest(stream) != expected_digest:
                            raise ValueError("Source voice checksum does not match")
                    else:
                        digest = hashlib.sha256()
                        copied = 0
                        with temporary.open("xb") as target:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                copied += len(chunk)
                                if copied > _MAX_SNAPSHOT_BYTES:
                                    raise ValueError("Voice snapshot exceeds the size limit")
                                digest.update(chunk)
                                target.write(chunk)
                            target.flush()
                            os.fsync(target.fileno())
                        if copied != info.st_size or digest.hexdigest() != expected_digest:
                            raise ValueError("Source voice checksum does not match")
                        os.replace(temporary, destination)
                        _fsync_directory(self.snapshots_dir)
                # Protect a freshly pinned, not-yet-enqueued snapshot from the janitor.
                if os.utime in os.supports_follow_symlinks:
                    os.utime(destination, None, follow_symlinks=False)
                else:
                    _no_symlinks(destination)
                    os.utime(destination, None)
                return snapshot
            finally:
                temporary.unlink(missing_ok=True)

    def enqueue(self, items: list[dict], snapshots=None):
        if not isinstance(items, list) or not items:
            raise ValueError("At least one validated speech request is required")
        if snapshots is not None and not isinstance(snapshots, dict):
            raise ValueError("Voice snapshots must be a mapping")
        group_id = uuid.uuid4().hex
        now = time.time()
        with self._transaction():
            prepared = []
            checked_snapshots = {}
            for ordinal, item in enumerate(items):
                if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
                    raise ValueError("A validated speech request is required")
                payload = {key: value for key, value in item.items() if key != "key"}
                key = item.get("key")
                filename = speech_filename(key)
                voice = item.get("voice")
                if voice is not None and (not isinstance(voice, str) or not _VOICE.fullmatch(voice)):
                    raise ValueError("Invalid voice identifier")
                snapshot = None
                if item.get("mode", "preset") == "preset" and snapshots is not None:
                    if voice not in checked_snapshots:
                        checked_snapshots[voice] = self._snapshot(snapshots.get(voice), voice)
                    snapshot = checked_snapshots[voice]
                prepared.append((uuid.uuid4().hex, group_id, ordinal, "queued", now,
                                 json.dumps(payload, allow_nan=False),
                                 json.dumps(snapshot) if snapshot else None,
                                 item["text"], key, voice,
                                 snapshot["name"] if snapshot else voice, filename))
            if self.max_audio_bytes:
                outstanding = self._db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
                reserved = (outstanding + len(prepared)) * self.max_audio_bytes
            else:
                # Admission reserves estimated output against actual free disk;
                # this is not a per-file size or speech-duration ceiling.
                reserved = sum(self._estimated_audio_bytes(json.loads(row[0])) for row in self._db.execute(
                    "SELECT payload_json FROM jobs WHERE state IN ('queued','running')"))
                reserved += sum(self._estimated_audio_bytes(item) for item in items)
            needed = reserved + self.min_free_bytes
            if shutil.disk_usage(self.root).free < needed:
                raise OSError(errno.ENOSPC, "Insufficient storage for queued audio results")
            self._db.execute("INSERT INTO groups(id,created_at) VALUES (?,?)", (group_id, now))
            self._db.executemany("""
                INSERT INTO jobs(id,group_id,ordinal,state,created_at,payload_json,snapshot_json,
                                 text,download_key,voice,voice_name,filename)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, prepared)
        return self.get_group(group_id)

    @staticmethod
    def _estimated_audio_bytes(payload):
        duration = payload.get("duration")
        if (not isinstance(duration, bool) and isinstance(duration, (int, float))
                and math.isfinite(duration) and duration > 0):
            seconds = duration
        else:
            speed = payload.get("speed", 1)
            if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                    or not math.isfinite(speed) or speed <= 0):
                speed = 1
            seconds = max(1, len(payload["text"]) * 0.5 / speed)
        return 44 + math.ceil(seconds * 24000 * 2)

    def claim(self, slot: int):
        if isinstance(slot, bool) or not isinstance(slot, int) or slot < 0:
            raise ValueError("A nonnegative worker slot is required")
        with self._transaction():
            row = self._db.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
            if row is None:
                return None
            started = time.time()
            self._db.execute("""UPDATE jobs SET state='running',started_at=?,finished_at=NULL,
                attempts=attempts+1,slot=?,error_code=NULL,error_detail=NULL WHERE id=?""",
                             (started, slot, row["id"]))
            return {"id": row["id"], "group_id": row["group_id"],
                    "payload": json.loads(row["payload_json"]),
                    "snapshot": json.loads(row["snapshot_json"]) if row["snapshot_json"] else None,
                    "filename": row["filename"], "attempts": row["attempts"] + 1,
                    "artifact_path": str(self.results_dir / f"{row['id']}.wav")}

    def _read_artifact(self, job_id):
        path = self.results_dir / f"{_identifier(job_id)}.wav"
        with _read_regular(path, self.max_audio_bytes) as (stream, info):
            return self._inspect_artifact(path, stream, info)

    @staticmethod
    def _inspect_artifact(path, stream, info):
        try:
            with wave.open(stream, "rb") as audio:
                frames = audio.getnframes()
                if (audio.getnchannels() != 1 or audio.getsampwidth() != 2
                        or audio.getframerate() != 24000 or audio.getcomptype() != "NONE"
                        or frames < 1):
                    raise ValueError("Invalid completed audio format or duration")
                remaining = frames
                while remaining:
                    count = min(remaining, 65536)
                    if len(audio.readframes(count)) != count * 2:
                        raise ValueError("Completed WAV audio is truncated")
                    remaining -= count
        except (wave.Error, EOFError) as error:
            raise ValueError("Invalid completed WAV file") from error
        digest = _stream_digest(stream)
        after = os.fstat(stream.fileno())
        if (after.st_size != info.st_size or after.st_mtime_ns != info.st_mtime_ns
                or after.st_ino != info.st_ino or after.st_dev != info.st_dev):
            raise ValueError("Completed audio changed during validation")
        return {"path": str(path), "sample_rate": 24000, "bytes": info.st_size,
                "wav_sha256": digest, "duration": frames / 24000}

    def _validate_result(self, job_id, result, actual=None):
        if not isinstance(result, dict):
            raise ValueError("Completed audio metadata is required")
        path = result.get("path")
        if not isinstance(path, (str, Path)):
            raise ValueError("Invalid completed audio path")
        expected = self.results_dir / f"{_identifier(job_id)}.wav"
        if Path(path).absolute() != expected:
            raise ValueError("Completed audio must use its assigned job path")
        if actual is None:
            actual = self._read_artifact(job_id)
        for field in ("sample_rate", "bytes"):
            value = result.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value != actual[field]:
                raise ValueError("Completed audio metadata does not match the file")
        duration = result.get("duration")
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or abs(duration - actual["duration"]) > 1 / 24000):
            raise ValueError("Completed audio duration does not match the file")
        if result.get("wav_sha256") != actual["wav_sha256"]:
            raise ValueError("Completed audio checksum does not match")
        return actual

    def _mark_success(self, job_id, result):
        # Recovery can discover a file after its writer renamed it but died
        # before publication. Persist that directory entry before committing
        # success, exactly as on the ordinary completion path.
        _fsync_directory(self.results_dir)
        self._db.execute("""UPDATE jobs SET state='succeeded',finished_at=?,slot=NULL,
            error_code=NULL,error_detail=NULL,result_json=? WHERE id=?""",
                         (time.time(), json.dumps(result), job_id))

    def complete(self, job_id, result):
        with self._transaction():
            row = self._row(job_id)
            if row["state"] == "succeeded":
                if self._validate_result(job_id, result) != json.loads(row["result_json"]):
                    raise ValueError("A published job result cannot be replaced")
                return
            if row["state"] != "running":
                raise ValueError("Only a running job can publish audio")
            actual = self._validate_result(job_id, result)
            self._mark_success(job_id, actual)

    def fail(self, job_id, code, detail):
        safe_code = re.sub(r"[^A-Za-z0-9_-]", "_", str(code))[:64] or "generation_failed"
        safe_detail = "".join(c for c in str(detail) if c.isprintable())[:500] or "Speech generation failed"
        with self._transaction():
            row = self._row(job_id)
            if row["state"] in ("succeeded", "failed"):
                return
            self._db.execute("""UPDATE jobs SET state='failed',finished_at=?,slot=NULL,
                error_code=?,error_detail=?,result_json=NULL WHERE id=?""",
                             (time.time(), safe_code, safe_detail, job_id))

    def _existing_result(self, job_id):
        try:
            return self._read_artifact(job_id)
        except (OSError, ValueError):
            return None

    def _remove_partials(self, job_id=None):
        _no_symlinks(self.results_dir)
        for path in self.results_dir.iterdir():
            if (_PARTIAL.fullmatch(path.name)
                    and (job_id is None or path.name.startswith(f"{job_id}.wav"))
                    and (path.is_file() or path.is_symlink())):
                path.unlink()

    def requeue(self, job_id):
        with self._transaction():
            row = self._row(job_id)
            if row["state"] != "running":
                return
            result = self._existing_result(job_id)
            if result is not None:
                self._mark_success(job_id, result)
            else:
                self._db.execute("""UPDATE jobs SET state='queued',started_at=NULL,
                    finished_at=NULL,slot=NULL,error_code=NULL,error_detail=NULL WHERE id=?""", (job_id,))
            self._remove_partials(job_id)

    def recover(self):
        recovered = {"succeeded": 0, "requeued": 0, "missing_results": 0}
        with self._transaction():
            rows = self._db.execute("SELECT * FROM jobs WHERE state!='failed'").fetchall()
            for row in rows:
                result = self._existing_result(row["id"])
                if row["state"] == "succeeded":
                    if result is None or result != json.loads(row["result_json"]):
                        self._db.execute("""UPDATE jobs SET state='failed',error_code='artifact_unavailable',
                            error_detail='Completed audio is unavailable.',result_json=NULL WHERE id=?""", (row["id"],))
                        recovered["missing_results"] += 1
                elif result is not None:
                    self._mark_success(row["id"], result)
                    recovered["succeeded"] += 1
                elif row["state"] == "running":
                    self._db.execute("""UPDATE jobs SET state='queued',started_at=NULL,
                        finished_at=NULL,slot=NULL,error_code=NULL,error_detail=NULL WHERE id=?""", (row["id"],))
                    recovered["requeued"] += 1
            self._remove_partials()
        return recovered

    @staticmethod
    def _public_job(row):
        succeeded = row["state"] == "succeeded"
        result = json.loads(row["result_json"]) if succeeded and row["result_json"] else None
        public = {
            "id": row["id"], "group_id": row["group_id"], "state": row["state"],
            "text": row["text"], "key": row["download_key"], "voice": row["voice"],
            "voice_name": row["voice_name"], "filename": row["filename"],
            "created_at": row["created_at"], "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "error": {"code": row["error_code"], "detail": row["error_detail"]}
            if row["state"] == "failed" else None,
            "download_url": f"/v1/audio/jobs/{row['group_id']}/{row['id']}/audio" if succeeded else None,
        }
        if result:
            public.update({key: result[key] for key in ("sample_rate", "bytes", "duration")})
        return public

    def get_group(self, group_id):
        _identifier(group_id)
        with self._lock:
            group = self._db.execute("SELECT * FROM groups WHERE id=?", (group_id,)).fetchone()
            if group is None:
                raise KeyError("Job or group was not found")
            rows = self._db.execute("SELECT * FROM jobs WHERE group_id=? ORDER BY ordinal", (group_id,)).fetchall()
            counts = dict.fromkeys(_STATES, 0)
            for row in rows:
                counts[row["state"]] += 1
            state = ("running" if counts["running"] else "queued" if counts["queued"]
                     else "failed" if counts["failed"] else "succeeded")
            return {"id": group_id, "created_at": group["created_at"], "state": state,
                    "jobs": [self._public_job(row) for row in rows], "counts": counts}

    def get_job(self, group_id, job_id=None):
        with self._lock:
            row = self._row(group_id) if job_id is None else self._row(job_id, group_id)
            return self._public_job(row)

    def get_result_path(self, group_id, job_id=None):
        with self._lock:
            row = self._row(group_id) if job_id is None else self._row(job_id, group_id)
            job_id = row["id"]
            if row["state"] != "succeeded":
                raise JobNotReadyError("Audio is not ready")
            try:
                result = self._validate_result(job_id, json.loads(row["result_json"]))
            except (OSError, ValueError, TypeError) as error:
                raise KeyError("Completed audio was not found") from error
            return Path(result["path"])

    def acquire_result(self, group_id, job_id=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Job store is closed")
            row = self._row(group_id) if job_id is None else self._row(job_id, group_id)
            job_id = row["id"]
            if row["state"] != "succeeded":
                raise JobNotReadyError("Audio is not ready")
            stream = None
            try:
                path = self.results_dir / f"{job_id}.wav"
                stream, info = _open_regular(path, self.max_audio_bytes)
                actual = self._inspect_artifact(path, stream, info)
                result = self._validate_result(job_id, json.loads(row["result_json"]), actual)
                stream.seek(0)
                lease = ResultLease(self, job_id, stream, row["filename"], result["bytes"], result["sample_rate"])
                self._lease_counts[job_id] = self._lease_counts.get(job_id, 0) + 1
                lease._registered = True
                return lease
            except BaseException as error:
                if stream is not None:
                    stream.close()
                if isinstance(error, (OSError, ValueError, TypeError)):
                    raise KeyError("Completed audio was not found") from error
                raise

    @staticmethod
    def _history_page(limit, offset):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("Group history limit must be between 1 and 1000")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < 2**63:
            raise ValueError("Group history offset must be a nonnegative integer")

    def list_groups(self, limit=20, offset=0, include_active=True):
        self._history_page(limit, offset)
        if not isinstance(include_active, bool):
            raise ValueError("Active group selection must be boolean")
        with self._lock:
            active = self._db.execute("""SELECT g.id,g.created_at,g.rowid AS admission
                FROM groups AS g WHERE EXISTS (SELECT 1 FROM jobs AS j
                WHERE j.group_id=g.id AND j.state IN ('queued','running'))""").fetchall() if include_active else []
            history = self._db.execute("""SELECT g.id,g.created_at,g.rowid AS admission
                FROM groups AS g WHERE NOT EXISTS (SELECT 1 FROM jobs AS j
                WHERE j.group_id=g.id AND j.state IN ('queued','running'))
                ORDER BY g.created_at DESC,g.rowid DESC LIMIT ? OFFSET ?""", (limit, offset)).fetchall()
            unique = {row["id"]: row for row in (*active, *history)}
            ordered = sorted(unique.values(), key=lambda row: (row["created_at"], row["admission"]), reverse=True)
            return [self.get_group(row["id"]) for row in ordered]

    def history_has_more(self, limit=20, offset=0):
        self._history_page(limit, offset)
        with self._lock:
            count = self._db.execute("""SELECT COUNT(*) FROM groups AS g
                WHERE NOT EXISTS (SELECT 1 FROM jobs AS j
                WHERE j.group_id=g.id AND j.state IN ('queued','running'))""").fetchone()[0]
            return count > offset + limit

    def stats(self):
        with self._lock:
            result = dict.fromkeys(_STATES, 0)
            for row in self._db.execute("SELECT state,COUNT(*) AS n FROM jobs GROUP BY state"):
                result[row["state"]] = row["n"]
            return result

    def cleanup(self):
        now = time.time()
        cutoff = now - self.retention_seconds
        removed = {"jobs": 0, "results": 0, "snapshots": 0}
        with self._transaction():
            _no_symlinks(self.snapshots_dir)
            _no_symlinks(self.results_dir)
            expired = [row for row in self._db.execute(
                "SELECT id FROM jobs WHERE state IN ('succeeded','failed') AND finished_at<?", (cutoff,)).fetchall()
                if not self._lease_counts.get(row["id"])] if self.retention_seconds > 0 else []
            for row in expired:
                self._db.execute("DELETE FROM jobs WHERE id=?", (row["id"],))
            removed["jobs"] = len(expired)
            self._db.execute("DELETE FROM groups WHERE NOT EXISTS (SELECT 1 FROM jobs WHERE group_id=groups.id)")
            live_snapshots = set()
            for row in self._db.execute("SELECT snapshot_json FROM jobs WHERE state IN ('queued','running') AND snapshot_json IS NOT NULL"):
                live_snapshots.add(json.loads(row["snapshot_json"])["sha256"])
            # Holding the write transaction prevents another store connection from
            # enqueuing a reference after this check but before unlinking its file.
            for path in self.snapshots_dir.iterdir():
                match = re.fullmatch(r"([0-9a-f]{64})\.pt", path.name)
                if (match and match[1] not in live_snapshots
                        and path.lstat().st_mtime < now - _SNAPSHOT_GRACE_SECONDS
                        and (path.is_file() or path.is_symlink())):
                    path.unlink()
                    removed["snapshots"] += 1
                elif (_SNAPSHOT_PARTIAL.fullmatch(path.name)
                      and path.lstat().st_mtime < now - _SNAPSHOT_GRACE_SECONDS
                      and (path.is_file() or path.is_symlink())):
                    path.unlink()
            retained = {row[0] for row in self._db.execute("SELECT id FROM jobs")}
            for path in self.results_dir.iterdir():
                match = re.fullmatch(r"([0-9a-f]{32})\.wav", path.name)
                if (match and match[1] not in retained and not self._lease_counts.get(match[1])
                        and (path.is_file() or path.is_symlink())):
                    path.unlink()
                    removed["results"] += 1
                elif (_PARTIAL.fullmatch(path.name) and path.name[:32] not in retained
                      and (path.is_file() or path.is_symlink())):
                    path.unlink()
        return removed

    def close(self):
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
