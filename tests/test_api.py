"""HTTP contract and worker-isolation tests; no model weights or GPU required."""

from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import asyncio
import io
import errno
import json
import math
from pathlib import Path
import struct
import tempfile
import threading
import time
from urllib.parse import urlsplit
import wave
import zipfile

from fastapi.testclient import TestClient
import httpx
from starlette.datastructures import UploadFile as StarletteUploadFile
import pytest

from api import BodyLimitMiddleware, LeasedAudioResponse, Settings, create_app
from job_service import JobService
from engine import ProtectedVoiceError, VoiceDeletionError, VoiceNotFoundError


API_KEY = "test-only-key-0123456789abcdef0123456789"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
VALID_REQUEST = {
    "text": "Salom, bugun havo yaxshi.",
    "language": "uz",
    "voice": "female_1",
    "gender": "female",
}


def make_wav(seconds=0.1):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(
            b"".join(
                struct.pack("<h", int(5000 * math.sin(2 * math.pi * 440 * i / 24000)))
                for i in range(round(24000 * seconds))
            )
        )
    return output.getvalue()


class FakeEngine:
    """Controllable inference boundary, including a genuinely blocked worker."""

    sample_rate = 24000
    languages = [
        {"code": "uz", "name": "Uzbek"},
        {"code": "ru", "name": "Russian"},
        {"code": "en", "name": "English"},
    ]
    voices = [
        {
            "id": voice_id,
            "name": voice_id,
            "gender": gender,
            "reference_language": "uz",
            "synthetic": True,
        }
        for voice_id, gender in [
            ("female_1", "female"),
            ("female_2", "female"),
            ("male_1", "male"),
            ("male_2", "male"),
        ]
    ]

    def __init__(self, *, block_load=False, block_first=False, load_error=None, error=None,
                 asr_available=True):
        self.load_started = threading.Event()
        self.allow_load = threading.Event()
        if not block_load:
            self.allow_load.set()
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_first = block_first
        self.load_error = load_error
        self.error = error
        self.calls = []
        self.operations = []
        self.work_count = 0
        self.voices = [dict(voice) for voice in type(self).voices]
        self.capabilities = {"asr_available": asr_available, "normalization_available": True}
        self.voice_design_attributes = {
            "gender": ["female", "male"],
            "age": ["young adult", "middle-aged"],
            "style": ["whisper"],
        }
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()
        self.wav = make_wav()

    def load(self):
        self.load_started.set()
        if not self.allow_load.wait(timeout=5):
            raise RuntimeError("Fake load was not released")
        if self.load_error:
            raise self.load_error

    def synthesize(self, text, language, voice):
        with self.guard:
            self.calls.append((text, language, voice))
        return self._perform(lambda: self._speech_result(voice))

    def _speech_result(self, voice):
        if voice is not None and not any(item["id"] == voice for item in self.voices):
            raise ValueError("Unknown voice profile")
        return self.wav

    def _perform(self, result):
        with self.guard:
            self.work_count += 1
            call_number = self.work_count
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.block_first and call_number == 1:
                if not self.release.wait(timeout=5):
                    raise RuntimeError("Fake inference was not released")
            if self.error:
                raise self.error
            return result()
        finally:
            with self.guard:
                self.active -= 1

    def execute(self, operation, payload):
        if isinstance(payload.get("audio"), Path):
            payload = {**payload, "audio": payload["audio"].read_bytes()}
        with self.guard:
            self.operations.append((operation, payload))
        if operation == "speech":
            return self.synthesize(payload["text"], payload.get("language"), payload.get("voice"))
        return self._perform(lambda: self._result(operation, payload))

    def _result(self, operation, payload):
        if operation in {"design_voice", "clone_voice"}:
            profile = {
                "id": f"custom_{len(self.voices) - 3}",
                "name": payload["name"],
                "gender": payload["gender"],
                "reference_language": payload.get("reference_language", payload.get("language")),
                "synthetic": operation == "design_voice",
            }
            self.voices.append(profile)
            return dict(profile)
        if operation == "delete_voice":
            voice = payload["voice"]
            if voice in {item["id"] for item in type(self).voices}:
                raise ProtectedVoiceError()
            if not any(item["id"] == voice for item in self.voices):
                raise VoiceNotFoundError()
            self.voices = [item for item in self.voices if item["id"] != voice]
            return {"id": voice, "deleted": True}
        if operation == "voice_sample":
            if not any(item["id"] == payload["voice"] for item in self.voices):
                raise VoiceNotFoundError()
            return self.wav
        if operation == "clone_speech":
            return self.wav
        if operation == "transcribe":
            return {"text": "Assalomu alaykum."}
        if operation == "batch":
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as archive:
                for index, _ in enumerate(payload["items"], start=1):
                    archive.writestr(f"{index:03}.wav", self.wav)
            return output.getvalue()
        raise AssertionError(f"Unexpected operation: {operation}")


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Condition did not become true before timeout")


def make_app(engine=None, **overrides):
    settings = Settings(api_key=API_KEY, **overrides)
    return create_app(settings=settings, engine=engine or FakeEngine())


def wait_ready(client):
    wait_until(lambda: client.get("/readyz").status_code == 200)


@pytest.fixture
def service():
    engine = FakeEngine()
    with TestClient(make_app(engine)) as client:
        wait_ready(client)
        yield client, engine


@pytest.mark.parametrize("route", ["/v1/languages", "/v1/voices", "/v1/audio/speech"])
@pytest.mark.parametrize("authorization", [None, "Bearer incorrect", "Basic invalid"])
def test_api_requires_valid_bearer_key(service, route, authorization):
    client, engine = service
    headers = {"Authorization": authorization} if authorization else {}
    response = (
        client.post(route, headers=headers, json=VALID_REQUEST)
        if route.endswith("speech")
        else client.get(route, headers=headers)
    )
    assert response.status_code == 401
    assert "Bearer" in response.headers.get("www-authenticate", "")
    assert API_KEY not in response.text
    assert engine.calls == []


def test_language_and_gender_filtered_voice_catalogs(service):
    client, _ = service
    languages = client.get("/v1/languages", headers=HEADERS)
    assert languages.status_code == 200
    assert {item["code"] for item in languages.json()["data"]} == {"uz", "ru", "en"}
    voices = client.get("/v1/voices", headers=HEADERS)
    assert voices.status_code == 200
    assert len(voices.json()["data"]) == 4
    female = client.get("/v1/voices?gender=female", headers=HEADERS)
    assert female.status_code == 200
    assert {item["id"] for item in female.json()["data"]} == {"female_1", "female_2"}
    assert all(item["gender"] == "female" for item in female.json()["data"])
    assert client.get("/v1/voices?gender=other", headers=HEADERS).status_code == 422


@pytest.mark.parametrize("key_fields", [{}, {"key": None}])
def test_success_returns_decodable_complete_wav_and_selected_voice(service, key_fields):
    client, engine = service
    request = dict(VALID_REQUEST, voice="male_2", gender="male", language="ru", **key_fields)
    response = client.post("/v1/audio/speech", headers=HEADERS, json=request)
    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0] == "audio/wav"
    assert response.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    assert response.content == engine.wav
    with wave.open(io.BytesIO(response.content), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 24000
        assert audio.getnframes() == 2400
        assert len(audio.readframes(2400)) == 4800
    assert engine.calls == [(request["text"], "ru", "male_2")]
    assert "key" not in engine.operations[-1][1]


@pytest.mark.parametrize("voice_fields", [
    {"voice": "female_1"},
    {"mode": "design", "instruct": "female, young adult"},
    {"mode": "auto"},
])
def test_speech_key_names_download_without_changing_model_input(service, voice_fields):
    client, engine = service
    request = {"text": "Davom etish uchun tilni tanlang.", "key": "assistant.welcome",
               "language": "uz", **voice_fields}
    response = client.post("/v1/audio/speech", headers=HEADERS, json=request)
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="assistant.welcome.wav"'
    assert response.headers["content-type"].split(";")[0] == "audio/wav"
    assert response.content == engine.wav
    operation, received = engine.operations[-1]
    assert operation == "speech"
    assert "key" not in received
    assert received["text"] == request["text"]
    assert received["language"] == "uz"
    assert received["mode"] == voice_fields.get("mode", "preset")
    for field, value in voice_fields.items():
        assert received[field] == value


@pytest.mark.parametrize("key", [
    "", " ", "../speech", "assistant\\welcome", 'assistant"welcome',
    "assistant\r\nX-Injected: true", "assistant..welcome", "assistant.", "o'zbek",
    "a" * 129, "CON", "aux.welcome", "Lpt9", 123, True, [],
])
def test_invalid_download_key_is_rejected_before_worker(service, key):
    client, engine = service
    response = client.post("/v1/audio/speech", headers=HEADERS,
                           json=dict(VALID_REQUEST, key=key))
    assert response.status_code == 422
    assert engine.operations == []


def test_cors_exposes_download_filename_to_configured_frontend():
    origin = "https://frontend.example"
    with TestClient(make_app(cors_origins=(origin,))) as client:
        wait_ready(client)
        response = client.post("/v1/audio/speech", headers={**HEADERS, "Origin": origin},
                               json=dict(VALID_REQUEST, key="assistant.welcome"))
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    exposed = {name.strip().lower() for name in response.headers["access-control-expose-headers"].split(",")}
    assert "content-disposition" in exposed
    assert response.headers["content-disposition"] == 'attachment; filename="assistant.welcome.wav"'


@pytest.mark.parametrize(
    "updates",
    [
        {"text": "   \n\t"},
        {"text": "private\x00text"},
        {"text": "private\x1btext"},
        {"text": 123},
        {"language": "not-a-language"},
        {"voice": "unknown_voice"},
        {"gender": "male"},
        {"gender": "other"},
        {"unknown_control": 1.25},
    ],
)
def test_invalid_requests_do_not_reach_inference(service, updates):
    client, engine = service
    response = client.post(
        "/v1/audio/speech", headers=HEADERS, json=dict(VALID_REQUEST, **updates)
    )
    assert response.status_code == 422
    assert engine.calls == []


def test_validation_error_does_not_echo_private_text_or_extra_field_value(service):
    client, _ = service
    private_text = "PRIVATE_TEXT_THAT_MUST_NOT_APPEAR_IN_ERRORS"
    private_value = "PRIVATE_EXTRA_FIELD_VALUE"
    response = client.post(
        "/v1/audio/speech",
        headers=HEADERS,
        json=dict(VALID_REQUEST, text=private_text, unknown=private_value),
    )
    assert response.status_code == 422
    assert private_text not in response.text
    assert private_value not in response.text
    assert API_KEY not in response.text


def test_character_limit_prevents_inference():
    engine = FakeEngine()
    with TestClient(make_app(engine, max_text_chars=10)) as client:
        wait_ready(client)
        response = client.post("/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST)
        assert response.status_code == 422
        assert engine.calls == []


def test_default_settings_disable_optional_limits_and_expiry(monkeypatch):
    names = ("MAX_TEXT_CHARS", "MAX_BODY_BYTES", "MAX_AUDIO_BYTES", "QUEUE_SIZE",
             "REQUEST_TIMEOUT", "JOB_RETENTION_SECONDS")
    for name in names:
        monkeypatch.delenv(f"TTS_{name}", raising=False)
    monkeypatch.setenv("TTS_API_KEY", API_KEY)
    settings = Settings.from_env()
    for name in names:
        assert getattr(settings, name.lower()) == 0


@pytest.mark.parametrize("already_exists", [True, False])
def test_configured_tmpdir_is_created_and_used_without_removing_other_files(tmp_path, monkeypatch, already_exists):
    directory = tmp_path / "data" / "tmp"
    unrelated = directory / "keep-this-file"
    if already_exists:
        directory.mkdir(parents=True)
        unrelated.write_bytes(b"existing data")
    stale_cache = tmp_path / "old-temp"
    stale_cache.mkdir()
    monkeypatch.setenv("TMPDIR", str(directory))
    monkeypatch.setattr(tempfile, "tempdir", str(stale_cache))
    engine = FakeEngine()
    app = make_app(engine)
    assert directory.is_dir()
    assert Path(tempfile.gettempdir()) == directory.resolve()
    assert app.state.jobs.root.parent == directory.resolve()
    staged = []
    original_temporary = tempfile.NamedTemporaryFile

    def track_staging(*args, **kwargs):
        target = original_temporary(*args, **kwargs)
        staged.append(Path(target.name))
        return target

    monkeypatch.setattr("api.tempfile.NamedTemporaryFile", track_staging)
    with TestClient(app) as client:
        wait_ready(client)
        response = client.post("/v1/audio/clone", headers=HEADERS,
            files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
            data={"text": "Salom.", "ref_text": "Reference."})
        assert response.status_code == 200
        assert staged and all(path.parent == directory.resolve() for path in staged)
    assert directory.is_dir()
    assert all(not path.exists() for path in staged)
    assert not app.state.jobs.root.exists()
    if already_exists:
        assert unrelated.read_bytes() == b"existing data"
    assert not tuple(stale_cache.iterdir())


@pytest.mark.parametrize("name", ["max_text_chars", "max_body_bytes", "max_audio_bytes", "queue_size",
                                  "request_timeout", "job_retention_seconds"])
@pytest.mark.parametrize("value", [-1, True, math.nan, math.inf])
def test_optional_limits_reject_invalid_settings(name, value):
    with pytest.raises(ValueError):
        Settings(api_key=API_KEY, **{name: value})


def test_unlimited_body_middleware_preserves_streaming_receive():
    async def scenario():
        calls = []

        async def receive():
            calls.append("received")
            return {"type": "http.request", "body": b"chunk", "more_body": True}

        async def send(message):
            raise AssertionError("The bypass must not send its own response")

        async def app(scope, actual_receive, actual_send):
            assert not calls
            assert actual_receive is receive
            assert actual_send is send
            assert (await actual_receive())["more_body"] is True

        await BodyLimitMiddleware(app, max_bytes=0)({"type": "http"}, receive, send)
        assert calls == ["received"]

    asyncio.run(scenario())


def test_default_speech_limit_accepts_long_text(service):
    client, engine = service
    text = "Uzun matn. " * 1000
    response = client.post("/v1/audio/speech", headers=HEADERS, json=dict(VALID_REQUEST, text=text))
    assert response.status_code == 200
    assert engine.calls[-1][0] == text.strip()


def test_default_upload_and_transcript_limits_accept_large_reference(service, monkeypatch):
    client, engine = service
    sizes = []
    original_read = StarletteUploadFile.read

    async def track_read(upload, size=-1):
        sizes.append(size)
        return await original_read(upload, size)

    monkeypatch.setattr(StarletteUploadFile, "read", track_read)
    reference = b"reference bytes" * 650000
    transcript = "Reference transcript. " * 100
    response = client.post("/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", reference, "audio/wav")},
        data={"text": "Salom.", "ref_text": transcript})
    assert response.status_code == 200
    operation, payload = engine.operations[-1]
    assert operation == "clone_speech"
    assert len(reference) > 8 * 1024 * 1024
    assert payload["audio"] == reference
    assert payload["ref_text"] == transcript.strip()
    assert sizes and all(0 < size <= 65536 for size in sizes)


@pytest.mark.parametrize("max_body_bytes", [0, 3 * 1024 * 1024])
@pytest.mark.parametrize("valid", [True, False])
def test_large_multipart_text_fields_preserve_native_upload_cleanup(monkeypatch, max_body_bytes, valid):
    engine = FakeEngine()
    closed = []
    original_close = StarletteUploadFile.close

    async def track_close(upload):
        await original_close(upload)
        closed.append(upload)

    monkeypatch.setattr(StarletteUploadFile, "close", track_close)
    text = "x" * (1024 * 1024 + 1)
    fields = {"ref_text": text}
    if valid:
        fields["text"] = text
    with TestClient(make_app(engine, max_body_bytes=max_body_bytes)) as client:
        wait_ready(client)
        response = client.post("/v1/audio/clone", headers=HEADERS,
            files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")}, data=fields)
        assert response.status_code == (200 if valid else 422)
        assert closed and all(upload.file.closed for upload in closed)
        if valid:
            assert engine.operations[-1][1]["text"] == text
            assert engine.operations[-1][1]["ref_text"] == text
        else:
            assert not engine.operations


def test_large_multipart_text_still_respects_configured_body_limit():
    engine = FakeEngine()
    with TestClient(make_app(engine, max_body_bytes=1024 * 1024)) as client:
        wait_ready(client)
        response = client.post("/v1/audio/clone", headers=HEADERS,
            files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
            data={"text": "Salom.", "ref_text": "x" * (1024 * 1024 + 1)})
        assert response.status_code == 413
        assert not engine.operations


def test_upload_staging_disk_failure_removes_temporary_file_and_returns_507(tmp_path, monkeypatch):
    engine = FakeEngine()
    app = make_app(engine)
    staged = []
    original_temporary = tempfile.NamedTemporaryFile

    def stage_in_test_directory(*args, **kwargs):
        target = original_temporary(*args, **kwargs, dir=tmp_path)
        staged.append(Path(target.name))
        return target

    monkeypatch.setattr("api.tempfile.NamedTemporaryFile", stage_in_test_directory)
    original_io = app.state.worker.io

    async def fail_staging_write(operation, *args):
        if getattr(operation, "__name__", None) == "write":
            raise OSError(errno.ENOSPC, "Disk is full")
        return await original_io(operation, *args)

    monkeypatch.setattr(app.state.worker, "io", fail_staging_write)
    with TestClient(app) as client:
        wait_ready(client)
        response = client.post("/v1/audio/clone", headers=HEADERS,
            files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
            data={"text": "Salom.", "ref_text": "Reference."})
        assert response.status_code == 507
        assert staged and all(not path.exists() for path in staged)
        assert not engine.operations


def test_cancellation_during_upload_staging_removes_file_before_admin_admission(tmp_path, monkeypatch):
    engine = FakeEngine()
    app = make_app(engine)
    staged = []
    original_temporary = tempfile.NamedTemporaryFile

    def stage_in_test_directory(*args, **kwargs):
        target = original_temporary(*args, **kwargs, dir=tmp_path)
        staged.append(Path(target.name))
        return target

    monkeypatch.setattr("api.tempfile.NamedTemporaryFile", stage_in_test_directory)

    async def scenario():
        writing = asyncio.Event()
        original_io = app.state.worker.io

        async def block_staging_write(operation, *args):
            if getattr(operation, "__name__", None) == "write":
                writing.set()
                await asyncio.Event().wait()
            return await original_io(operation, *args)

        monkeypatch.setattr(app.state.worker, "io", block_staging_write)
        async with app.router.lifespan_context(app):
            async with asyncio.timeout(3):
                while not app.state.worker.ready:
                    await asyncio.sleep(0.005)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                request = asyncio.create_task(client.post("/v1/audio/clone", headers=HEADERS,
                    files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
                    data={"text": "Salom.", "ref_text": "Reference."}))
                await asyncio.wait_for(writing.wait(), 3)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert staged and all(not path.exists() for path in staged)
                assert not engine.operations
                assert app.state.worker.stats()["admin_pending"] == 0

    asyncio.run(scenario())


def test_saved_voice_reference_text_has_no_character_quota(service):
    client, engine = service
    text = "Reference transcript. " * 100
    response = client.post("/v1/voices/design", headers=HEADERS,
        json={"name": "Long reference", "gender": "female", "instruct": "female", "reference_text": text})
    assert response.status_code == 201
    assert engine.operations[-1][1]["reference_text"] == text.strip()


def test_default_job_submission_accepts_more_than_one_thousand_items(service):
    client, _ = service
    response = client.post("/v1/audio/jobs", headers=HEADERS,
                           json={"items": [dict(VALID_REQUEST, text="x" * 1001)] * 1001})
    assert response.status_code == 202
    assert len(response.json()["jobs"]) == 1001


@pytest.mark.parametrize("route,payload", [
    ("/v1/audio/speech", VALID_REQUEST),
    ("/v1/voices/design", {"name": "My voice", "gender": "female", "instruct": "female"}),
])
def test_disabled_timeout_waits_for_real_model_completion(route, payload):
    engine = FakeEngine(block_first=True)
    app = make_app(engine, request_timeout=0)
    with TestClient(app) as client:
        wait_ready(client)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(client.post, route, headers=HEADERS, json=payload)
            try:
                assert engine.started.wait(timeout=2)
                time.sleep(0.05)
                assert not pending.done()
                assert engine.active == 1
            finally:
                engine.release.set()
            assert pending.result(timeout=2).status_code in {200, 201}


@pytest.mark.parametrize("claimed_length", [None, "1"])
def test_actual_body_limit_applies_even_if_content_length_is_wrong(claimed_length):
    engine = FakeEngine()
    with TestClient(make_app(engine, max_body_bytes=128)) as client:
        wait_ready(client)
        headers = dict(HEADERS, **{"Content-Type": "application/json"})
        if claimed_length:
            headers["Content-Length"] = claimed_length
        response = client.post("/v1/audio/speech", headers=headers, content=b" " * 129)
        assert response.status_code == 413
        assert engine.calls == []


def test_liveness_is_available_while_model_loads_but_speech_is_not():
    engine = FakeEngine(block_load=True)
    with TestClient(make_app(engine)) as client:
        try:
            assert engine.load_started.wait(timeout=2)
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 503
            assert client.post(
                "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST
            ).status_code == 503
            engine.allow_load.set()
            wait_ready(client)
            assert client.post(
                "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST
            ).status_code == 200
        finally:
            engine.allow_load.set()


def test_failed_model_load_is_not_ready_and_does_not_expose_internal_error():
    private_error = "PRIVATE_INTERNAL_MODEL_PATH_AND_TOKEN"
    engine = FakeEngine(load_error=RuntimeError(private_error))
    with TestClient(make_app(engine)) as client:
        assert engine.load_started.wait(timeout=2)
        assert client.get("/healthz").status_code == 200
        for response in [
            client.get("/readyz"),
            client.post("/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST),
        ]:
            assert response.status_code == 503
            assert private_error not in response.text


def test_generation_failure_is_safe_and_worker_accepts_the_next_request():
    private_error = "PRIVATE_INFERENCE_ERROR_AND_PATH"
    engine = FakeEngine(error=RuntimeError(private_error))
    with TestClient(make_app(engine)) as client:
        wait_ready(client)
        failed = client.post("/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST)
        assert failed.status_code in {500, 503}
        assert private_error not in failed.text
        engine.error = None
        assert client.post(
            "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST
        ).status_code == 200


def test_http_timeout_does_not_release_a_still_running_model():
    engine = FakeEngine(block_first=True)
    app = make_app(engine, request_timeout=0.2)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            first = client.post("/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST)
            assert first.status_code == 504
            assert engine.started.is_set()
            assert engine.active == 1
            with ThreadPoolExecutor(max_workers=1) as requests:
                second = requests.submit(
                    client.post, "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST
                )
                try:
                    wait_until(lambda: app.state.jobs.stats()["queued"] == 1, timeout=0.15)
                    assert len(engine.calls) == 1
                    assert engine.max_active == 1
                finally:
                    engine.release.set()
                assert second.result(timeout=2).status_code == 200
            assert len(engine.calls) == 2
            assert engine.max_active == 1
        finally:
            engine.release.set()


def test_durable_queue_accepts_more_work_without_starting_another_inference():
    engine = FakeEngine(block_first=True)
    app = make_app(engine, queue_size=1, request_timeout=3)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            with ThreadPoolExecutor(max_workers=2) as requests:
                first = requests.submit(
                    client.post, "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST
                )
                second = None
                try:
                    assert engine.started.wait(timeout=2)
                    second = requests.submit(
                        client.post,
                        "/v1/audio/speech",
                        headers=HEADERS,
                        json=VALID_REQUEST,
                    )
                    wait_until(lambda: app.state.jobs.stats()["queued"] == 1)
                    overflow = client.post(
                        "/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST] * 50}
                    )
                    assert overflow.status_code == 202
                    assert len(overflow.json()["jobs"]) == 50
                    assert app.state.jobs.stats()["queued"] == 51
                    assert len(engine.calls) == 1
                finally:
                    engine.release.set()
                assert first.result(timeout=2).status_code == 200
                assert second.result(timeout=2).status_code == 200
            wait_until(lambda: app.state.jobs.stats()["succeeded"] == 52)
            assert len(engine.calls) == 52
            assert engine.max_active == 1
        finally:
            engine.release.set()


def test_waiting_request_does_not_expire_before_execution_starts():
    engine = FakeEngine(block_first=True)
    app = make_app(engine, request_timeout=0.1)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            first = client.post("/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST)
            assert first.status_code == 504
            with ThreadPoolExecutor(max_workers=1) as requests:
                waiting = requests.submit(client.post, "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST)
                wait_until(lambda: app.state.jobs.stats()["queued"] == 1)
                time.sleep(0.15)
                assert not waiting.done()
                assert len(engine.calls) == 1
                engine.release.set()
                assert waiting.result(timeout=2).status_code == 200
            wait_until(lambda: not app.state.worker.busy)
            assert len(engine.calls) == 2
            recovered = client.post(
                "/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST
            )
            assert recovered.status_code == 200
            assert len(engine.calls) == 3
            assert engine.max_active == 1
        finally:
            engine.release.set()


def test_docs_serves_swagger_and_linked_voice_studio_assets(service):
    client, _ = service

    class AssetLinks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.scripts = set()
            self.styles = set()

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == "script" and values.get("src"):
                self.scripts.add(values["src"])
            if tag == "link" and values.get("rel") == "stylesheet":
                self.styles.add(values.get("href"))

    response = client.get("/docs")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    html = AssetLinks()
    html.feed(response.text)
    scripts = {urlsplit(url).path: url for url in html.scripts}
    styles = {urlsplit(url).path: url for url in html.styles if url}
    assert "/static/swagger-ui-bundle.js" in scripts
    assert "/static/swagger-ui.css" in styles
    assert "/static/voice-studio.js" in scripts
    assert "/static/voice-studio.css" in styles
    for asset, content_type in [
        (scripts["/static/voice-studio.js"], "javascript"),
        (styles["/static/voice-studio.css"], "text/css"),
    ]:
        served = client.get(asset)
        assert served.status_code == 200 and served.content
        assert content_type in served.headers["content-type"]
        if content_type == "javascript":
            import hashlib

            assert urlsplit(asset).query == "v=" + hashlib.sha256(served.content).hexdigest()[:12]

    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["HTTPBearer"]["scheme"] == "bearer"
    for method, route in [("get", "/v1/voices"), ("post", "/v1/voices/clone"),
                          ("post", "/v1/audio/speech"), ("delete", "/v1/voices/{voice_id}")]:
        assert schema["paths"][route][method]["security"] == [{"HTTPBearer": []}]


def test_openapi_describes_wav_response_and_bearer_auth(service):
    client, _ = service
    document = client.get("/openapi.json").json()
    response = document["paths"]["/v1/audio/speech"]["post"]["responses"]["200"]
    assert "audio/wav" in response["content"]
    schemes = document["components"]["securitySchemes"].values()
    assert any(item.get("scheme") == "bearer" for item in schemes)
    speech_schema = document["components"]["schemas"]["SpeechRequest"]
    assert "key" in speech_schema["properties"]
    assert "key" not in speech_schema["required"]
    delete = document["paths"]["/v1/voices/{voice_id}"]["delete"]
    assert delete["security"] == [{"HTTPBearer": []}]
    assert {"200", "403", "404", "503"} <= set(delete["responses"])
    assert delete["parameters"][0]["name"] == "voice_id"
    assert delete["parameters"][0]["required"] is True


def test_clone_form_schema_keeps_optional_swagger_inputs_empty(service):
    client, _ = service
    document = client.get("/openapi.json").json()
    for route, fields in [
        ("/v1/audio/clone", ["key", "ref_text", "instruct"]),
        ("/v1/voices/clone", ["ref_text"]),
    ]:
        reference = document["paths"][route]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
        schema = document["components"]["schemas"][reference.rsplit("/", 1)[-1]]
        for field in fields:
            property_schema = schema["properties"][field]
            assert field not in schema["required"]
            assert property_schema["examples"] == [""]
            assert {"type": "null"} in property_schema["anyOf"]
            assert property_schema.get("default") is None
        if route == "/v1/audio/clone":
            assert schema["properties"]["options"]["default"] == "{}"


def test_admin_worker_expiry_returns_http_timeout_instead_of_internal_cancellation():
    engine = FakeEngine()
    app = make_app(engine)
    with TestClient(app) as client:
        wait_ready(client)
        original_put = app.state.admin_worker.queue.put_nowait

        def expire_before_worker_runs(job):
            if job is not None:
                job.deadline = time.monotonic() - 1
            return original_put(job)

        app.state.admin_worker.queue.put_nowait = expire_before_worker_runs
        response = client.post("/v1/voices/design", headers=HEADERS,
                               json={"name": "Expired", "gender": "female", "instruct": "female"})
        assert response.status_code == 504
        assert engine.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Salom.", "mode": "design", "instruct": "female, young adult"},
        {"text": "Salom.", "mode": "auto", "language": None},
        {"text": "Salom.", "voice": "female_1", "language": None},
    ],
)
def test_design_auto_and_nullable_language_reach_worker(service, payload):
    client, engine = service
    response = client.post("/v1/audio/speech", headers=HEADERS, json=payload)
    assert response.status_code == 200
    operation, received = engine.operations[-1]
    assert operation == "speech"
    assert received["text"] == payload["text"]
    assert received["mode"] == payload.get("mode", "preset")
    assert received.get("language") is None
    if "instruct" in payload:
        assert received["instruct"] == payload["instruct"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Salom."},
        {"text": "Salom.", "mode": "design"},
        {"text": "Salom.", "mode": "design", "instruct": "   "},
        {"text": "Salom.", "mode": "design", "instruct": "female", "voice": "female_1"},
        {"text": "Salom.", "mode": "auto", "voice": "female_1"},
        {"text": "Salom.", "mode": "auto", "instruct": "female"},
        {"text": "Salom.", "mode": "auto", "gender": "female"},
        {"text": "Salom.", "mode": "other"},
    ],
)
def test_mode_field_conflicts_are_rejected_before_worker(service, payload):
    client, engine = service
    assert client.post("/v1/audio/speech", headers=HEADERS, json=payload).status_code == 422
    assert engine.operations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_step", 0), ("num_step", 65), ("num_step", True), ("num_step", 1.5),
        ("guidance_scale", -0.01), ("guidance_scale", 10.01),
        ("t_shift", 0), ("t_shift", 1.01),
        ("position_temperature", -0.01), ("position_temperature", 10.01),
        ("class_temperature", -0.01), ("class_temperature", 10.01),
        ("layer_penalty_factor", -0.01), ("layer_penalty_factor", 10.01),
        ("denoise", "false"), ("preprocess_prompt", 0), ("postprocess_output", "true"),
        ("audio_chunk_duration", 2.99), ("audio_chunk_duration", 30.01),
        ("audio_chunk_threshold", 2.99), ("audio_chunk_threshold", 60.01),
        ("pad_duration", -0.01), ("pad_duration", 1.01),
        ("fade_duration", -0.01), ("fade_duration", 1.01),
        ("not_a_control", 1),
    ],
)
def test_generation_controls_are_strict_and_bounded(service, field, value):
    client, engine = service
    request = dict(VALID_REQUEST, generation={field: value})
    assert client.post("/v1/audio/speech", headers=HEADERS, json=request).status_code == 422
    assert engine.operations == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed", 0.49), ("speed", 2.01), ("speed", "1.0"),
        ("duration", 0.99), ("duration", 30.01),
        ("seed", -1), ("seed", 4294967296), ("seed", True),
        ("normalize_text", "true"),
    ],
)
def test_speech_controls_are_strict_and_bounded(service, field, value):
    client, engine = service
    request = dict(VALID_REQUEST, **{field: value})
    assert client.post("/v1/audio/speech", headers=HEADERS, json=request).status_code == 422
    assert engine.operations == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["speed", "duration", "guidance_scale", "t_shift"])
def test_nonfinite_speech_and_generation_values_are_rejected(service, field, value):
    client, engine = service
    request = dict(VALID_REQUEST)
    if field in {"speed", "duration"}:
        request[field] = value
    else:
        request["generation"] = {field: value}
    # Raw JSON exercises the server: HTTPX correctly refuses json=NaN before sending.
    response = client.post(
        "/v1/audio/speech", headers={**HEADERS, "Content-Type": "application/json"},
        content=json.dumps(request, allow_nan=True),
    )
    assert response.status_code == 422
    assert engine.operations == []


def test_all_supported_controls_are_forwarded_without_being_silently_ignored(service):
    client, engine = service
    generation = {
        "num_step": 64, "guidance_scale": 10.0, "t_shift": 1.0,
        "position_temperature": 0.0, "class_temperature": 10.0,
        "layer_penalty_factor": 0.0, "denoise": False, "preprocess_prompt": False,
        "postprocess_output": False, "audio_chunk_duration": 30.0,
        "audio_chunk_threshold": 60.0, "pad_duration": 1.0, "fade_duration": 0.0,
    }
    request = dict(VALID_REQUEST, speed=0.5, duration=30.0, seed=4294967295,
                   normalize_text=True, generation=generation)
    response = client.post("/v1/audio/speech", headers=HEADERS, json=request)
    assert response.status_code == 200
    operation, received = engine.operations[-1]
    assert operation == "speech"
    assert received["generation"] == generation
    assert {field: received[field] for field in ["speed", "duration", "seed", "normalize_text"]} == {
        "speed": 0.5, "duration": 30.0, "seed": 4294967295, "normalize_text": True,
    }


def test_chunk_threshold_cannot_be_smaller_than_target_chunk(service):
    client, engine = service
    response = client.post(
        "/v1/audio/speech", headers=HEADERS,
        json=dict(VALID_REQUEST, generation={"audio_chunk_duration": 20.0, "audio_chunk_threshold": 10.0}),
    )
    assert response.status_code == 422
    assert engine.operations == []


def test_capabilities_exposes_controls_limits_and_real_engine_availability(service):
    client, engine = service
    assert client.get("/v1/capabilities").status_code == 401
    response = client.get("/v1/capabilities", headers=HEADERS)
    assert response.status_code == 200
    capabilities = response.json()
    assert capabilities["asr_available"] is True
    assert set(capabilities["modes"]) == {"preset", "design", "auto", "clone"}
    assert capabilities["voice_design_attributes"] == engine.voice_design_attributes
    assert all(value is None for value in capabilities["limits"].values())
    assert "selected speech region" in capabilities["reference_preparation"]
    assert capabilities["generation_schema"]["additionalProperties"] is False
    assert capabilities["generation_schema"]["properties"]["num_step"]["maximum"] == 64
    assert "[laughter]" in capabilities["nonverbal_tags"]
    assert capabilities["voice_design_training_languages"] == ["en", "zh"]
    assert set(capabilities["pronunciation"]) == {"English", "Chinese"}


def test_saved_design_voice_becomes_selectable_and_has_a_sample(service):
    client, engine = service
    request = {"name": "My designed voice", "gender": "female", "language": "uz",
               "instruct": "female, young adult", "seed": 123, "generation": {"num_step": 16}}
    response = client.post("/v1/voices/design", headers=HEADERS, json=request)
    assert response.status_code == 201
    voice = response.json()
    assert voice["name"] == request["name"]
    assert voice["synthetic"] is True
    assert engine.operations[-1][0] == "design_voice"
    assert engine.operations[-1][1]["generation"]["num_step"] == 16
    catalog = client.get("/v1/voices?gender=female", headers=HEADERS).json()["data"]
    assert voice["id"] in {item["id"] for item in catalog}
    sample = client.get(f"/v1/voices/{voice['id']}/sample", headers=HEADERS)
    assert sample.status_code == 200
    assert sample.content == engine.wav
    speech = client.post("/v1/audio/speech", headers=HEADERS,
                         json=dict(VALID_REQUEST, voice=voice["id"]))
    assert speech.status_code == 200


@pytest.mark.parametrize("updates", [{"speed": 1.0}, {"duration": 8.0}, {"name": " "},
                                     {"instruct": "female\x00"}, {"language": "unknown"}])
def test_saved_design_rejects_invalid_anchor_fields(service, updates):
    client, engine = service
    request = {"name": "My voice", "gender": "female", "instruct": "female"}
    response = client.post("/v1/voices/design", headers=HEADERS, json=dict(request, **updates))
    assert response.status_code == 422
    assert engine.operations == []


@pytest.mark.parametrize("key_fields,filename", [
    ({}, "speech.wav"),
    ({"key": "assistant.welcome"}, "assistant.welcome.wav"),
    ({"key": ""}, "speech.wav"),
    ({"key": " \t "}, "speech.wav"),
])
def test_one_shot_clone_forwards_reference_and_options(service, key_fields, filename):
    client, engine = service
    reference = make_wav(seconds=3)
    options = {"speed": 1.2, "seed": 91, "normalize_text": True,
               "generation": {"num_step": 16, "denoise": False}}
    response = client.post(
        "/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", reference, "audio/wav")},
        data={"text": "Davom etish uchun tilni tanlang.", "language": "uz", "ref_text": " Reference transcript. ",
              "instruct": " female ", "options": json.dumps(options), **key_fields},
    )
    assert response.status_code == 200
    assert response.content == engine.wav
    assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
    operation, payload = engine.operations[-1]
    assert operation == "clone_speech"
    assert "key" not in payload
    assert payload["text"] == "Davom etish uchun tilni tanlang."
    assert payload["audio"] == reference
    assert payload["ref_text"] == "Reference transcript."
    assert payload["instruct"] == "female"
    assert payload["speed"] == 1.2
    assert payload["seed"] == 91
    assert payload["generation"]["num_step"] == 16


@pytest.mark.parametrize("key", [
    "../speech", "assistant\\welcome", 'assistant"welcome',
    "assistant\r\nX-Injected: true", "assistant..welcome", " CON ", "NUL", "a" * 129,
])
def test_invalid_multipart_download_key_does_not_reach_worker(service, key):
    client, engine = service
    response = client.post(
        "/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
        data={"text": "Salom.", "key": key, "ref_text": "Reference."},
    )
    assert response.status_code == 422
    assert engine.operations == []


@pytest.mark.parametrize("optional_fields", [
    {}, {"ref_text": "", "instruct": "", "key": ""},
    {"ref_text": " \t ", "instruct": " \t ", "key": " \t "},
])
def test_blank_clone_fields_delegate_transcription_and_reference_voice(service, optional_fields):
    client, engine = service
    response = client.post(
        "/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
        data={"text": "Salom.", **optional_fields},
    )
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    operation, payload = engine.operations[-1]
    assert operation == "clone_speech"
    assert payload["ref_text"] is None and payload["instruct"] is None
    assert "key" not in payload


@pytest.mark.parametrize("ref_text_fields", [{}, {"ref_text": ""}, {"ref_text": " \t "}])
def test_saved_clone_blank_transcript_delegates_to_asr(service, ref_text_fields):
    client, engine = service
    response = client.post(
        "/v1/voices/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
        data={"name": "My voice", "gender": "female", **ref_text_fields},
    )
    assert response.status_code == 201
    operation, payload = engine.operations[-1]
    assert operation == "clone_voice" and payload["ref_text"] is None


@pytest.mark.parametrize("filename,media_type", [
    ("reference.wav", "audio/wav"),
    ("recording.m4a", "audio/mp4"),
    ("voice.ogg", "audio/ogg"),
    ("blob", "application/octet-stream"),
    ("looks-like.wav", "audio/webm"),
])
def test_saved_clone_and_asr_use_same_operation_queue(service, filename, media_type):
    client, engine = service
    reference = make_wav(seconds=3)
    saved = client.post(
        "/v1/voices/clone", headers=HEADERS,
        files={"ref_audio": (filename, reference, media_type)},
        data={"name": "My reference voice", "gender": "male", "reference_language": "uz",
              "ref_text": "Sample transcript.", "preprocess_prompt": "false"},
    )
    assert saved.status_code == 201
    voice = saved.json()
    assert voice["synthetic"] is False
    assert engine.operations[-1][0] == "clone_voice"
    assert engine.operations[-1][1]["preprocess_prompt"] is False
    assert client.post(
        "/v1/audio/speech", headers=HEADERS,
        json=dict(VALID_REQUEST, voice=voice["id"], gender="male"),
    ).status_code == 200
    transcription = client.post(
        "/v1/audio/transcriptions", headers=HEADERS,
        files={"audio": (filename, reference, media_type)},
    )
    assert transcription.status_code == 200
    assert transcription.json() == {"text": "Assalomu alaykum."}
    assert engine.operations[-1] == ("transcribe", {"audio": reference})
    assert engine.max_active == 1


@pytest.mark.parametrize("options", ["not JSON", "[]", '{"unknown": 1}',
                                     '{"generation": {"num_step": 65}}', '{"speed": NaN}'])
def test_invalid_multipart_options_do_not_reach_worker(service, options):
    client, engine = service
    response = client.post(
        "/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
        data={"text": "Salom.", "ref_text": "Reference.", "options": options},
    )
    assert response.status_code == 422
    assert engine.operations == []


@pytest.mark.parametrize("route,field,data", [
    ("/v1/audio/clone", "ref_audio", {"text": "Salom.", "ref_text": "Reference."}),
    ("/v1/voices/clone", "ref_audio", {"name": "Voice", "gender": "female", "ref_text": "Reference."}),
    ("/v1/audio/transcriptions", "audio", {}),
])
def test_missing_empty_and_oversized_uploads_are_rejected(route, field, data):
    engine = FakeEngine()
    with TestClient(make_app(engine, max_audio_bytes=128)) as client:
        wait_ready(client)
        missing = client.post(route, headers=HEADERS, data=data)
        assert missing.status_code == 422
        empty = client.post(route, headers=HEADERS,
                            files={field: ("reference.wav", b"", "audio/wav")}, data=data)
        assert empty.status_code == 422
        oversized = client.post(route, headers=HEADERS,
                                files={field: ("reference.wav", b"x" * 129, "audio/wav")}, data=data)
        assert oversized.status_code == 413
        assert engine.operations == []


@pytest.mark.parametrize("ref_text_fields", [{}, {"ref_text": ""}, {"ref_text": " \t "}])
def test_transcription_requirement_is_enforced_and_explicit_text_works_without_asr(ref_text_fields):
    engine = FakeEngine(asr_available=False)
    with TestClient(make_app(engine)) as client:
        wait_ready(client)
        reference = {"ref_audio": ("reference.wav", engine.wav, "audio/wav")}
        unavailable = client.post("/v1/audio/clone", headers=HEADERS, files=reference,
                                  data={"text": "Salom.", **ref_text_fields})
        assert unavailable.status_code == 503
        saved_unavailable = client.post("/v1/voices/clone", headers=HEADERS, files=reference,
                                       data={"name": "My voice", "gender": "female", **ref_text_fields})
        assert saved_unavailable.status_code == 503
        assert engine.operations == []
        transcribe = client.post("/v1/audio/transcriptions", headers=HEADERS,
                                 files={"audio": ("reference.wav", engine.wav, "audio/wav")})
        assert transcribe.status_code == 503
        assert client.post("/v1/audio/clone", headers=HEADERS, files=reference,
                           data={"text": "Salom.", "ref_text": "Reference transcript."}).status_code == 200


@pytest.mark.parametrize("ref_text", ["private\x00text"])
def test_invalid_reference_transcript_is_rejected(service, ref_text):
    client, engine = service
    response = client.post(
        "/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
        data={"text": "Salom.", "ref_text": ref_text},
    )
    assert response.status_code == 422
    assert engine.operations == []


def test_engine_reference_validation_errors_are_redacted(service):
    client, engine = service
    engine.error = ValueError("PRIVATE_REFERENCE_DETAILS_MUST_NOT_LEAK")
    response = client.post(
        "/v1/audio/clone", headers=HEADERS,
        files={"ref_audio": ("reference.wav", engine.wav, "audio/wav")},
        data={"text": "Salom.", "ref_text": "Reference."},
    )
    assert response.status_code == 422
    assert "PRIVATE_REFERENCE_DETAILS_MUST_NOT_LEAK" not in response.text


def test_unknown_sample_is_404_and_never_runs_worker(service):
    client, engine = service
    assert client.get("/v1/voices/unknown/sample", headers=HEADERS).status_code == 404
    assert engine.operations == []


@pytest.mark.parametrize("source", ["clone", "design"])
def test_saved_voice_can_be_reused_and_deleted_from_the_catalog(service, source):
    client, engine = service
    original = client.get("/v1/voices", headers=HEADERS).json()["data"]
    if source == "clone":
        saved = client.post(
            "/v1/voices/clone", headers=HEADERS,
            files={"ref_audio": ("phone.m4a", engine.wav, "audio/mp4")},
            data={"name": "Reusable voice", "gender": "female", "ref_text": "Reference transcript."},
        )
    else:
        saved = client.post("/v1/voices/design", headers=HEADERS,
                            json={"name": "Reusable voice", "gender": "female", "instruct": "female"})
    assert saved.status_code == 201
    voice_id = saved.json()["id"]
    assert client.get(f"/v1/voices/{voice_id}/sample", headers=HEADERS).status_code == 200
    speech = client.post("/v1/audio/speech", headers=HEADERS,
                         json=dict(VALID_REQUEST, voice=voice_id, key="assistant.welcome"))
    assert speech.status_code == 200
    assert speech.headers["content-disposition"] == 'attachment; filename="assistant.welcome.wav"'

    deleted = client.delete(f"/v1/voices/{voice_id}", headers=HEADERS)
    assert deleted.status_code == 200
    assert deleted.json() == {"id": voice_id, "deleted": True}
    assert deleted.headers["cache-control"] == "no-store"
    assert engine.operations[-1] == ("delete_voice", {"voice": voice_id})
    assert client.get("/v1/voices", headers=HEADERS).json()["data"] == original
    assert client.get(f"/v1/voices/{voice_id}/sample", headers=HEADERS).status_code == 404
    assert client.post("/v1/audio/speech", headers=HEADERS,
                       json=dict(VALID_REQUEST, voice=voice_id)).status_code == 422
    assert client.delete(f"/v1/voices/{voice_id}", headers=HEADERS).status_code == 404


@pytest.mark.parametrize("authorization", [None, "Bearer incorrect", "Basic invalid"])
def test_delete_voice_requires_authentication_before_lookup(service, authorization):
    client, engine = service
    headers = {"Authorization": authorization} if authorization else {}
    response = client.delete("/v1/voices/unknown", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert engine.operations == []


@pytest.mark.parametrize("voice_id", ["female_1", "female_2", "male_1", "male_2"])
def test_delete_presets_is_forbidden_without_changing_the_catalog(service, voice_id):
    client, engine = service
    original = [dict(voice) for voice in engine.voices]
    response = client.delete(f"/v1/voices/{voice_id}", headers=HEADERS)
    assert response.status_code == 403
    assert engine.voices == original
    assert engine.operations == [("delete_voice", {"voice": voice_id})]
    assert client.get(f"/v1/voices/{voice_id}/sample", headers=HEADERS).status_code == 200


@pytest.mark.parametrize("voice_id", ["unknown", "clone_" + "f" * 32])
def test_delete_unknown_voice_is_404_from_the_worker(service, voice_id):
    client, engine = service
    response = client.delete(f"/v1/voices/{voice_id}", headers=HEADERS)
    assert response.status_code == 404
    assert engine.operations == [("delete_voice", {"voice": voice_id})]


def test_delete_voice_is_unavailable_while_model_loads():
    engine = FakeEngine(block_load=True)
    with TestClient(make_app(engine)) as client:
        try:
            assert engine.load_started.wait(timeout=2)
            response = client.delete("/v1/voices/unknown", headers=HEADERS)
            assert response.status_code == 503
            assert engine.operations == []
        finally:
            engine.allow_load.set()


@pytest.mark.parametrize("error,status", [
    (VoiceNotFoundError("PRIVATE_VOICE_PATH"), 404),
    (ProtectedVoiceError("PRIVATE_VOICE_PATH"), 403),
    (VoiceDeletionError("PRIVATE_VOICE_PATH"), 503),
])
def test_voice_domain_errors_are_redacted_and_worker_recovers(error, status):
    engine = FakeEngine(error=error)
    with TestClient(make_app(engine)) as client:
        wait_ready(client)
        response = client.delete("/v1/voices/unknown", headers=HEADERS)
        assert response.status_code == status
        assert "PRIVATE_VOICE_PATH" not in response.text
        if status == 503:
            assert int(response.headers["retry-after"]) > 0
            assert "same voice ID" in response.json()["detail"]
        engine.error = None
        assert client.post("/v1/audio/speech", headers=HEADERS, json=VALID_REQUEST).status_code == 200


def test_delete_waits_for_running_speech_and_keeps_voice_available_until_then():
    engine = FakeEngine(block_first=True)
    voice_id = "clone_" + "a" * 32
    engine.voices.append({**engine.voices[0], "id": voice_id})
    app = make_app(engine)
    with TestClient(app) as client:
        wait_ready(client)
        with ThreadPoolExecutor(max_workers=2) as requests:
            speech = requests.submit(client.post, "/v1/audio/speech", headers=HEADERS,
                                     json=dict(VALID_REQUEST, voice=voice_id))
            try:
                assert engine.started.wait(timeout=2)
                deletion = requests.submit(client.delete, f"/v1/voices/{voice_id}", headers=HEADERS)
                wait_until(lambda: app.state.worker.stats()["admin_pending"] == 1)
                assert [operation for operation, _ in engine.operations] == ["speech"]
                assert voice_id in {voice["id"] for voice in client.get("/v1/voices", headers=HEADERS).json()["data"]}
            finally:
                engine.release.set()
            assert speech.result(timeout=2).status_code == 200
            assert deletion.result(timeout=2).status_code == 200
        assert engine.max_active == 1
        assert [operation for operation, _ in engine.operations] == ["speech", "delete_voice"]


@pytest.mark.parametrize("queued_operation", ["delete", "sample"])
def test_voice_lookup_happens_after_earlier_queued_deletion(queued_operation):
    engine = FakeEngine(block_first=True)
    voice_id = "clone_" + "b" * 32
    engine.voices.append({**engine.voices[0], "id": voice_id})
    app = make_app(engine)
    with TestClient(app) as client:
        wait_ready(client)
        with ThreadPoolExecutor(max_workers=2) as requests:
            first = requests.submit(client.delete, f"/v1/voices/{voice_id}", headers=HEADERS)
            try:
                assert engine.started.wait(timeout=2)
                if queued_operation == "delete":
                    second = requests.submit(client.delete, f"/v1/voices/{voice_id}", headers=HEADERS)
                else:
                    second = requests.submit(client.get, f"/v1/voices/{voice_id}/sample", headers=HEADERS)
                if queued_operation == "delete":
                    wait_until(lambda: app.state.worker.stats()["admin_pending"] == 1)
                else:
                    time.sleep(0.03)
                    assert not second.done()
                assert [operation for operation, _ in engine.operations] == ["delete_voice"]
            finally:
                engine.release.set()
            assert first.result(timeout=2).status_code == 200
            assert second.result(timeout=2).status_code == 404
        assert engine.max_active == 1
        assert [operation for operation, _ in engine.operations] == (
            ["delete_voice", "delete_voice"] if queued_operation == "delete" else ["delete_voice"]
        )


def test_configured_frontend_can_preflight_and_send_delete():
    origin = "https://frontend.example"
    with TestClient(make_app(cors_origins=(origin,))) as client:
        wait_ready(client)
        request_headers = {"Origin": origin, "Access-Control-Request-Method": "DELETE",
                           "Access-Control-Request-Headers": "Authorization"}
        allowed = client.options("/v1/voices/unknown", headers=request_headers)
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == origin
        assert "DELETE" in allowed.headers["access-control-allow-methods"].split(", ")
        denied = client.options("/v1/voices/unknown", headers={**request_headers, "Origin": "https://unlisted.example"})
        assert denied.status_code == 400
        response = client.delete("/v1/voices/unknown", headers={**HEADERS, "Origin": origin})
        assert response.status_code == 404
        assert response.headers["access-control-allow-origin"] == origin


def test_batch_returns_binary_zip_and_forwards_all_items(service):
    client, engine = service
    items = [dict(VALID_REQUEST, key="assistant.welcome"), {"text": "Hello.", "mode": "auto", "language": "en"}]
    response = client.post("/v1/audio/batch", headers=HEADERS, json={"items": items})
    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".wav")]
        assert len(names) == 2
        for name in names:
            with wave.open(io.BytesIO(archive.read(name)), "rb") as audio:
                assert audio.getframerate() == 24000
                assert audio.getnframes() > 0
    operation, payload = engine.operations[-1]
    assert operation == "batch"
    assert [item["text"] for item in payload["items"]] == [item["text"] for item in items]
    assert payload["items"][0]["key"] == "assistant.welcome"
    assert "key" not in payload["items"][1]
    assert payload["items"][1]["mode"] == "auto"


@pytest.mark.parametrize("items", [
    [],
    [dict(VALID_REQUEST), dict(VALID_REQUEST, language="unknown")],
    [dict(VALID_REQUEST), dict(VALID_REQUEST, voice="unknown")],
    [dict(VALID_REQUEST), dict(VALID_REQUEST, gender="male")],
])
def test_invalid_batch_rejects_all_items_before_any_generation(service, items):
    client, engine = service
    response = client.post("/v1/audio/batch", headers=HEADERS, json={"items": items})
    assert response.status_code == 422
    assert engine.operations == []


def test_batch_accepts_more_than_four_items_and_long_text(service):
    client, _ = service
    response = client.post("/v1/audio/batch", headers=HEADERS,
                           json={"items": [dict(VALID_REQUEST, text="x" * 2001)] * 5})
    assert response.status_code == 200


def test_profile_creation_timeout_keeps_speech_serialized_until_profile_work_finishes():
    engine = FakeEngine(block_first=True)
    app = make_app(engine, request_timeout=0.2)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            first = client.post("/v1/voices/design", headers=HEADERS,
                                json={"name": "Queued profile", "gender": "female", "instruct": "female"})
            assert first.status_code == 504
            assert engine.active == 1
            with ThreadPoolExecutor(max_workers=1) as requests:
                second = requests.submit(client.post, "/v1/audio/speech", headers=HEADERS,
                                         json=VALID_REQUEST)
                try:
                    time.sleep(0.04)
                    assert not second.done()
                    assert app.state.admin_worker.busy
                    assert len(engine.operations) == 1
                    assert engine.operations[0][0] == "design_voice"
                    assert engine.max_active == 1
                finally:
                    engine.release.set()
                assert second.result(timeout=2).status_code == 200
            assert [operation for operation, _ in engine.operations] == ["design_voice", "speech"]
            assert engine.max_active == 1
        finally:
            engine.release.set()


def test_jobs_accept_50_items_and_expose_individual_status_and_downloads(tmp_path):
    engine = FakeEngine(block_first=True)
    app = make_app(engine, jobs_dir=tmp_path)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            payload = {"items": [dict(VALID_REQUEST, text=f"Line {index}", key=f"clip.{index}")
                                 for index in range(50)]}
            accepted = client.post("/v1/audio/jobs", headers=HEADERS, json=payload)
            assert accepted.status_code == 202
            group = accepted.json()
            assert len(group["jobs"]) == 50
            assert len({job["id"] for job in group["jobs"]}) == 50
            assert str(tmp_path) not in accepted.text
            assert engine.started.wait(timeout=2)
            status = client.get(f"/v1/audio/jobs/{group['id']}", headers=HEADERS)
            assert status.status_code == 200
            assert status.json()["counts"] == {"queued": 49, "running": 1, "succeeded": 0, "failed": 0}
            listing = client.get("/v1/audio/jobs", headers=HEADERS).json()
            assert listing["data"][0]["id"] == group["id"]
            assert listing["execution"]["parallel_workers"] == 1
            assert listing["execution"]["queued"] == 49
            assert client.get(f"/v1/audio/jobs/{group['id']}/{group['jobs'][0]['id']}/audio",
                              headers=HEADERS).status_code == 409
            engine.release.set()
            wait_until(lambda: app.state.jobs.stats()["succeeded"] == 50)
            final = client.get(f"/v1/audio/jobs/{group['id']}", headers=HEADERS).json()
            assert final["state"] == "succeeded"
            assert final["counts"]["succeeded"] == 50
            result = client.get(final["jobs"][0]["download_url"], headers=HEADERS)
            assert result.status_code == 200
            assert result.content == engine.wav
            assert result.headers["content-disposition"] == 'attachment; filename="clip.0.wav"'
            assert result.headers["cache-control"] == "no-store"
            assert result.headers["x-audio-sample-rate"] == "24000"
            assert engine.max_active == 1
            assert [payload["generation"]["num_step"] for op, payload in engine.operations if op == "speech"] == [32] * 50
            assert all("key" not in payload for op, payload in engine.operations if op == "speech")
        finally:
            engine.release.set()


def test_completed_job_can_download_while_another_job_is_running(tmp_path):
    class SecondBlockedEngine(FakeEngine):
        def synthesize(self, text, language, voice):
            if text == "Second":
                self.started.set()
                assert self.release.wait(timeout=5)
            return super().synthesize(text, language, voice)

    engine = SecondBlockedEngine()
    app = make_app(engine, jobs_dir=tmp_path)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            group = client.post("/v1/audio/jobs", headers=HEADERS,
                                json={"items": [dict(VALID_REQUEST, text=text) for text in ("First", "Second")]}).json()
            wait_until(lambda: app.state.jobs.stats()["succeeded"] == 1 and app.state.jobs.stats()["running"] == 1)
            status = client.get(f"/v1/audio/jobs/{group['id']}", headers=HEADERS).json()
            assert [job["state"] for job in status["jobs"]] == ["succeeded", "running"]
            assert client.get(status["jobs"][0]["download_url"], headers=HEADERS).content == engine.wav
        finally:
            engine.release.set()


@pytest.mark.parametrize("route,method", [
    ("/v1/audio/jobs", "get"), ("/v1/audio/jobs", "post"),
    ("/v1/audio/jobs/" + "a" * 32, "get"),
    ("/v1/audio/jobs/" + "a" * 32 + "/" + "b" * 32 + "/audio", "get"),
])
def test_job_routes_require_authentication(service, route, method):
    client, _ = service
    request = getattr(client, method)
    kwargs = {"json": {"items": [VALID_REQUEST]}} if method == "post" else {}
    assert request(route, **kwargs).status_code == 401


@pytest.mark.parametrize("items", [[],
    [VALID_REQUEST, dict(VALID_REQUEST, voice="missing")],
    [VALID_REQUEST, dict(VALID_REQUEST, language="missing")]])
def test_invalid_job_group_is_rejected_atomically(service, items):
    client, engine = service
    result = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": items})
    assert result.status_code == 422
    assert engine.calls == []
    assert client.get("/v1/audio/jobs", headers=HEADERS).json()["data"] == []


def test_job_history_is_available_while_model_loads_and_survives_restart(tmp_path):
    with TestClient(make_app(jobs_dir=tmp_path)) as client:
        wait_ready(client)
        group = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).json()
        wait_until(lambda: client.get(f"/v1/audio/jobs/{group['id']}", headers=HEADERS).json()["state"] == "succeeded")
    blocked = FakeEngine(block_load=True)
    with TestClient(make_app(blocked, jobs_dir=tmp_path)) as client:
        try:
            assert blocked.load_started.wait(timeout=2)
            assert client.get("/readyz").status_code == 503
            listing = client.get("/v1/audio/jobs", headers=HEADERS)
            assert listing.status_code == 200
            assert listing.json()["data"][0]["id"] == group["id"]
            stored = client.get(f"/v1/audio/jobs/{group['id']}", headers=HEADERS).json()
            assert client.get(stored["jobs"][0]["download_url"], headers=HEADERS).status_code == 200
            assert client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).status_code == 503
        finally:
            blocked.allow_load.set()


def test_job_download_checks_group_membership_and_artifact_integrity(tmp_path):
    app = make_app(jobs_dir=tmp_path)
    with TestClient(app) as client:
        wait_ready(client)
        first = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).json()
        second = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).json()
        wait_until(lambda: app.state.jobs.stats()["succeeded"] == 2)
        first_id = first["jobs"][0]["id"]
        assert client.get(f"/v1/audio/jobs/{second['id']}/{first_id}/audio", headers=HEADERS).status_code == 404
        assert client.get("/v1/audio/jobs/not-a-uuid", headers=HEADERS).status_code == 404
        path = app.state.jobs.get_result_path(first_id)
        path.write_bytes(b"PRIVATE corrupt audio data")
        result = client.get(f"/v1/audio/jobs/{first['id']}/{first_id}/audio", headers=HEADERS)
        assert result.status_code == 404
        assert "PRIVATE" not in result.text
        assert str(tmp_path) not in result.text


def test_insufficient_job_storage_returns_safe_507(service, monkeypatch):
    client, _ = service

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "PRIVATE disk path and contents")

    monkeypatch.setattr(client.app.state.jobs, "enqueue", no_space)
    result = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]})
    assert result.status_code == 507
    assert "PRIVATE" not in result.text


def test_history_keeps_all_active_groups_and_paginates_terminal_groups(tmp_path):
    engine = FakeEngine(block_first=True)
    app = make_app(engine, jobs_dir=tmp_path)
    with TestClient(app) as client:
        try:
            wait_ready(client)
            groups = [client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).json()
                      for _ in range(23)]
            listing = client.get("/v1/audio/jobs?limit=2&offset=0", headers=HEADERS).json()
            assert {group["id"] for group in listing["data"]} == {group["id"] for group in groups}
            assert listing["history"] == {"limit": 2, "offset": 0, "has_more": False}
            engine.release.set()
            wait_until(lambda: app.state.jobs.stats()["succeeded"] == 23)
            first = client.get("/v1/audio/jobs?limit=2&offset=0", headers=HEADERS).json()
            second = client.get("/v1/audio/jobs?limit=2&offset=2", headers=HEADERS).json()
            assert len(first["data"]) == len(second["data"]) == 2
            assert first["history"]["has_more"] is True
            assert second["history"] == {"limit": 2, "offset": 2, "has_more": True}
            assert {group["id"] for group in first["data"]}.isdisjoint(group["id"] for group in second["data"])
            assert client.get("/v1/audio/jobs?limit=1001", headers=HEADERS).status_code == 422
            assert client.get("/v1/audio/jobs?offset=-1", headers=HEADERS).status_code == 422
        finally:
            engine.release.set()


def test_slow_job_metadata_does_not_block_health_or_readiness(service, monkeypatch):
    client, _ = service
    group = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).json()
    entered = threading.Event()
    release = threading.Event()
    original = client.app.state.jobs.get_group

    def slow_get(group_id):
        entered.set()
        assert release.wait(timeout=3)
        return original(group_id)

    monkeypatch.setattr(client.app.state.jobs, "get_group", slow_get)
    with ThreadPoolExecutor(max_workers=1) as requests:
        pending = requests.submit(client.get, f"/v1/audio/jobs/{group['id']}", headers=HEADERS)
        try:
            assert entered.wait(timeout=2)
            started = time.monotonic()
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 200
            assert time.monotonic() - started < 0.5
            assert not pending.done()
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 200


def test_download_lease_survives_expiry_before_first_response_read(tmp_path, monkeypatch):
    app = make_app(jobs_dir=tmp_path, job_retention_seconds=604800)
    with TestClient(app) as client:
        wait_ready(client)
        group = client.post("/v1/audio/jobs", headers=HEADERS, json={"items": [VALID_REQUEST]}).json()
        wait_until(lambda: app.state.jobs.stats()["succeeded"] == 1)
        job_id = group["jobs"][0]["id"]
        original = app.state.jobs.acquire_result
        leases = []

        def expire_after_open(group_id, item_id=None):
            lease = original(group_id, item_id)
            leases.append(lease)
            with app.state.jobs._lock:
                app.state.jobs._db.execute("UPDATE jobs SET finished_at=? WHERE id=?", (time.time() - 700000, job_id))
            app.state.jobs.cleanup()
            return lease

        monkeypatch.setattr(app.state.jobs, "acquire_result", expire_after_open)
        result = client.get(f"/v1/audio/jobs/{group['id']}/{job_id}/audio", headers=HEADERS)
        assert result.status_code == 200
        assert result.content == make_wav()
        assert result.headers["content-length"] == str(len(result.content))
        assert len(leases) == 1
        assert leases[0].stream.closed
        app.state.jobs.cleanup()
        with pytest.raises(KeyError):
            app.state.jobs.get_job(job_id)


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_download_closes_lease_when_sending_fails_or_is_cancelled(failure):
    class Lease:
        filename = "speech.wav"
        sample_rate = 24000

        def __init__(self):
            self.stream = io.BytesIO(make_wav())
            self.size = len(self.stream.getvalue())

        def close(self):
            self.stream.close()

    async def scenario():
        service = JobService(None, None, None, None)
        lease = Lease()

        async def send(message):
            raise failure()

        async def receive():
            return {"type": "http.disconnect"}

        try:
            response = LeasedAudioResponse(lease, service)
            with pytest.raises(failure):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            assert lease.stream.closed
        finally:
            await service.stop()
    asyncio.run(scenario())
