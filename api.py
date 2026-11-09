"""Authenticated speech API with durable FIFO jobs and bounded inference."""

import asyncio
import errno
import hashlib
import io
import json
import logging
import math
import os
import secrets
import tempfile
import time
import unicodedata
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.params import Form as FormParameter
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from engine import ProtectedVoiceError, VoiceDeletionError, VoiceNotFoundError, speech_filename
from job_service import JobService
from jobs import JobNotReadyError, JobStore

logger = logging.getLogger("text_speak")


class LeasedAudioResponse(StreamingResponse):
    """Retain an already-open result descriptor through send errors and disconnects."""

    def __init__(self, lease, service):
        self.lease = lease
        self.service = service

        async def chunks():
            while True:
                chunk = await service.io(lease.stream.read, 65536)
                if not chunk:
                    return
                yield chunk

        super().__init__(chunks(), media_type="audio/wav", headers={
            "Content-Disposition": f'attachment; filename="{lease.filename}"',
            "Content-Length": str(lease.size), "Cache-Control": "no-store",
            "X-Audio-Sample-Rate": str(lease.sample_rate),
        })

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.service.close_result(self.lease)


@dataclass(frozen=True)
class Settings:
    api_key: str
    max_text_chars: int = 0
    max_body_bytes: int = 0
    max_audio_bytes: int = 0
    queue_size: int = 0
    request_timeout: float = 0.0
    jobs_dir: str | Path | None = None
    parallel_workers: int = 2
    job_retention_seconds: float = 0.0
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.api_key) < 32 or self.api_key != self.api_key.strip():
            raise ValueError("TTS_API_KEY must contain at least 32 characters without outer spaces")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (self.max_text_chars, self.max_body_bytes, self.max_audio_bytes, self.queue_size)):
            raise ValueError("Request and queue limits must be nonnegative integers; zero disables a limit")
        if (isinstance(self.request_timeout, bool) or not isinstance(self.request_timeout, (int, float))
                or not math.isfinite(self.request_timeout) or self.request_timeout < 0):
            raise ValueError("Request timeout must be finite and nonnegative; zero disables the timeout")
        if (isinstance(self.parallel_workers, bool) or not isinstance(self.parallel_workers, int)
                or not 1 <= self.parallel_workers <= 4):
            raise ValueError("Parallel worker count must be between 1 and 4")
        if (isinstance(self.job_retention_seconds, bool) or not isinstance(self.job_retention_seconds, (int, float))
                or not math.isfinite(self.job_retention_seconds) or self.job_retention_seconds < 0):
            raise ValueError("Job retention must be finite and nonnegative; zero disables expiry")

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.environ.get("TTS_API_KEY", ""),
            max_text_chars=int(os.environ.get("TTS_MAX_TEXT_CHARS", "0")),
            max_body_bytes=int(os.environ.get("TTS_MAX_BODY_BYTES", "0")),
            max_audio_bytes=int(os.environ.get("TTS_MAX_AUDIO_BYTES", "0")),
            queue_size=int(os.environ.get("TTS_QUEUE_SIZE", "0")),
            request_timeout=float(os.environ.get("TTS_REQUEST_TIMEOUT", "0")),
            jobs_dir=os.environ.get("TTS_JOBS_DIR", "/data/jobs"),
            parallel_workers=int(os.environ.get("TTS_PARALLEL_WORKERS", "2")),
            job_retention_seconds=float(os.environ.get("TTS_JOB_RETENTION_SECONDS", "0")),
            cors_origins=tuple(x.strip() for x in os.environ.get("TTS_CORS_ORIGINS", "").split(",") if x.strip()),
        )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class GenerationOptions(StrictModel):
    num_step: int = Field(default=32, ge=1, le=64)
    guidance_scale: float = Field(default=2.0, ge=0, le=10)
    t_shift: float = Field(default=0.1, gt=0, le=1)
    position_temperature: float = Field(default=5.0, ge=0, le=10)
    class_temperature: float = Field(default=0.0, ge=0, le=10)
    layer_penalty_factor: float = Field(default=5.0, ge=0, le=10)
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = Field(default=15.0, ge=3, le=30)
    audio_chunk_threshold: float = Field(default=30.0, ge=3, le=60)
    pad_duration: float = Field(default=0.1, ge=0, le=1)
    fade_duration: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def chunk_bounds(self):
        if self.audio_chunk_threshold < self.audio_chunk_duration:
            raise ValueError("audio_chunk_threshold must be at least audio_chunk_duration")
        return self


class SpeechOptions(StrictModel):
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    duration: float | None = Field(default=None, ge=1, le=30, description="Overrides speed; postprocessing may trim silence. Fixed duration bypasses long-form chunking.")
    normalize_text: bool = False
    seed: int | None = Field(default=None, ge=0, le=4294967295)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class SpeechRequest(SpeechOptions):
    text: str = Field(min_length=1, description="Text already written in the selected language; no translation.")
    key: str | None = Field(default=None, max_length=128, examples=["assistant.welcome"], description="Optional download filename stem: dot-separated ASCII letters, digits, underscores or hyphens; .wav is appended. Windows reserved names are rejected.")
    language: str | None = Field(default=None, min_length=1, max_length=32, examples=["uz"], description="Supported code, or null for language-agnostic generation.")
    mode: Literal["preset", "design", "auto"] = "preset"
    voice: str | None = Field(default=None, min_length=1, max_length=64, examples=["female_1"])
    instruct: str | None = Field(default=None, min_length=1, max_length=500, examples=["female, young adult, moderate pitch"])
    gender: Literal["female", "male"] | None = Field(default=None, description="Optional consistency check against the selected voice.")

    @field_validator("key")
    @classmethod
    def validate_key(cls, value):
        speech_filename(value)
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank")
        if any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in value):
            raise ValueError("Text contains unsupported control characters")
        return value

    @field_validator("instruct")
    @classmethod
    def validate_instruct(cls, value):
        if value is not None:
            value = value.strip()
            if not value or any(unicodedata.category(c) == "Cc" for c in value):
                raise ValueError("Instruction must contain valid speaker attributes")
        return value

    @model_validator(mode="after")
    def mode_fields(self):
        if self.mode == "preset" and self.voice is None:
            raise ValueError("preset mode requires voice")
        if self.mode == "design" and (not self.instruct or self.voice is not None):
            raise ValueError("design mode requires instruct and does not accept voice")
        if self.mode == "auto" and any(value is not None for value in (self.voice, self.instruct, self.gender)):
            raise ValueError("auto mode does not accept voice, instruct, or gender")
        return self


class DesignVoiceRequest(SpeechOptions):
    name: str = Field(min_length=1, max_length=80)
    gender: Literal["female", "male"]
    language: str = Field(default="uz", min_length=1, max_length=32)
    instruct: str = Field(min_length=1, max_length=500)
    reference_text: str = Field(default="Assalomu alaykum. Bugun siz bilan ravon va tushunarli ovozda suhbatlashamiz. Har bir so'z aniq va tabiiy talaffuz qilinadi.", min_length=1)

    @field_validator("name", "instruct", "reference_text")
    @classmethod
    def validate_strings(cls, value):
        value = value.strip()
        if not value or any(unicodedata.category(c) == "Cc" for c in value):
            raise ValueError("Value must contain valid nonblank text")
        return value

    @model_validator(mode="after")
    def anchor_controls(self):
        if self.duration is not None or self.speed is not None:
            raise ValueError("A saved design voice uses a fixed 8-second reference; set speed/duration during speech generation")
        return self


class BatchRequest(StrictModel):
    items: list[SpeechRequest] = Field(min_length=1)


class JobSubmissionRequest(StrictModel):
    items: list[SpeechRequest] = Field(min_length=1)


@dataclass
class SpeechJob:
    operation: str
    payload: dict
    future: asyncio.Future
    deadline: float


class SpeechWorker:
    """The executor is owned here, so a disconnected client never unlocks the GPU."""

    def __init__(self, engine, settings):
        self.engine = engine
        self.settings = settings
        self.queue = asyncio.Queue(maxsize=settings.queue_size)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-model")
        self.ready = False
        self.busy = False
        self.failed = False
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.run(), name="tts-worker")

    async def run(self):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self.executor, self.engine.load)
        except Exception:
            self.failed = True
            logger.exception("TTS model initialization failed")
            return
        self.ready = True
        logger.info("TTS model ready: %d languages, %d voices", len(self.engine.languages), len(self.engine.voices))
        while True:
            job = await self.queue.get()
            try:
                if job is None:
                    return
                if job.future.cancelled() or time.monotonic() >= job.deadline:
                    if not job.future.done():
                        job.future.set_exception(TimeoutError("Speech request expired in queue"))
                    continue
                self.busy = True
                try:
                    result = await loop.run_in_executor(self.executor, self.engine.execute, job.operation, job.payload)
                except (VoiceNotFoundError, ProtectedVoiceError, VoiceDeletionError) as error:
                    if isinstance(error, VoiceDeletionError):
                        logger.error("Voice profile deletion could not complete", exc_info=False)
                    if not job.future.done():
                        job.future.set_exception(type(error)())
                except ValueError:
                    if not job.future.done():
                        job.future.set_exception(ValueError("Invalid reference audio or generation settings; see /v1/capabilities"))
                except Exception:
                    # Do not include exception text: third-party errors can contain submitted text.
                    logger.error("Speech generation failed", exc_info=False)
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("Speech generation failed"))
                else:
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    self.busy = False
            finally:
                self.queue.task_done()

    async def stop(self):
        self.ready = False
        # Discard waiting clients; finish actual model work before executor shutdown.
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if job is not None and not job.future.done():
                job.future.cancel()
            self.queue.task_done()
        if self.task and not self.task.done():
            await self.queue.put(None)
            await self.task
        self.executor.shutdown(wait=True)

    async def speech(self, body):
        return await self.submit("speech", body.model_dump(exclude_none=True, exclude={"key"}))

    async def submit(self, operation, payload):
        if not self.ready:
            raise HTTPException(503, "Model is not ready", headers={"Retry-After": "10"})
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        timeout = self.settings.request_timeout or None
        deadline = time.monotonic() + timeout if timeout is not None else math.inf
        job = SpeechJob(operation, payload, future, deadline)
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            raise HTTPException(429, "Speech queue is full; retry later", headers={"Retry-After": "5"}) from None
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        except TimeoutError:
            raise HTTPException(504, "Speech request timed out; retry later") from None
        except VoiceNotFoundError:
            raise HTTPException(404, "Voice was not found") from None
        except ProtectedVoiceError:
            raise HTTPException(403, "Built-in voices cannot be deleted") from None
        except VoiceDeletionError:
            raise HTTPException(503, "Voice deletion could not complete; retry DELETE with the same voice ID", headers={"Retry-After": "5"}) from None
        except RuntimeError:
            raise HTTPException(503, "Speech generation failed; retry later") from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        finally:
            # Cancellation does not stop the dedicated model worker or its running thread.
            if not future.done():
                future.cancel()


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.max_bytes:
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await JSONResponse({"detail": "Request body is too large"}, status_code=413)(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, engine=None):
    settings = settings or Settings.from_env()
    if configured_temp := os.environ.get("TMPDIR"):
        temporary_root = Path(configured_temp).expanduser()
        if temporary_root.is_symlink():
            raise ValueError("TMPDIR must be a directory, not a symbolic link")
        # Existing data volumes can hide the directory prepared in the image.
        # Create it as the application user before Starlette or the engine spool
        # uploads, and replace any earlier cached fallback to /tmp.
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=temporary_root):
            pass
        tempfile.tempdir = str(temporary_root.resolve())

    class ReferenceRequest(Request):
        def form(self, *, max_files=1000, max_fields=1000, max_part_size=None):
            # FastAPI owns parsing and upload cleanup. Override only the hidden
            # per-text-field byte ceiling, using the configured whole-body cap.
            return super().form(max_files=max_files, max_fields=max_fields,
                max_part_size=(settings.max_body_bytes or math.inf) if max_part_size is None else max_part_size)

    class ReferenceRoute(APIRoute):
        def get_route_handler(self):
            handler = super().get_route_handler()
            if self.body_field is None or not isinstance(self.body_field.field_info, FormParameter):
                return handler

            async def accept_reference(request):
                return await handler(ReferenceRequest(request.scope, request.receive, request._send))

            return accept_reference

    injected_engine = engine is not None
    if engine is None:
        from engine import OmniVoiceEngine

        graph_setting = os.environ.get("TTS_FLASHINFER_GRAPHS", "0")
        if graph_setting not in {"0", "1"}:
            raise ValueError("TTS_FLASHINFER_GRAPHS must be 0 or 1")
        download_setting = os.environ.get("TTS_AUTO_DOWNLOAD_MODELS", "1")
        if download_setting not in {"0", "1"}:
            raise ValueError("TTS_AUTO_DOWNLOAD_MODELS must be 0 or 1")
        engine_config = dict(
            model_id=os.environ.get("TTS_MODEL_ID", "k2-fsa/OmniVoice"),
            model_revision=os.environ.get("TTS_MODEL_REVISION", "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"),
            cache_dir=os.environ.get("HF_HOME", "/data/huggingface"),
            voices_dir=os.environ.get("TTS_VOICES_DIR", "/data/voices"),
            device=os.environ.get("TTS_DEVICE", "cuda:0"),
            num_steps=32,
            model_path=os.environ.get("TTS_MODEL_PATH"),
            asr_model_path=os.environ.get("TTS_ASR_MODEL_PATH"),
            acceleration=os.environ.get("TTS_ACCELERATION", "none"),
            flashinfer_graphs=graph_setting == "1",
            tensorrt_plan=os.environ.get("TTS_TENSORRT_PLAN"),
            tensorrt_manifest=os.environ.get("TTS_TENSORRT_MANIFEST"),
            auto_download_models=download_setting == "1",
        )
        # Profile creation/transcription stays on its validated parent backend;
        # private speech processes own the optional TensorRT execution contexts.
        admin_config = {**engine_config, "flashinfer_graphs": False}
        if engine_config["acceleration"] == "tensorrt":
            admin_config["acceleration"] = "flashinfer"
        engine = OmniVoiceEngine(**admin_config)
        # The owner provisions models first; child processes only open those
        # same local files and never start their own large downloads.
        engine_config.update(model_path=engine.model_path, asr_model_path=engine.asr_model_path,
                             auto_download_models=False)
    from inference_pool import LocalSpeechBackend, ProcessSpeechBackend

    temporary_jobs = tempfile.TemporaryDirectory(prefix="text-speak-jobs-") if injected_engine and settings.jobs_dir is None else None
    jobs_root = Path(temporary_jobs.name if temporary_jobs else settings.jobs_dir or "/data/jobs")
    store = JobStore(jobs_root, retention_seconds=settings.job_retention_seconds)
    admin_worker = SpeechWorker(engine, settings)
    backend = (LocalSpeechBackend(engine, admin_worker.executor, jobs_root) if injected_engine else
               ProcessSpeechBackend(engine_config, jobs_root, slots=settings.parallel_workers,
                                    execution_timeout=settings.request_timeout))
    worker = JobService(engine, admin_worker, backend, store, request_timeout=settings.request_timeout)

    @asynccontextmanager
    async def lifespan(app):
        await admin_worker.start()
        await worker.start()
        try:
            yield
        finally:
            try:
                await worker.stop()
            finally:
                try:
                    await admin_worker.stop()
                finally:
                    await asyncio.to_thread(store.close)
                    if temporary_jobs is not None:
                        await asyncio.to_thread(temporary_jobs.cleanup)

    app = FastAPI(
        title="Text Speak API",
        version="2.2.0",
        description=("OmniVoice inference API: preset voices, voice design, auto voice, audio cloning, "
                     "saved voice profiles, transcription, generation controls and batch ZIP output. "
                     "Use Authorize with a Bearer API key. Audio is 24 kHz mono PCM16 WAV. "
                     "Text must already be in the selected language. "
                     "Durable FIFO speech jobs use bounded parallel inference; waiting jobs do not expire in the queue. "
                     "Request, text, upload and execution limits are disabled by default; see /v1/capabilities for configured limits. "
                     "OmniVoice pretrained model: CC-BY-NC."),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.worker = worker
    app.state.admin_worker = admin_worker
    app.state.jobs = store
    app.router.route_class = ReferenceRoute
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.get("/docs", include_in_schema=False)
    async def api_docs():
        response = get_swagger_ui_html(
            openapi_url="/openapi.json", title="Text Speak API",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
            swagger_favicon_url="data:,",
            swagger_ui_parameters={"persistAuthorization": False},
        )
        html = response.body.decode("utf-8")
        html = html.replace("</head>", '<link rel="stylesheet" href="/static/voice-studio.css">\n</head>', 1)
        studio_version = hashlib.sha256((Path(__file__).parent / "static/voice-studio.js").read_bytes()).hexdigest()[:12]
        html = html.replace("</body>", f'<script src="/static/voice-studio.js?v={studio_version}" defer></script>\n</body>', 1)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"],
            expose_headers=["Content-Disposition"],
        )
    bearer = HTTPBearer(auto_error=False)

    async def authorize(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if credentials is None or not secrets.compare_digest(credentials.credentials.encode(), settings.api_key.encode()):
            raise HTTPException(401, "Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})

    async def require_ready():
        if not worker.ready:
            raise HTTPException(503, "Model is not ready", headers={"Retry-After": "10"})

    secured = [Depends(authorize), Depends(require_ready)]
    authenticated = [Depends(authorize)]

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error):
        # Pydantic's default response echoes invalid inputs; keep submitted text private.
        errors = [{"loc": item["loc"], "type": item["type"], "msg": item["msg"]} for item in error.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.get("/healthz", tags=["Health"])
    async def health():
        return {"status": "ok"}

    @app.get("/readyz", tags=["Health"])
    async def ready():
        return JSONResponse({"status": "ready" if worker.ready else ("failed" if worker.failed else "loading")}, status_code=200 if worker.ready else 503)

    @app.get("/v1/languages", tags=["Catalog"], dependencies=secured)
    async def languages():
        return {"data": engine.languages}

    @app.get("/v1/voices", tags=["Catalog"], dependencies=secured)
    async def voices(gender: Annotated[Literal["female", "male"] | None, Query()] = None):
        async with worker.catalog_lock:
            return {"data": [dict(voice) for voice in engine.voices if gender is None or voice["gender"] == gender]}

    @app.get("/v1/capabilities", tags=["Catalog"], dependencies=secured)
    async def capabilities():
        acceleration = {}
        if getattr(backend, "config", {}).get("acceleration") == "tensorrt":
            acceleration = {"inference_acceleration": "tensorrt+flashinfer",
                            "administrative_acceleration": engine.capabilities.get("inference_acceleration")}
        elif (getattr(backend, "config", {}).get("acceleration") == "flashinfer"
              and backend.config.get("flashinfer_graphs")):
            acceleration = {"inference_acceleration": "flashinfer+cuda-graphs",
                            "administrative_acceleration": engine.capabilities.get("inference_acceleration")}
        return {
            **getattr(engine, "capabilities", {}),
            **acceleration,
            "modes": ["preset", "design", "auto", "clone"],
            "voice_design_attributes": getattr(engine, "voice_design_attributes", {}),
            "generation_schema": GenerationOptions.model_json_schema(),
            "limits": {"text_characters": settings.max_text_chars or None, "body_bytes": settings.max_body_bytes or None,
                       "audio_bytes": settings.max_audio_bytes or None, "reference_seconds": None,
                       "batch_items": None, "batch_total_characters": None,
                       "job_batch_items": None, "queue_size": None, "admin_queue_size": settings.queue_size or None,
                       "request_timeout": settings.request_timeout or None,
                       "job_retention_seconds": settings.job_retention_seconds or None},
            "reference_preparation": "Audio is converted automatically. Long references use a selected speech region with a matching automatic transcript when needed.",
            "execution": await execution_status(),
            "nonverbal_tags": ["[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]", "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]"],
            "pronunciation": {"English": "CMU phones: [B EY1 S]", "Chinese": "Pinyin with tone numbers: ZHE2"},
            "voice_design_training_languages": ["en", "zh"],
        }

    def validate_language(language):
        if language is not None and language not in {item["code"] for item in engine.languages}:
            raise HTTPException(422, "Unsupported language; use GET /v1/languages")

    async def validate_speech(body):
        if settings.max_text_chars and len(body.text) > settings.max_text_chars:
            raise HTTPException(422, f"Text must contain at most {settings.max_text_chars} characters")
        validate_language(body.language)
        if body.mode == "preset":
            async with worker.catalog_lock:
                voice = next((item for item in engine.voices if item["id"] == body.voice), None)
                if voice is None:
                    raise HTTPException(422, "Unknown voice; use GET /v1/voices")
                if body.gender is not None and body.gender != voice["gender"]:
                    raise HTTPException(422, "Gender does not match selected voice")

    async def execution_status():
        effective = (len(backend.worker_pids) if hasattr(backend, "worker_pids")
                     else backend.slots if worker.ready else 0)
        return {"parallel_workers": effective, "ready": worker.ready,
                "configured_workers": getattr(backend, "requested_slots", backend.slots),
                "capacity_reason": getattr(backend, "capacity_reason", None), **await worker.storage("stats")}

    def storage_error(error):
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            return HTTPException(507, "Insufficient storage for speech jobs; retry after freeing space")
        return HTTPException(503, "Speech job storage is unavailable; retry later")

    @app.post("/v1/audio/jobs", tags=["Speech jobs"], dependencies=secured, status_code=202)
    async def submit_jobs(body: JobSubmissionRequest):
        for item in body.items:
            await validate_speech(item)
        try:
            group = await worker.enqueue([item.model_dump(exclude_none=True) for item in body.items])
        except (OSError, ValueError) as error:
            raise storage_error(error) from None
        return JSONResponse(group, status_code=202, headers={"Cache-Control": "no-store"})

    @app.get("/v1/audio/jobs", tags=["Speech jobs"], dependencies=authenticated)
    async def list_jobs(limit: Annotated[int, Query(ge=1, le=1000)] = 20,
                        offset: Annotated[int, Query(ge=0)] = 0):
        try:
            await worker.cleanup_if_due()
            data = await worker.storage("list_groups", limit, offset, True)
            has_more = await worker.storage("history_has_more", limit, offset)
        except OSError as error:
            raise storage_error(error) from None
        return JSONResponse({"data": data, "execution": await execution_status(),
                             "history": {"limit": limit, "offset": offset, "has_more": has_more}},
                            headers={"Cache-Control": "no-store"})

    @app.get("/v1/audio/jobs/{group_id}", tags=["Speech jobs"], dependencies=authenticated)
    async def get_jobs(group_id: str):
        try:
            group = await worker.storage("get_group", group_id)
        except KeyError:
            raise HTTPException(404, "Speech job group was not found") from None
        return JSONResponse(group, headers={"Cache-Control": "no-store"})

    @app.get("/v1/audio/jobs/{group_id}/{job_id}/audio", tags=["Speech jobs"], dependencies=authenticated,
             response_class=StreamingResponse)
    async def download_job(group_id: str, job_id: str):
        try:
            lease = await worker.storage("acquire_result", group_id, job_id)
        except KeyError:
            raise HTTPException(404, "Speech audio was not found") from None
        except JobNotReadyError:
            raise HTTPException(409, "Speech audio is not ready") from None
        try:
            return LeasedAudioResponse(lease, worker)
        except BaseException:
            await worker.close_result(lease)
            raise

    def wav_response(data, filename="speech.wav"):
        with wave.open(io.BytesIO(data), "rb") as audio:
            rate = audio.getframerate()
        return Response(data, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store", "X-Audio-Sample-Rate": str(rate)})

    def parse_options(raw):
        try:
            return SpeechOptions.model_validate_json(raw)
        except (ValidationError, ValueError):
            raise HTTPException(422, "options must be a valid JSON object of speech controls") from None

    async def read_reference(upload):
        path = None
        try:
            # Multipart files already spool to disk. Keep only one small read
            # buffer in memory while transferring ownership to the admin queue.
            with tempfile.NamedTemporaryFile(prefix="text-speak-upload-", delete=False) as target:
                path = Path(target.name)
                size = 0
                while chunk := await upload.read(65536):
                    size += len(chunk)
                    if settings.max_audio_bytes and size > settings.max_audio_bytes:
                        raise HTTPException(413, "Reference audio is too large")
                    await worker.io(target.write, chunk)
                if not size:
                    raise HTTPException(422, "Reference audio is empty")
            await upload.close()
            return path
        except BaseException as error:
            try:
                await upload.close()
            finally:
                await worker.discard_upload(path)
            if isinstance(error, OSError):
                raise storage_error(error) from None
            raise

    def ensure_asr_available(ref_text):
        if ref_text is None and not getattr(engine, "capabilities", {}).get("asr_available", False):
            raise HTTPException(503, "Automatic transcription is not installed; provide ref_text")

    def validate_reference_text(ref_text):
        if ref_text is not None:
            ref_text = ref_text.strip()
            if not ref_text:
                ref_text = None
            elif any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in ref_text):
                raise HTTPException(422, "ref_text must contain valid text")
        ensure_asr_available(ref_text)
        return ref_text

    @app.post(
        "/v1/audio/speech", tags=["Speech"], dependencies=secured,
        response_class=Response,
        responses={
            200: {"description": "Complete mono PCM16 WAV audio", "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}},
            **{status: {"description": detail} for status, detail in {
                401: "Missing or invalid Bearer API key", 413: "Request body too large", 422: "Invalid input",
                507: "Insufficient storage", 503: "Model loading or generation failed", 504: "Execution timed out",
            }.items()},
        },
    )
    async def speech(body: SpeechRequest):
        await validate_speech(body)
        try:
            wav = await worker.speech(body)
        except (OSError, ValueError) as error:
            raise storage_error(error) from None
        return wav_response(wav, speech_filename(body.key))

    @app.post("/v1/voices/design", tags=["Voices"], dependencies=secured, status_code=201)
    async def design_voice(body: DesignVoiceRequest):
        validate_language(body.language)
        result = await worker.submit_admin("design_voice", body.model_dump(exclude_none=True))
        return JSONResponse(result, status_code=201, headers={"Cache-Control": "no-store"})

    @app.post("/v1/voices/clone", tags=["Voices"], dependencies=secured, status_code=201)
    async def clone_voice(
        ref_audio: Annotated[UploadFile, File(description="Nonempty audio reference: WAV, FLAC, MP3, M4A/AAC, OGG/Opus, WebM, AMR, 3GP, WMA and other formats listed in /v1/capabilities. Audio is converted automatically; long references use a selected speech region with a matching automatic transcript when needed.")],
        name: Annotated[str, Form(min_length=1, max_length=80)],
        gender: Annotated[Literal["female", "male"], Form()],
        reference_language: Annotated[str | None, Form(max_length=32)] = None,
        ref_text: Annotated[str | None, Form(examples=[""])] = None,
        preprocess_prompt: Annotated[bool, Form()] = True,
    ):
        validate_language(reference_language)
        ref_text = validate_reference_text(ref_text)
        name = name.strip()
        if not name or any(unicodedata.category(c) == "Cc" for c in name):
            raise HTTPException(422, "name must contain valid nonblank text")
        data = await read_reference(ref_audio)
        result = await worker.submit_admin("clone_voice", {"audio": data, "ref_text": ref_text, "name": name,
            "gender": gender, "reference_language": reference_language, "preprocess_prompt": preprocess_prompt}, owned_upload=data)
        return JSONResponse(result, status_code=201, headers={"Cache-Control": "no-store"})

    @app.post("/v1/audio/clone", tags=["Speech"], dependencies=secured, response_class=Response,
              responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
    async def clone_speech(
        ref_audio: Annotated[UploadFile, File(description="Nonempty audio reference; phone and common audio formats are converted automatically. Long references use a selected speech region with a matching automatic transcript when needed. See /v1/capabilities.")],
        text: Annotated[str, Form(min_length=1)],
        key: Annotated[str | None, Form(max_length=128, examples=[""], description="Optional download filename key, for example assistant.welcome -> assistant.welcome.wav. Blank or omitted -> speech.wav.")] = None,
        language: Annotated[str | None, Form(max_length=32)] = None,
        ref_text: Annotated[str | None, Form(examples=[""], description="Exact words spoken in the reference audio. Leave blank for automatic transcription. If a long reference needs a selected region, that region is transcribed automatically.")] = None,
        instruct: Annotated[str | None, Form(max_length=500, examples=[""], description="Optional supported speaker attributes. Leave blank to use the reference voice without additional instructions.")] = None,
        options: Annotated[str, Form(description="JSON: speed, duration, seed, normalize_text, generation")] = "{}",
    ):
        controls = parse_options(options)
        try:
            body = SpeechRequest(text=text, key=key if key and key.strip() else None, language=language, mode="auto", **controls.model_dump())
        except ValidationError:
            raise HTTPException(422, "Invalid text, download key, language or speech controls") from None
        await validate_speech(body)
        ref_text = validate_reference_text(ref_text)
        if instruct is not None:
            instruct = instruct.strip() or None
        data = await read_reference(ref_audio)
        result = await worker.submit_admin("clone_speech", {**body.model_dump(exclude_none=True, exclude={"key"}), "audio": data,
            "ref_text": ref_text, "instruct": instruct}, owned_upload=data)
        return wav_response(result, speech_filename(body.key))

    @app.post("/v1/audio/transcriptions", tags=["Speech"], dependencies=secured)
    async def transcribe(audio: Annotated[UploadFile, File(description="Nonempty audio file; phone and common audio formats are converted automatically. See /v1/capabilities.")]):
        ensure_asr_available(None)
        data = await read_reference(audio)
        result = await worker.submit_admin("transcribe", {"audio": data}, owned_upload=data)
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/v1/voices/{voice_id}/sample", tags=["Voices"], dependencies=secured, response_class=Response,
             responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
    async def voice_sample(voice_id: str):
        async with worker.catalog_lock:
            if voice_id not in {item["id"] for item in engine.voices}:
                raise HTTPException(404, "Voice was not found")
        return wav_response(await worker.submit_admin("voice_sample", {"voice": voice_id}), "voice-sample.wav")

    @app.delete(
        "/v1/voices/{voice_id}", tags=["Voices"], dependencies=secured,
        description="Permanently delete a saved clone/design voice, its reference audio and cached prompt. Built-in voices are protected. Use an ID returned by GET /v1/voices.",
        responses={
            200: {"description": "Saved voice deleted", "content": {"application/json": {"example": {"id": "clone_...", "deleted": True}}}},
            403: {"description": "Built-in voice is protected"},
            404: {"description": "Voice was not found"},
            503: {"description": "Deletion could not complete; retry with the same voice ID"},
        },
    )
    async def delete_voice(voice_id: str):
        result = await worker.submit_admin("delete_voice", {"voice": voice_id})
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.post("/v1/audio/batch", tags=["Speech"], dependencies=secured, response_class=Response,
              responses={200: {"content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}}}})
    async def batch(body: BatchRequest):
        for item in body.items:
            await validate_speech(item)
        result = await worker.submit_admin("batch", {"items": [item.model_dump(exclude_none=True) for item in body.items]})
        return Response(result, media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="audio-batch.zip"', "Cache-Control": "no-store"})

    return app
