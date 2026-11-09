"""OmniVoice inference and persistent, synthetic speaker profiles."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext
from collections import OrderedDict
from dataclasses import asdict, fields
import copy
import importlib.util
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
import zipfile


logger = logging.getLogger(__name__)

VOICE_SPECS = (
    {"id": "female_1", "gender": "female", "name": "Dilnoza", "instruct": "female, young adult, moderate pitch", "reference_language": "uz", "synthetic": True},
    {"id": "female_2", "gender": "female", "name": "Madina", "instruct": "female, middle-aged, low pitch", "reference_language": "uz", "synthetic": True},
    {"id": "male_1", "gender": "male", "name": "Jasur", "instruct": "male, young adult, moderate pitch", "reference_language": "uz", "synthetic": True},
    {"id": "male_2", "gender": "male", "name": "Sardor", "instruct": "male, middle-aged, low pitch", "reference_language": "uz", "synthetic": True},
)
_SEEDS = {"female_1": 1729, "female_2": 3079, "male_1": 4567, "male_2": 6007}
_REFERENCE_TEXT = (
    "Assalomu alaykum. Bugun siz bilan ravon va tushunarli ovozda suhbatlashamiz. "
    "Har bir so'z aniq va tabiiy talaffuz qilinadi."
)
_PROFILE_VERSION = 1
_PROFILE_FILES = ("prompt.pt", "reference.wav", "manifest.json")
_CUSTOM_VOICE_ID = r"(?:clone|design)_[0-9a-f]{32}"
REFERENCE_AUDIO_FORMATS = ("wav", "flac", "mp3", "m4a", "aac", "ogg", "opus", "webm", "amr", "3gp", "wma", "aiff", "caf", "mp4", "mkv", "avi", "au", "ac3", "eac3", "ape", "wv", "tta")
# Only self-contained media containers: playlists and external input protocols are excluded.
_AUDIO_DEMUXERS = "aac,ac3,eac3,aiff,amr,amrnb,amrwb,ape,asf,asf_o,au,avi,caf,dts,flac,matroska,webm,mov,mp4,m4a,3gp,3g2,mj2,mp3,mpeg,mpegts,ogg,oma,qcp,rm,tak,tta,voc,w64,wav,wv"


def speech_filename(key: str | None, default: str = "speech.wav") -> str:
    if key is None:
        return default
    if not isinstance(key, str) or len(key) > 128 or not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", key):
        raise ValueError("key must contain 1 to 128 ASCII letters, digits, underscores or hyphens in dot-separated segments")
    if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", key.split(".")[0], re.IGNORECASE):
        raise ValueError("key must not use a reserved Windows filename")
    return f"{key}.wav"


class VoiceNotFoundError(LookupError):
    """The requested saved voice is unavailable."""


class ProtectedVoiceError(PermissionError):
    """Built-in voice identities cannot be deleted."""


class VoiceDeletionError(RuntimeError):
    """A voice deletion needs storage repair or another cleanup attempt."""


class _SegmentBudgetError(ValueError):
    """A task needs smaller model working sets, rather than rejection."""


class WorkerRestartRequired(RuntimeError):
    """A failed private CUDA context must be replaced before another job."""

    recoverable = True


class OmniVoiceEngine:
    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        cache_dir: str | Path,
        voices_dir: str | Path,
        device: str = "cuda:0",
        num_steps: int = 32,
        model_path: str | Path | None = None,
        asr_model_path: str | Path | None = None,
        acceleration: str = "none",
        flashinfer_graphs: bool = False,
        tensorrt_plan: str | Path | None = None,
        tensorrt_manifest: str | Path | None = None,
        profile_writer: bool = True,
        auto_download_models: bool = True,
    ):
        if not model_id or not model_revision:
            raise ValueError("An explicit model ID and revision are required.")
        if not 1 <= num_steps <= 64:
            raise ValueError("num_steps must be between 1 and 64.")
        if acceleration not in ("none", "flashinfer", "tensorrt"):
            raise ValueError("acceleration must be none, flashinfer or tensorrt.")
        if not isinstance(flashinfer_graphs, bool):
            raise ValueError("flashinfer_graphs must be a boolean.")
        if not isinstance(auto_download_models, bool):
            raise ValueError("auto_download_models must be a boolean.")
        if acceleration == "tensorrt" and (not tensorrt_plan or not tensorrt_manifest):
            raise ValueError("TensorRT requires a pinned plan and manifest.")
        self.model_id = model_id
        self.model_revision = model_revision
        self.cache_dir = Path(cache_dir)
        self.voices_dir = Path(voices_dir)
        self.device = device
        self.num_steps = num_steps
        self.acceleration = acceleration
        self.flashinfer_graphs = flashinfer_graphs
        self.tensorrt_plan = Path(tensorrt_plan) if tensorrt_plan else None
        self.tensorrt_manifest = Path(tensorrt_manifest) if tensorrt_manifest else None
        self._trt_backend = None
        self._flashinfer_backend = None
        self._active_acceleration = "none"
        self._profile_writer = profile_writer
        self._pinned_prompts = OrderedDict()
        self.model_path = Path(model_path) if model_path else None
        self.asr_model_path = Path(asr_model_path).resolve() if asr_model_path else None
        self.auto_download_models = auto_download_models
        if auto_download_models:
            from model_assets import MODELS

            pinned = MODELS["omnivoice"]
            if self.model_path is None and (model_id, model_revision) == (pinned["repo_id"], pinned["revision"]):
                self.model_path = self.cache_dir / "models" / "omnivoice"
            if self.asr_model_path is None:
                self.asr_model_path = self.cache_dir / "models" / "whisper"
        self.sample_rate = 0
        self.languages: list[dict[str, str]] = []
        self.voices = [
            {key: spec[key] for key in ("id", "gender", "name", "reference_language", "synthetic")}
            for spec in VOICE_SPECS
        ]
        self._model = None
        self._prompts = {}
        self._language_ids: set[str] = set()
        self._lock = threading.RLock()
        self._ready = False
        self._asr_loaded = False
        self._custom_ids: set[str] = set()
        self._samples = {}
        self.capabilities = {}
        self.voice_design_attributes = {}

    def load(self) -> None:
        """Load the pinned checkpoint and create missing profiles once."""
        with self._lock:
            if self._ready:
                return
            import numpy as np
            import soundfile as sf
            import torch
            from huggingface_hub import snapshot_download
            from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt
            from omnivoice.models.omnivoice import _combine_text, _resolve_instruct, _tokenize_with_nonverbal_tags
            from omnivoice.utils.text import chunk_text_punctuation
            from omnivoice.utils.voice_design import _INSTRUCT_CATEGORIES
            from omnivoice.utils.lang_map import (
                LANG_IDS, LANG_NAMES, LANG_NAME_TO_ID, lang_display_name,
            )

            if self.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("The configured CUDA device is unavailable.")
            if self.auto_download_models:
                from model_assets import MODELS, ensure_snapshot

                pinned = MODELS["omnivoice"]
                if (self.model_id, self.model_revision) == (pinned["repo_id"], pinned["revision"]):
                    self.model_path = ensure_snapshot("omnivoice", self.model_path)
                self.asr_model_path = ensure_snapshot("whisper", self.asr_model_path)
            if not isinstance(LANG_NAMES, set) or not isinstance(LANG_IDS, set):
                raise RuntimeError("Unsupported upstream language registry format.")
            if set(LANG_NAME_TO_ID) != LANG_NAMES or set(LANG_NAME_TO_ID.values()) != LANG_IDS:
                raise RuntimeError("The upstream language registry is inconsistent.")
            self._np, self._sf, self._torch = np, sf, torch
            self._config_class = OmniVoiceGenerationConfig
            self._resolve_instruct = _resolve_instruct
            self._combine_text = _combine_text
            self._tokenize_text = _tokenize_with_nonverbal_tags
            self._chunk_text = chunk_text_punctuation
            self._prompt_class = VoiceClonePrompt
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self._profile_writer:
                self.voices_dir.mkdir(parents=True, exist_ok=True)
            elif not self.voices_dir.is_dir():
                raise RuntimeError("The profile owner must initialize voices before synthesis workers start.")
            logger.info("Loading OmniVoice checkpoint %s at %s", self.model_id, self.model_revision)
            if self.model_path is not None:
                checkpoint = str(self.model_path.resolve(strict=True))
                required = ("config.json", "model.safetensors", "tokenizer.json", "audio_tokenizer/config.json")
                if any(not (self.model_path / name).is_file() for name in required):
                    raise RuntimeError("The offline model snapshot is incomplete.")
            else:
                from model_assets import offline_mode

                checkpoint = snapshot_download(
                    repo_id=self.model_id,
                    revision=self.model_revision,
                    cache_dir=str(self.cache_dir),
                    local_files_only=not self.auto_download_models or offline_mode(),
                )
            self._model = OmniVoice.from_pretrained(
                checkpoint,
                device_map=self.device,
                dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                load_asr=False,
                attn_implementation="sdpa",
            )
            self._model.eval()
            self.sample_rate = int(self._model.sampling_rate)
            if not 8000 <= self.sample_rate <= 192000:
                raise RuntimeError("The model returned an invalid sample rate.")
            self._language_ids = set(LANG_IDS)
            self.languages = sorted(
                ({"code": LANG_NAME_TO_ID[name], "name": lang_display_name(name)} for name in LANG_NAMES),
                key=lambda item: item["name"].casefold(),
            )
            if "uz" not in self._language_ids:
                raise RuntimeError("The checkpoint does not support the Uzbek profile language.")
            self._prompts = {
                spec["id"]: self._load_profile(spec, VoiceClonePrompt) for spec in VOICE_SPECS
            }
            self.voices = [{key: spec[key] for key in ("id", "gender", "name", "reference_language", "synthetic")} for spec in VOICE_SPECS]
            self._samples = {spec["id"]: self.voices_dir / spec["id"] / "reference.wav" for spec in VOICE_SPECS}
            self._configure_capabilities(_INSTRUCT_CATEGORIES)
            if self._profile_writer:
                self._load_custom_profiles()
            self._configure_acceleration()
            self._ready = True
            logger.info("OmniVoice ready with %d languages, %d voices and %s acceleration",
                        len(self.languages), len(self.voices), self._active_acceleration)

    def _configure_acceleration(self):
        self.capabilities["inference_acceleration"] = "none"
        if self.acceleration == "none":
            return
        # This deployment is validated on RTX 5090 / CUDA 12.8 / FP16. An
        # explicit baseline restart is safer than patching an untested runtime.
        if (not self.device.startswith("cuda") or self._model.dtype != self._torch.float16
                or self._torch.cuda.get_device_capability(self.device) != (12, 0)
                or self._torch.version.cuda != "12.8"
                or os.environ.get("FLASHINFER_CUDA_ARCH_LIST") != "12.0a"
                or os.environ.get("FLASHINFER_USE_CUDA_NORM") != "1"
                or self._model.llm.config.rope_parameters.get("rope_type") != "default"):
            raise RuntimeError("FlashInfer requires the validated CUDA 12.8 / SM120a / FP16 configuration with FLASHINFER_USE_CUDA_NORM=1; use TTS_ACCELERATION=none otherwise.")
        try:
            import flashinfer
            from omnivoice.models.omnivoice_flashinfer import apply_flashinfer

            native_iterative = self._model._generate_iterative if self.acceleration == "tensorrt" else None

            # Global package installation enables CuTe via a .pth file. Keep
            # the CUDA norm kernel used by the user-approved audio sample.
            if not flashinfer.norm._USE_CUDA_NORM:
                raise RuntimeError("FlashInfer was imported with an unvalidated normalization backend.")
            # Patch only after existing profiles have been loaded or created.
            # The optional controller bounds exact-shape native graph storage.
            with self._torch.inference_mode():
                apply_flashinfer(self._model, enable_cuda_graph=False)
                if self.acceleration == "flashinfer" and self.flashinfer_graphs:
                    from flashinfer_backend import install

                    self._flashinfer_backend = install(self._model)
            # Verify actual offline kernels before advertising readiness. _generate
            # forks/restores the RNG for a seed; this never writes voice assets.
            arguments, seed = self._speech_arguments({
                "text": "Assalomu alaykum.", "language": "uz", "voice": "female_1",
                "seed": 1729, "generation": {"num_step": self.num_steps},
            })
            self._generate(arguments, seed)
        except Exception as exc:
            # A failed patch may leave modified forwards. Never silently continue
            # with that model; recovery requires a fresh baseline initialization.
            raise RuntimeError("FlashInfer initialization failed; restart with TTS_ACCELERATION=none to use the baseline.") from exc
        self._active_acceleration = "flashinfer"
        self.capabilities["inference_acceleration"] = "flashinfer"
        if self._flashinfer_backend is not None:
            self._active_acceleration = "flashinfer+cuda-graphs"
            self.capabilities["inference_acceleration"] = self._active_acceleration
        if self.acceleration == "tensorrt":
            from trt_backend import install

            self._trt_backend = install(
                self._model, self.tensorrt_plan, self.tensorrt_manifest,
                native_iterative=native_iterative,
                model_id=self.model_id, model_revision=self.model_revision,
                model_path=self.model_path,
            )
            # Execute the installed path before advertising readiness. An
            # initialization failure requires a fresh model, as with FlashInfer.
            trt_arguments, trt_seed = self._speech_arguments({
                "text": "Assalomu alaykum.", "language": "uz", "voice": "female_1",
                "seed": 1729, "generation": {"num_step": 32},
            })
            self._generate(trt_arguments, trt_seed)
            if not self._trt_backend.describe()["segments"].get("tensorrt", 0):
                raise RuntimeError("TensorRT warmup did not execute a supported TensorRT segment.")
            self._active_acceleration = "tensorrt+flashinfer"
            self.capabilities["inference_acceleration"] = self._active_acceleration

    def _load_profile(self, spec: dict, prompt_class):
        profile_dir = self.voices_dir / spec["id"]
        expected = {
            **spec,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "seed": _SEEDS[spec["id"]],
            "transcript": _REFERENCE_TEXT,
            "profile_version": _PROFILE_VERSION,
            "sample_rate": self.sample_rate,
            "anchor_duration": 8.0,
            "num_steps": self.num_steps,
        }
        if profile_dir.exists():
            try:
                metadata = json.loads((profile_dir / "manifest.json").read_text(encoding="utf-8"))
                digest_keys = {"prompt_sha256", "reference_sha256"}
                if not isinstance(metadata, dict) or set(metadata) != set(expected) | digest_keys:
                    raise ValueError("Manifest fields do not match this profile version.")
                if any(metadata[key] != value for key, value in expected.items()):
                    raise ValueError("Saved identity metadata differs from the configured profile.")
                for filename, key in (("prompt.pt", "prompt_sha256"), ("reference.wav", "reference_sha256")):
                    if self._digest(profile_dir / filename) != metadata[key]:
                        raise ValueError(f"Saved {filename} checksum does not match.")
                prompt = prompt_class.load(str(profile_dir / "prompt.pt"))
                if prompt.ref_text != _REFERENCE_TEXT:
                    raise ValueError("Saved prompt transcript does not match the anchor.")
                return prompt
            except Exception as exc:
                raise RuntimeError(
                    f"Voice profile {spec['id']} is inconsistent; its identity was preserved. "
                    "Restore its matching files or configure a separate voices directory."
                ) from exc

        if not self._profile_writer:
            raise RuntimeError("Synthesis workers cannot create missing built-in profiles.")
        logger.info("Creating synthetic voice profile %s", spec["id"])
        devices = [self._torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with self._torch.random.fork_rng(devices=devices):
            self._torch.manual_seed(expected["seed"])
            reference = self._audio(self._model.generate(
                text=_REFERENCE_TEXT,
                language=spec["reference_language"],
                instruct=spec["instruct"],
                duration=8.0,
                num_step=self.num_steps,
                postprocess_output=False,
            ))
        if not 3.0 <= len(reference) / self.sample_rate <= 10.0:
            raise RuntimeError(f"Voice profile {spec['id']} has an invalid anchor duration.")
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=(self._torch.from_numpy(reference), self.sample_rate),
            ref_text=_REFERENCE_TEXT,
        )
        stage = Path(tempfile.mkdtemp(prefix=f".{spec['id']}-", dir=self.voices_dir))
        try:
            prompt.save(str(stage / "prompt.pt"))
            self._sf.write(str(stage / "reference.wav"), reference, self.sample_rate, subtype="PCM_16")
            metadata = {
                **expected,
                "prompt_sha256": self._digest(stage / "prompt.pt"),
                "reference_sha256": self._digest(stage / "reference.wav"),
            }
            with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            # Publish all identity assets together; existing profiles are never replaced.
            if profile_dir.exists():
                raise RuntimeError(f"Voice profile {spec['id']} appeared during initialization.")
            os.rename(stage, profile_dir)
        finally:
            if stage.exists():
                for filename in ("prompt.pt", "reference.wav", "manifest.json"):
                    (stage / filename).unlink(missing_ok=True)
                stage.rmdir()
        return prompt

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _audio(self, outputs):
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError("The model returned an invalid audio batch.")
        audio = self._np.asarray(outputs[0], dtype=self._np.float32)
        if audio.ndim != 1 or audio.size < 1:
            raise RuntimeError("Generated audio has an invalid shape.")
        if not self._np.isfinite(audio).all():
            raise RuntimeError("Generated audio contains non-finite samples.")
        peak = float(self._np.max(self._np.abs(audio)))
        rms = float(self._np.sqrt(self._np.mean(audio.astype(self._np.float64) ** 2)))
        if peak < 1e-4 or rms < 1e-5:
            raise RuntimeError("The model generated silent audio.")
        if peak > 1.0:
            audio = audio * (0.98 / peak)
        return self._np.ascontiguousarray(audio)

    def synthesize(self, text: str, language: str, voice: str) -> bytes:
        return self.execute("speech", {"text": text, "language": language, "voice": voice})

    def profile_snapshot_descriptor(self, voice_id: str) -> dict:
        """Describe a validated immutable prompt for the accepting coordinator."""
        with self._lock:
            voice = next((item for item in self.voices if item["id"] == voice_id), None)
            if not self._ready or voice is None:
                raise VoiceNotFoundError("Voice was not found.")
            directory = self.voices_dir / ("custom" if voice_id in self._custom_ids else "") / voice_id
            prompt_path = directory / "prompt.pt"
            if self._is_link(directory) or self._is_link(prompt_path):
                raise RuntimeError("Voice profile storage is invalid.")
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            digest = manifest["prompt_sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or self._digest(prompt_path) != digest:
                raise RuntimeError("Voice prompt checksum differs.")
            return {"path": str(prompt_path), "sha256": digest, "gender": voice["gender"], "name": voice["name"]}

    def execute_pinned_speech(self, payload: dict, snapshot_path: str | Path | None, snapshot: dict | None) -> bytes:
        """Run an accepted job using its prompt, even after catalog deletion."""
        with self._lock:
            if payload.get("mode", "preset") != "preset":
                return self.execute("speech", payload)
            voice_id = payload.get("voice")
            if (not isinstance(snapshot, dict) or not isinstance(voice_id, str)
                    or snapshot.get("voice_id") != voice_id or snapshot.get("gender") not in ("male", "female")):
                raise ValueError("The queued voice snapshot is invalid.")
            digest = snapshot.get("sha256", "")
            name = self._text(snapshot.get("name"), 80)
            path = Path(snapshot_path) if snapshot_path is not None else None
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or path is None or self._is_link(path):
                raise ValueError("The queued voice snapshot is invalid.")
            prompt = self._pinned_prompts.get(digest)
            if prompt is None:
                if self._digest(path) != digest:
                    raise ValueError("The queued voice snapshot checksum differs.")
                prompt = self._prompt_class.load(str(path))
                self._pinned_prompts[digest] = prompt
                if len(self._pinned_prompts) > 64:
                    self._pinned_prompts.popitem(last=False)
            else:
                self._pinned_prompts.move_to_end(digest)
            old_voices = self.voices
            old_prompt = self._prompts.get(voice_id)
            self._prompts[voice_id] = prompt
            self.voices = [item for item in self.voices if item["id"] != voice_id] + [
                {"id": voice_id, "gender": snapshot["gender"], "name": name},
            ]
            try:
                return self.execute("speech", payload)
            finally:
                self.voices = old_voices
                if old_prompt is None:
                    self._prompts.pop(voice_id, None)
                else:
                    self._prompts[voice_id] = old_prompt

    def execute(self, operation: str, payload: dict) -> bytes | dict:
        """Run an inference operation; paths and saved tensors remain server-only."""
        with self._lock:
            if not self._ready:
                raise RuntimeError("OmniVoice is not loaded.")
            if not isinstance(payload, dict):
                raise ValueError("Operation settings must be an object.")
            if not self._profile_writer and operation in {"clone_voice", "design_voice", "delete_voice"}:
                raise RuntimeError("Synthesis workers cannot modify saved voice profiles.")
            if operation == "speech":
                arguments, seed = self._speech_arguments(payload)
                return self._wav(self._generate(arguments, seed))
            if operation == "clone_speech":
                options = {key: payload[key] for key in ("text", "language", "speed", "duration", "normalize_text", "generation", "seed") if key in payload}
                arguments, seed = self._speech_arguments({**options, "mode": "auto"})
                instruct = self._instruction(payload.get("instruct"))
                with self._uploaded_audio(payload.get("audio")) as (audio, rate):
                    audio, transcript, _ = self._prepare_reference(audio, rate, payload.get("ref_text"))
                    arguments["voice_clone_prompt"] = self._make_prompt(audio, rate, transcript, arguments["generation_config"].preprocess_prompt)
                if instruct:
                    arguments["instruct"] = instruct
                return self._wav(self._generate(arguments, seed))
            if operation == "clone_voice":
                return self._clone_voice(payload)
            if operation == "design_voice":
                return self._design_voice(payload)
            if operation == "delete_voice":
                return self._delete_voice(payload.get("voice"))
            if operation == "transcribe":
                with self._uploaded_audio(payload.get("audio")) as (audio, rate):
                    return {"text": self._transcribe_samples(audio, rate)}
            if operation == "voice_sample":
                voice = payload.get("voice")
                if not isinstance(voice, str) or voice not in self._samples:
                    raise VoiceNotFoundError("Voice was not found.")
                try:
                    return self._samples[voice].read_bytes()
                except FileNotFoundError as exc:
                    raise VoiceNotFoundError("Voice sample was not found.") from exc
            if operation == "batch":
                items = payload.get("items")
                if not isinstance(items, list) or not items or any(not isinstance(item, dict) for item in items):
                    raise ValueError("A batch requires nonempty speech items.")
                filenames = [speech_filename(item.get("key"), f"{index:03d}.wav") for index, item in enumerate(items, 1)]
                if len({name.casefold() for name in filenames}) != len(filenames):
                    raise ValueError("Batch audio filenames must be unique, ignoring letter case.")
                prepared = [self._speech_arguments(item) for item in items]
                output = io.BytesIO()
                manifest = []
                with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
                    for (arguments, seed), item, filename in zip(prepared, items, filenames):
                        archive.writestr(zipfile.ZipInfo(filename), self._wav(self._generate(arguments, seed)))
                        keys = ("key", "text", "language", "voice", "mode", "gender", "instruct", "speed", "duration", "normalize_text", "generation", "seed")
                        manifest.append({"file": filename, "options": {key: item[key] for key in keys if key in item}})
                    archive.writestr(zipfile.ZipInfo("manifest.json"), json.dumps({"items": manifest}, ensure_ascii=False, indent=2).encode("utf-8"))
                return output.getvalue()
            raise ValueError("Unknown inference operation.")

    def _configure_capabilities(self, categories):
        names = ("gender", "age", "pitch", "style", "english_accent", "chinese_dialect")
        self.voice_design_attributes = {name: list(category) if isinstance(category, dict) else sorted(category) for name, category in zip(names, categories)}
        self.voice_design_attributes["translations"] = {en: zh for category in categories if isinstance(category, dict) for en, zh in category.items()}
        normalization = {}
        if importlib.util.find_spec("tn") is not None and importlib.util.find_spec("pynini") is not None:
            normalization.update({code: "numbers_dates_currency" for code in ("en", "zh") if code in self._language_ids})
        try:
            from num2words import CONVERTER_CLASSES
            normalization.update({code: "integers" for code in self._language_ids if code not in ("en", "zh") and code in CONVERTER_CLASSES})
        except ImportError:
            pass
        self.capabilities = {
            "asr_available": self._asr_available(), "asr_loaded": self._asr_loaded,
            "normalization_languages": sorted(normalization), "normalization_modes": normalization,
            "generation_fields": [field.name for field in fields(self._config_class)],
            "max_custom_voices": None, "custom_quota_bytes": None,
            "max_generated_seconds": None, "max_segment_seconds": 30,
            "reference_preparation": "Long recordings use an automatically selected voiced excerpt and its matching automatic transcript.",
            "max_sequence_tokens": 4096, "max_attention_mask_bytes": 32 * 1024 * 1024,
            "reference_audio_formats": list(REFERENCE_AUDIO_FORMATS),
            "voice_design_attributes": self.voice_design_attributes,
        }

    def _asr_available(self):
        path = self.asr_model_path
        return bool(path and path.is_dir() and all((path / name).is_file() for name in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json")))

    @staticmethod
    def _text(value, limit=0):
        if not isinstance(value, str) or not value.strip() or (limit and len(value) > limit):
            raise ValueError(f"Text must contain between 1 and {limit} characters." if limit else "Text must not be blank.")
        return value.strip()

    @classmethod
    def _speech_text(cls, value):
        text = cls._text(value)
        # The pinned model deletes CR/LF when combining target/reference text.
        # Preserve their word boundaries before duration estimation and tokenization.
        # Leave ordinary single-line scripts unchanged, including spelling/tags.
        if "\r" in text or "\n" in text:
            text = re.sub(r"[ \t\r\n]+", " ", text)
        return text

    def _language(self, value):
        if value is not None and (not isinstance(value, str) or value not in self._language_ids):
            raise ValueError("Unsupported language code.")
        return value

    @staticmethod
    def _number(value, name, minimum, maximum, integer=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= maximum or not math.isfinite(value) or (integer and not isinstance(value, int)):
            raise ValueError(f"Invalid {name}.")
        return value

    def _instruction(self, value, gender=None):
        if value is not None and (not isinstance(value, str) or len(value) > 512):
            raise ValueError("Invalid voice instruction.")
        if gender is not None and gender not in ("male", "female"):
            raise ValueError("Invalid gender.")
        normalized = self._resolve_instruct(value)
        tags = re.split(r"\s*[,，]\s*", normalized or "")
        actual = "male" if any(tag in ("male", "男") for tag in tags) else "female" if any(tag in ("female", "女") for tag in tags) else None
        if gender and actual and actual != gender:
            raise ValueError("Voice instruction conflicts with gender.")
        if gender and not actual:
            normalized = self._resolve_instruct(f"{gender}, {normalized}" if normalized else gender)
        return normalized

    def _generation(self, settings):
        if not isinstance(settings, dict) or set(settings) - set(self.capabilities["generation_fields"]):
            raise ValueError("Unknown generation settings.")
        ranges = {
            "num_step": (1, 64), "guidance_scale": (0, 10), "t_shift": (0, 1),
            "layer_penalty_factor": (0, 10), "position_temperature": (0, 10), "class_temperature": (0, 10),
            "audio_chunk_duration": (3, 30), "audio_chunk_threshold": (3, 60), "pad_duration": (0, 1), "fade_duration": (0, 1),
        }
        values = {"num_step": self.num_steps, **settings}
        for key, value in values.items():
            if key in ("denoise", "preprocess_prompt", "postprocess_output"):
                if not isinstance(value, bool):
                    raise ValueError(f"Invalid {key}.")
            else:
                self._number(value, key, *ranges[key], integer=key == "num_step")
        config = self._config_class(**values)
        if config.t_shift <= 0:
            raise ValueError("t_shift must be greater than zero.")
        if config.audio_chunk_threshold < config.audio_chunk_duration:
            raise ValueError("Chunk threshold must be at least the chunk duration.")
        return config

    def _speech_arguments(self, payload):
        text = self._speech_text(payload.get("text"))
        language = self._language(payload.get("language"))
        mode = payload.get("mode", "preset")
        if mode not in ("preset", "design", "auto"):
            raise ValueError("Unknown speech mode.")
        arguments = {"text": text, "language": language, "generation_config": self._generation(payload.get("generation", {}))}
        gender = payload.get("gender")
        if mode == "preset":
            voice = payload.get("voice") or "female_1"
            if not isinstance(voice, str) or voice not in self._prompts:
                raise ValueError("Unknown voice profile.")
            actual_gender = next(item["gender"] for item in self.voices if item["id"] == voice)
            if gender is not None and gender != actual_gender:
                raise ValueError("Voice profile conflicts with gender.")
            prompt = self._prompts[voice]
            ref_text = getattr(prompt, "ref_text", None)
            if isinstance(ref_text, str) and ("\r" in ref_text or "\n" in ref_text):
                # Repair legacy multiline scripts only in this request's view.
                # Saved prompts and immutable queued snapshots retain their bytes.
                prompt = copy.copy(prompt)
                prompt.ref_text = self._speech_text(ref_text)
            arguments["voice_clone_prompt"] = prompt
            instruct = self._instruction(payload.get("instruct"))
            if instruct:
                self._instruction(instruct, actual_gender)
                arguments["instruct"] = instruct
        elif mode == "design":
            if payload.get("voice") is not None:
                raise ValueError("Design mode does not accept a preset voice.")
            instruct = self._instruction(payload.get("instruct"), gender)
            if not instruct:
                raise ValueError("Design mode requires a voice instruction or gender.")
            arguments["instruct"] = instruct
        elif any(payload.get(key) is not None for key in ("voice", "gender", "instruct")):
            raise ValueError("Auto mode selects its own voice.")
        for key, limits in (("speed", (0.5, 2)), ("duration", (1, 30))):
            if payload.get(key) is not None:
                arguments[key] = self._number(payload[key], key, *limits)
        if "duration" in arguments:
            # Upstream chunking re-estimates duration; keep fixed-duration requests on its direct path.
            arguments["generation_config"].audio_chunk_threshold = max(arguments["generation_config"].audio_chunk_threshold, arguments["duration"] + 1)
        normalize = payload.get("normalize_text", False)
        if not isinstance(normalize, bool):
            raise ValueError("normalize_text must be a boolean.")
        if normalize and language not in self.capabilities["normalization_languages"]:
            raise ValueError("Text normalization is unavailable for this language; specify a supported language.")
        arguments["normalize_text"] = normalize
        seed = payload.get("seed")
        if seed is not None:
            self._number(seed, "seed", 0, 2**32 - 1, integer=True)
        return arguments, seed

    def _generate(self, arguments, seed=None):
        try:
            self._preflight(arguments)
        except _SegmentBudgetError:
            return self._generate_segmented(arguments, seed)
        devices = [self._torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        with self._torch.random.fork_rng(devices=devices) if seed is not None else nullcontext():
            if seed is not None:
                self._torch.manual_seed(seed)
            try:
                outputs = self._model.generate(**arguments)
            except Exception as exc:
                if ((self._trt_backend is not None and self._trt_backend.failed)
                        or (self._flashinfer_backend is not None and self._flashinfer_backend.failed)):
                    raise WorkerRestartRequired("OmniVoice inference failed.") from exc
                raise RuntimeError("OmniVoice inference failed.") from exc
            return self._audio(outputs)

    def _generate_segmented(self, arguments, seed):
        """Retry oversized phrases in bounded working sets; keep the old path exact."""
        task = self._model._preprocess_all(
            **{key: arguments[key] for key in ("text", "language", "voice_clone_prompt", "instruct", "speed", "duration", "normalize_text") if key in arguments},
            preprocess_prompt=arguments["generation_config"].preprocess_prompt,
        )
        text = task.texts[0]
        base = {**arguments, "text": text, "normalize_text": False}
        config = self._config_class(**asdict(arguments["generation_config"]))
        config.audio_chunk_threshold = min(30, config.audio_chunk_threshold)
        base["generation_config"] = config
        total_duration = arguments.get("duration")
        devices = [self._torch.device(self.device).index or 0] if self.device.startswith("cuda") else []
        pieces = []

        def split(value):
            middle = len(value) // 2
            candidates = [match.end() for match in re.finditer(r"[.!?。！？;；,，\s]+", value)
                          if len(value) // 4 <= match.end() <= 3 * len(value) // 4]
            cut = min(candidates, key=lambda index: abs(index - middle)) if candidates else middle
            return value[:cut], value[cut:]

        with self._torch.random.fork_rng(devices=devices) if seed is not None else nullcontext():
            if seed is not None:
                self._torch.manual_seed(seed)
            pending = [text]
            while pending:
                chunk = pending.pop()
                if not chunk.strip():
                    continue
                current = {**base, "text": chunk}
                if total_duration is not None:
                    current["duration"] = total_duration * len(chunk) / len(text)
                try:
                    self._preflight(current, single_segment=True)
                except _SegmentBudgetError:
                    if len(chunk) <= 1:
                        raise ValueError("The voice reference or instruction cannot fit the model working memory.") from None
                    left, right = split(chunk)
                    pending.extend([right, left])
                    continue
                try:
                    audio = self._audio(self._model.generate(**current))
                except Exception as exc:
                    if ((self._trt_backend is not None and self._trt_backend.failed)
                            or (self._flashinfer_backend is not None and self._flashinfer_backend.failed)):
                        raise WorkerRestartRequired("OmniVoice inference failed.") from exc
                    raise RuntimeError("OmniVoice inference failed.") from exc
                pieces.append(audio)
                if "voice_clone_prompt" not in base and pending:
                    # Auto/design continuation keeps the first generated voice.
                    base["voice_clone_prompt"] = self._make_prompt(audio, self.sample_rate, chunk, False)
        if not pieces:
            raise ValueError("The text did not produce speech.")
        return self._np.concatenate(pieces)

    def _preflight(self, arguments, *, single_segment=False):
        """Bound the exact upstream tasks before allocating inference tensors on the GPU."""
        config = arguments["generation_config"]
        # Cached prompts supply tokens directly: this upstream path only resolves
        # text/language/instructions and estimates lengths; it never encodes audio.
        task = self._model._preprocess_all(
            **{key: arguments[key] for key in ("text", "language", "voice_clone_prompt", "instruct", "speed", "duration", "normalize_text") if key in arguments},
            preprocess_prompt=config.preprocess_prompt,
        )
        if task.batch_size != 1:
            raise RuntimeError("The inference preflight requires one task at a time.")
        frame_rate = float(self._model.audio_tokenizer.config.frame_rate)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise RuntimeError("The audio tokenizer has an invalid frame rate.")
        text, target_frames = task.texts[0], task.target_lens[0]
        if not text or target_frames < 1:
            raise ValueError("The text did not produce a valid speech duration.")
        if single_segment and target_frames > int(min(15, config.audio_chunk_threshold) * frame_rate):
            raise _SegmentBudgetError("Continuation needs a single bounded speech segment.")
        reference = task.ref_audio_tokens[0]
        reference_frames = int(reference.size(-1)) if reference is not None else 0
        if reference is not None and reference_frames < 1:
            raise ValueError("The voice prompt contains no audio tokens.")

        def segment_budget(chunk, frames, ref_text, ref_frames):
            if not 1 <= frames <= int(30 * frame_rate):
                raise _SegmentBudgetError("A speech segment requires a smaller model working set.")
            style = "<|denoise|>" if config.denoise and ref_frames else ""
            style += f"<|lang_start|>{task.langs[0] or 'None'}<|lang_end|><|instruct_start|>{task.instructs[0] or 'None'}<|instruct_end|>"
            combined = self._combine_text(text=chunk, ref_text=ref_text)
            # Both helpers tokenize on CPU. Do not call _prepare_inference_inputs:
            # that method allocates the audio tokens and quadratic attention mask.
            with self._torch.device("cpu"):
                style_length = self._model.text_tokenizer(style, return_tensors="pt").input_ids.size(-1)
                text_length = self._tokenize_text(f"<|text_start|>{combined}<|text_end|>", self._model.text_tokenizer).size(-1)
            sequence_length = int(style_length + text_length + ref_frames + frames)
            if sequence_length > 4096 or 2 * sequence_length**2 > 32 * 1024 * 1024:
                raise _SegmentBudgetError("Speech requires a smaller inference sequence/attention working set.")

        if target_frames <= int(config.audio_chunk_threshold * frame_rate):
            segments = [(text, target_frames, task.ref_texts[0], reference_frames)]
        else:
            # Mirror _generate_chunked, including its punctuation-only splitter
            # and its reuse of generated chunk zero for auto/design voices.
            chunk_length = int(config.audio_chunk_duration * frame_rate / (target_frames / len(text)))
            chunks = self._chunk_text(text=text, chunk_len=chunk_length, min_chunk_len=3)
            if not chunks:
                raise ValueError("The text did not produce any speech segments.")
            speed = task.speed[0] if task.speed else 1.0
            segments = []
            for index, chunk in enumerate(chunks):
                ref_text = task.ref_texts[0] if reference is not None else (chunks[0] if index else None)
                ref_frames = reference_frames if reference is not None else (segments[0][1] if index else 0)
                frames = self._model._estimate_target_tokens(chunk, ref_text, ref_frames or None, speed=speed)
                segments.append((chunk, frames, ref_text, ref_frames))
        for segment in segments:
            segment_budget(*segment)
        # Generate exactly the normalized target that was checked, without a
        # second normalization pass changing its size after the resource check.
        arguments["text"], arguments["normalize_text"] = text, False

    def _wav(self, audio, rate=None):
        output = io.BytesIO()
        self._sf.write(output, audio, rate or self.sample_rate, format="WAV", subtype="PCM_16")
        return output.getvalue()

    def _decode_upload(self, encoded):
        if isinstance(encoded, Path):
            if self._is_link(encoded) or not encoded.is_file() or encoded.stat().st_size < 1:
                raise ValueError("Reference audio must contain a regular nonempty uploaded file.")
            source = encoded
        elif isinstance(encoded, bytes) and encoded:
            source = io.BytesIO(encoded)
        else:
            raise ValueError("Reference audio must contain nonempty audio bytes.")
        try:
            try:
                info = self._sf.info(source)
            except Exception:
                info = None
            if info is not None and info.format in ("WAV", "WAVEX", "RF64", "FLAC"):
                if info.samplerate < 1 or info.channels < 1 or info.frames < 1:
                    raise ValueError("Reference audio must contain valid nonempty audio samples.")
                try:
                    audio, rate = self._sf.read(encoded if isinstance(encoded, Path) else io.BytesIO(encoded), dtype="float32", always_2d=True, frames=info.frames + 1)
                except Exception as exc:
                    raise ValueError("Reference audio could not be decoded.") from exc
                if rate != info.samplerate or audio.shape != (info.frames, info.channels):
                    raise ValueError("Reference audio contains invalid samples.")
            else:
                audio, rate = self._decode_media_upload(encoded)
            if not self._np.isfinite(audio).all():
                raise ValueError("Reference audio contains invalid samples.")
            mono = audio.mean(axis=1, dtype=self._np.float64)
            peak = float(self._np.max(self._np.abs(mono)))
            if peak < 1e-4 or float(self._np.sqrt(self._np.mean(mono ** 2))) < 1e-5:
                raise ValueError("Reference audio is silent.")
            if peak > 1:
                mono *= 0.98 / peak
            return self._np.ascontiguousarray(mono, dtype=self._np.float32), rate
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            raise ValueError("Reference audio could not be decoded.") from exc

    @contextmanager
    def _uploaded_audio(self, encoded):
        """Disk-backed decode for staged HTTP files, with bounded scratch arrays."""
        if not isinstance(encoded, Path):
            yield self._decode_upload(encoded)
            return
        if self._is_link(encoded) or not encoded.is_file() or encoded.stat().st_size < 1:
            raise ValueError("Reference audio must contain a regular nonempty uploaded file.")
        with tempfile.TemporaryDirectory(prefix="text-speak-pcm-") as directory:
            target = Path(directory) / "decoded.pcm"
            try:
                info = self._sf.info(encoded)
            except Exception:
                info = None
            if info is not None and info.format in ("WAV", "WAVEX", "RF64", "FLAC"):
                if info.samplerate < 1 or info.channels < 1 or info.frames < 1:
                    raise ValueError("Reference audio contains no usable samples.")
                frames = 0
                try:
                    with self._sf.SoundFile(encoded) as source, target.open("xb") as output:
                        while True:
                            block = source.read(max(1, 65536 // info.channels), dtype="float32", always_2d=True)
                            if not len(block):
                                break
                            output.write(block.mean(axis=1, dtype=self._np.float64).astype("<f4").tobytes())
                            frames += len(block)
                except Exception as exc:
                    raise ValueError("Reference audio could not be decoded.") from exc
                if frames != info.frames:
                    raise ValueError("Reference audio contains invalid samples.")
                rate = info.samplerate
            else:
                _, rate = self._decode_media_upload(encoded, spool_dir=directory)
            audio = self._np.memmap(target, mode="r+", dtype="<f4")
            try:
                peak, rms = self._audio_statistics(audio)
                if peak < 1e-4 or rms < 1e-5:
                    raise ValueError("Reference audio is silent.")
                if peak > 1:
                    for offset in range(0, len(audio), 65536):
                        audio[offset:offset + 65536] *= .98 / peak
                yield audio, rate
            finally:
                audio._mmap.close()

    def _audio_statistics(self, audio):
        peak, square_sum = 0.0, 0.0
        for offset in range(0, len(audio), 65536):
            block = self._np.asarray(audio[offset:offset + 65536], dtype=self._np.float64)
            if not self._np.isfinite(block).all():
                raise ValueError("Reference audio contains invalid samples.")
            peak = max(peak, float(self._np.max(self._np.abs(block))))
            square_sum += float(self._np.dot(block, block))
        return peak, math.sqrt(square_sum / max(1, len(audio)))

    def _decode_media_upload(self, encoded, *, spool_dir=None):
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            executable = get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("The audio format decoder is unavailable.") from exc
        rate = 24000
        with tempfile.TemporaryDirectory(prefix="text-speak-audio-") if spool_dir is None else nullcontext(spool_dir) as directory:
            source, target = Path(directory) / "input.media", Path(directory) / "decoded.pcm"
            if isinstance(encoded, Path):
                source = encoded
            else:
                source.write_bytes(encoded)
            command = [
                executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-xerror",
                "-max_alloc", "67108864", "-filter_threads", "1",
                "-protocol_whitelist", "file", "-format_whitelist", _AUDIO_DEMUXERS,
                "-probesize", "8388608", "-analyzeduration", "5000000", "-max_streams", "16",
                "-threads", "1", "-err_detect", "explode", "-i", str(source),
                "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
                "-af", "aresample=24000,asetpts=N/SR/TB", "-ac", "1", "-ar", str(rate),
                "-threads", "1",
                "-c:a", "pcm_f32le", "-f", "f32le", str(target),
            ]
            try:
                result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("Reference audio took too long to decode.") from exc
            except OSError as exc:
                raise RuntimeError("The audio format decoder could not start.") from exc
            if result.returncode != 0 or not target.is_file():
                raise ValueError("Reference audio could not be decoded; upload a supported self-contained audio file.")
            size = target.stat().st_size
            if size % 4 or size < 4:
                raise ValueError("Reference audio must contain valid decoded speech samples.")
            if spool_dir is not None:
                return None, rate
            return self._np.frombuffer(target.read_bytes(), dtype="<f4").reshape(-1, 1), rate

    def _transcribe_samples(self, audio, rate, *, reference_language=None):
        if not self._asr_available():
            raise RuntimeError("A complete local Whisper snapshot is required for transcription.")
        if not self._asr_loaded:
            try:
                self._model.load_asr_model(model_name=str(self.asr_model_path), device=self.device)
            except Exception as exc:
                raise RuntimeError("The local transcription model could not be loaded.") from exc
            self._asr_loaded = True
            self.capabilities["asr_loaded"] = True
        pipeline = getattr(self._model, "_asr_pipe", None)
        asr_config = getattr(getattr(pipeline, "model", None), "generation_config", None)
        language_ids = getattr(asr_config, "lang_to_id", {})
        use_language = (isinstance(reference_language, str) and callable(pipeline)
                        and isinstance(language_ids, dict) and f"<|{reference_language}|>" in language_ids)
        texts = []
        # Whisper handles each window independently, so full recordings never
        # depend on its single-window duration or token limit.
        for start in range(0, len(audio), max(1, int(rate * 25))):
            chunk = audio[start:start + int(rate * 25)]
            if not self._np.any(self._np.abs(chunk) > 1e-4):
                continue
            try:
                if use_language:
                    # A clone's declared recording language is a per-call hint.
                    # Do not mutate the shared Whisper generation configuration:
                    # subsequent autodetection and cross-language clones must work.
                    with self._torch.inference_mode():
                        text = pipeline({"array": self._np.squeeze(chunk), "sampling_rate": rate},
                                        generate_kwargs={"language": reference_language, "task": "transcribe"})["text"].strip()
                else:
                    text = self._model.transcribe((chunk, rate))
            except Exception as exc:
                raise RuntimeError("Reference transcription failed.") from exc
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        return self._speech_text(" ".join(texts))

    def _transcript(self, value, audio, rate, *, reference_language=None):
        return (self._transcribe_samples(audio, rate, reference_language=reference_language)
                if value is None else self._speech_text(value))

    def _prepare_reference(self, audio, rate, transcript, *, reference_language=None):
        if transcript is not None:
            transcript = self._speech_text(transcript)
        source_seconds = len(audio) / rate
        info = {"source_seconds": source_seconds, "selected_start_seconds": 0.0,
                "selected_seconds": source_seconds, "automatic_transcript": transcript is None}
        if source_seconds > 20:
            # Find actual speech rather than taking a silent beginning. A crop
            # receives its own ASR transcript, never the original full text.
            step = max(1, int(rate * .2))
            threshold = max(1e-4, self._audio_statistics(audio)[0] * .01)
            start = 0
            for offset in range(0, len(audio), step):
                if float(self._np.sqrt(self._np.mean(audio[offset:offset + step] ** 2))) > threshold:
                    start = max(0, offset - int(rate * .1))
                    break
            stop = min(len(audio), start + int(rate * 15))
            # Prefer a phrase boundary near the end of the selected window.
            for offset in range(stop - step, start + int(rate * 10), -step):
                if float(self._np.sqrt(self._np.mean(audio[offset:offset + step] ** 2))) < threshold:
                    stop = offset + step
                    break
            audio = self._np.ascontiguousarray(audio[start:stop])
            transcript = self._transcribe_samples(audio, rate, reference_language=reference_language)
            info.update(selected_start_seconds=start / rate, selected_seconds=len(audio) / rate,
                        automatic_transcript=True)
        else:
            if transcript is not None:
                with self._torch.device("cpu"):
                    tokens = self._tokenize_text(transcript, self._model.text_tokenizer).size(-1)
                if tokens > 2048:
                    # The uploaded recording still succeeds. Re-transcribe its
                    # actual words instead of storing an unusable huge prompt.
                    transcript = None
                    info.update(automatic_transcript=True, transcript_replaced=True)
            transcript = self._transcript(transcript, audio, rate, reference_language=reference_language)
        return audio, transcript, info

    def _make_prompt(self, audio, rate, text, preprocess=True):
        text = self._speech_text(text)
        tokenizer = self._model.audio_tokenizer.config
        hop = getattr(tokenizer, "hop_length", round(self.sample_rate / tokenizer.frame_rate))
        minimum = max(1, math.ceil(hop * rate / self.sample_rate))
        if len(audio) < minimum:
            audio = self._np.pad(audio, (0, minimum - len(audio)))
        return self._model.create_voice_clone_prompt(ref_audio=(self._torch.from_numpy(audio), rate), ref_text=text, preprocess_prompt=preprocess)

    @staticmethod
    def _is_link(path):
        try:
            attributes = path.lstat()
        except FileNotFoundError:
            return False
        # Windows junctions and other reparse points also require rejection.
        return path.is_symlink() or bool(getattr(attributes, "st_file_attributes", 0) & 0x400)

    def _checked_custom_root(self):
        root = self.voices_dir / "custom"
        try:
            if self._is_link(self.voices_dir) or self._is_link(root):
                raise VoiceDeletionError("Custom voice storage is unsafe or inconsistent.")
            if root.resolve().parent != self.voices_dir.resolve(strict=True):
                raise VoiceDeletionError("Custom voice storage is unsafe or inconsistent.")
            if root.exists() and not root.is_dir():
                raise VoiceDeletionError("Custom voice storage is unsafe or inconsistent.")
        except OSError as exc:
            raise VoiceDeletionError("Custom voice storage could not be verified.") from exc
        return root

    def _checked_custom_directory(self, voice_id, *, deleted=False):
        if not isinstance(voice_id, str) or not re.fullmatch(_CUSTOM_VOICE_ID, voice_id):
            raise VoiceDeletionError("Custom voice storage has an invalid identity.")
        root = self._checked_custom_root()
        directory = root / (f".deleted-{voice_id}" if deleted else voice_id)
        try:
            if self._is_link(directory) or not directory.is_dir() or directory.resolve(strict=True).parent != root.resolve(strict=True):
                raise VoiceDeletionError("Custom voice storage is unsafe or inconsistent.")
            contents = list(directory.iterdir())
            names = {path.name for path in contents}
            if names - set(_PROFILE_FILES) or (not deleted and names != set(_PROFILE_FILES)):
                raise VoiceDeletionError("Custom voice storage contains unexpected or missing files.")
            resolved_directory = directory.resolve(strict=True)
            if any(self._is_link(path) or not path.is_file() or path.resolve(strict=True).parent != resolved_directory for path in contents):
                raise VoiceDeletionError("Custom voice storage contains unsafe files.")
        except OSError as exc:
            raise VoiceDeletionError("Custom voice storage could not be verified.") from exc
        return directory

    def _purge_deleted_profile(self, voice_id):
        directory = self._checked_custom_directory(voice_id, deleted=True)
        try:
            for filename in _PROFILE_FILES:
                # Verify the remaining subset before each bounded, nonrecursive removal.
                self._checked_custom_directory(voice_id, deleted=True)
                (directory / filename).unlink(missing_ok=True)
            self._checked_custom_directory(voice_id, deleted=True)
            directory.rmdir()
        except OSError as exc:
            raise VoiceDeletionError("Voice deletion did not finish; retry with the same voice ID.") from exc

    def _cleanup_deleted_profiles(self):
        try:
            root = self._checked_custom_root()
            if not root.exists():
                return
            pending = [path.name[len(".deleted-"):] for path in root.iterdir() if re.fullmatch(rf"\.deleted-{_CUSTOM_VOICE_ID}", path.name)]
        except VoiceDeletionError:
            logger.warning("Pending voice deletion cleanup could not verify storage")
            return
        except OSError:
            logger.warning("Pending voice deletion cleanup could not read storage")
            return
        for voice_id in pending:
            try:
                self._purge_deleted_profile(voice_id)
            except VoiceDeletionError:
                logger.warning("Pending voice deletion cleanup did not finish")

    def _delete_voice(self, voice_id):
        if isinstance(voice_id, str) and voice_id in {spec["id"] for spec in VOICE_SPECS}:
            raise ProtectedVoiceError("Built-in voices cannot be deleted.")
        if not isinstance(voice_id, str) or not re.fullmatch(_CUSTOM_VOICE_ID, voice_id):
            raise VoiceNotFoundError("Voice was not found.")
        root = self._checked_custom_root()
        active, deleted = root / voice_id, root / f".deleted-{voice_id}"
        try:
            if voice_id in self._custom_ids:
                self._checked_custom_directory(voice_id)
                if deleted.exists() or self._is_link(deleted):
                    raise VoiceDeletionError("Custom voice storage contains conflicting deletion state.")
                os.rename(active, deleted)
                # The atomic rename commits logical deletion even if later cleanup fails.
                self._custom_ids.discard(voice_id)
                self._prompts.pop(voice_id, None)
                self._samples.pop(voice_id, None)
                self.voices = [voice for voice in self.voices if voice["id"] != voice_id]
            elif not deleted.exists() and not self._is_link(deleted):
                raise VoiceNotFoundError("Voice was not found.")
            elif active.exists() or self._is_link(active):
                raise VoiceDeletionError("Custom voice storage contains conflicting deletion state.")
        except OSError as exc:
            raise VoiceDeletionError("Voice deletion could not be committed; retry later.") from exc
        self._purge_deleted_profile(voice_id)
        return {"id": voice_id, "deleted": True}

    def _custom_capacity(self, additional_bytes=0):
        root = self._checked_custom_root()
        if additional_bytes > shutil.disk_usage(root if root.exists() else root.parent).free:
            raise OSError(28, "Insufficient disk space to save the voice profile")

    def _profile_details(self, payload, language_key):
        self._cleanup_deleted_profiles()
        self._custom_capacity()
        name = self._text(payload.get("name"), 80)
        gender = payload.get("gender")
        if gender not in ("male", "female"):
            raise ValueError("A custom voice requires male or female gender metadata.")
        return {"name": name, "gender": gender, "reference_language": self._language(payload.get(language_key))}

    def _clone_voice(self, payload):
        details = self._profile_details(payload, "reference_language")
        generation = payload.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("Generation settings must be an object.")
        generation = dict(generation)
        if "preprocess_prompt" in payload:
            generation["preprocess_prompt"] = payload["preprocess_prompt"]
        config = self._generation(generation)
        with self._uploaded_audio(payload.get("audio")) as (audio, rate):
            audio, text, preparation = self._prepare_reference(
                audio, rate, payload.get("ref_text"), reference_language=details["reference_language"])
            prompt = self._make_prompt(audio, rate, text, config.preprocess_prompt)
            return self._publish_custom({**details, "source": "clone", "synthetic": False}, audio, rate, prompt,
                                        {"generation": asdict(config), "reference_preparation": preparation})

    def _design_voice(self, payload):
        details = self._profile_details(payload, "language")
        if any(payload.get(key) is not None for key in ("speed", "duration")):
            raise ValueError("A saved design voice uses a fixed 8-second reference.")
        text = self._speech_text(payload.get("reference_text", _REFERENCE_TEXT))
        normalize = payload.get("normalize_text", False)
        if not isinstance(normalize, bool):
            raise ValueError("normalize_text must be a boolean.")
        if normalize and details["reference_language"] not in self.capabilities["normalization_languages"]:
            raise ValueError("Text normalization is unavailable for this language; specify a supported language.")
        if normalize:
            from omnivoice.utils.text import normalize_text
            text = self._speech_text(normalize_text(text, details["reference_language"]))
        source_length = len(text)
        # A design creates an eight-second voice anchor, not the final speech.
        # Select a matching amount of script instead of compressing a long
        # user-supplied script into eight seconds and persisting an unusable voice.
        target_frames = round(8 * self._model.audio_tokenizer.config.frame_rate)
        low, high, selected = 1, len(text), 1
        while low <= high:
            middle = (low + high) // 2
            if self._model._estimate_target_tokens(text[:middle], None, None) <= target_frames:
                selected, low = middle, middle + 1
            else:
                high = middle - 1
        if selected < len(text):
            prefix = text[:selected]
            endings = [match.end() for match in re.finditer(r"[.!?。！？;；\s]+", prefix)
                       if match.end() >= len(prefix) // 2]
            text = prefix[:endings[-1]] if endings else prefix
            text = text.strip()
        preparation = {"source_characters": source_length, "selected_characters": len(text),
                       "automatic_selection": len(text) < source_length}
        arguments, seed = self._speech_arguments({
            "text": text, "language": details["reference_language"], "mode": "design", "gender": details["gender"],
            "instruct": payload.get("instruct"), "generation": payload.get("generation", {}), "seed": payload.get("seed"), "duration": 8.0,
            "normalize_text": False,
        })
        audio = self._generate(arguments, seed)
        prompt = self._make_prompt(audio, self.sample_rate, text, arguments["generation_config"].preprocess_prompt)
        controls = {"generation": asdict(arguments["generation_config"]), "seed": seed, "instruct": arguments["instruct"], "duration": 8.0, "normalize_text": normalize,
                    "reference_text_preparation": preparation}
        return self._publish_custom({**details, "source": "design", "synthetic": True}, audio, self.sample_rate, prompt, controls)

    @staticmethod
    def _custom_metadata(manifest):
        keys = ("id", "name", "gender", "reference_language", "synthetic", "source", "ref_text")
        preparation = manifest.get("reference_preparation", manifest.get("controls", {}).get("reference_preparation"))
        script_preparation = manifest.get("controls", {}).get("reference_text_preparation")
        return {**{key: manifest[key] for key in keys},
                **({"reference_preparation": preparation} if preparation is not None else {}),
                **({"reference_text_preparation": script_preparation} if script_preparation is not None else {}),
                "sample_url": f"/v1/voices/{manifest['id']}/sample"}

    def _publish_custom(self, details, audio, rate, prompt, controls):
        root = self.voices_dir / "custom"
        root.mkdir(exist_ok=True)
        voice_id = details["source"] + "_" + uuid.uuid4().hex
        destination = root / voice_id
        stage = Path(tempfile.mkdtemp(prefix=f".{voice_id}-", dir=root))
        try:
            prompt.save(str(stage / "prompt.pt"))
            (stage / "reference.wav").write_bytes(self._wav(audio, rate))
            manifest = {
                **details, "id": voice_id, "ref_text": prompt.ref_text, "profile_version": 1,
                "model_id": self.model_id, "model_revision": self.model_revision, "sample_rate": rate, "controls": controls,
                "prompt_sha256": self._digest(stage / "prompt.pt"), "reference_sha256": self._digest(stage / "reference.wav"),
            }
            with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            self._custom_capacity(sum(path.stat().st_size for path in stage.iterdir()))
            os.rename(stage, destination)
        except OSError as exc:
            raise RuntimeError("The custom voice profile could not be published.") from exc
        finally:
            if stage.exists():
                for filename in ("prompt.pt", "reference.wav", "manifest.json"):
                    (stage / filename).unlink(missing_ok=True)
                stage.rmdir()
        metadata = self._custom_metadata(manifest)
        self._custom_ids.add(voice_id)
        self._prompts[voice_id] = prompt
        self._samples[voice_id] = destination / "reference.wav"
        self.voices.append(metadata)
        return dict(metadata)

    def _load_custom_profiles(self):
        self._cleanup_deleted_profiles()
        root = self.voices_dir / "custom"
        self._custom_ids = set()
        if not root.exists():
            return
        for directory in sorted(root.iterdir()):
            if directory.name.startswith("."):
                continue
            try:
                if not directory.is_dir() or directory.is_symlink() or not re.fullmatch(r"(?:clone|design)_[0-9a-f]{32}", directory.name):
                    raise ValueError("Invalid custom profile directory.")
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                required = {"id", "name", "gender", "reference_language", "synthetic", "source", "ref_text", "profile_version", "model_id", "model_revision", "sample_rate", "controls", "prompt_sha256", "reference_sha256"}
                if (not isinstance(manifest, dict) or not required.issubset(manifest)
                        or set(manifest) - required - {"reference_preparation"}
                        or not isinstance(manifest["controls"], dict)):
                    raise ValueError("Invalid custom profile manifest.")
                if manifest["id"] != directory.name or manifest["model_id"] != self.model_id or manifest["model_revision"] != self.model_revision or manifest["profile_version"] != 1 or manifest["source"] not in ("clone", "design") or manifest["synthetic"] is not (manifest["source"] == "design"):
                    raise ValueError("Custom profile identity metadata differs.")
                self._text(manifest["name"], 80)
                self._text(manifest["ref_text"])
                self._generation(manifest["controls"]["generation"])
                self._language(manifest["reference_language"])
                if manifest["gender"] not in ("male", "female"):
                    raise ValueError("Invalid custom profile gender.")
                for filename, key in (("prompt.pt", "prompt_sha256"), ("reference.wav", "reference_sha256")):
                    if self._digest(directory / filename) != manifest[key]:
                        raise ValueError("Custom profile checksum mismatch.")
                _, rate = self._decode_upload((directory / "reference.wav").read_bytes())
                if rate != manifest["sample_rate"]:
                    raise ValueError("Custom profile sample rate differs.")
                prompt = self._prompt_class.load(str(directory / "prompt.pt"))
                if prompt.ref_text != manifest["ref_text"]:
                    raise ValueError("Custom prompt transcript differs.")
                self._custom_capacity()
                self._custom_ids.add(directory.name)
                self._prompts[directory.name] = prompt
                self._samples[directory.name] = directory / "reference.wav"
                self.voices.append(self._custom_metadata(manifest))
            except Exception as exc:
                raise RuntimeError(f"Custom voice profile {directory.name} is inconsistent; its identity was preserved.") from exc
