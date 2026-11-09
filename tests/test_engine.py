"""Inference-boundary tests with fake ML modules; no GPU, weights or downloads."""

from contextlib import nullcontext
import ast
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import types
import wave
import zipfile

import imageio_ffmpeg
import numpy as np
import pytest

import engine as engine_module
from engine import OmniVoiceEngine, ProtectedVoiceError, VoiceDeletionError, VoiceNotFoundError, VOICE_SPECS


class TokenShape:
    def __init__(self, length):
        self.length = length

    def size(self, dimension):
        return self.length


class CpuTokenizer:
    multiplier = 1

    def __call__(self, text, **options):
        return types.SimpleNamespace(input_ids=TokenShape(len(text) * self.multiplier))


class DeviceContext:
    index = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@dataclass
class GenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0
    pad_duration: float = 0.1
    fade_duration: float = 0.1


def wav_bytes(seconds=4, rate=24000, channels=1, silent=False):
    samples = np.zeros(int(seconds * rate)) if silent else np.sin(np.arange(int(seconds * rate)) * 0.05) * 0.25
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(np.repeat((samples * 32767).astype("<i2"), channels).tobytes())
    return output.getvalue()


@pytest.fixture
def media(tmp_path):
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    counter = 0

    def encode(codec, extension, *, source=None, options=()):
        nonlocal counter
        counter += 1
        input_path = tmp_path / f"source-{counter}.wav"
        target = tmp_path / f"recording-{counter}.{extension}"
        input_path.write_bytes(wav_bytes() if source is None else source)
        result = subprocess.run(
            [executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
             "-i", str(input_path), "-map", "0:a:0", "-c:a", codec, *options, str(target)],
            capture_output=True, timeout=15, check=False,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        return target.read_bytes()

    return types.SimpleNamespace(encode=encode, executable=executable)


class FakeSoundFile(types.ModuleType):
    def __init__(self):
        super().__init__("soundfile")
        self.read_calls = 0
        self.special = {}

    def info(self, stream):
        data = stream.getvalue()
        if data in self.special:
            return self.special[data][0]
        with wave.open(stream) as audio:
            return types.SimpleNamespace(format="WAV", frames=audio.getnframes(), channels=audio.getnchannels(), samplerate=audio.getframerate())

    def read(self, stream, **options):
        self.read_calls += 1
        data = stream.getvalue()
        if data in self.special:
            info, samples = self.special[data]
            return samples, info.samplerate
        with wave.open(stream) as audio:
            samples = np.frombuffer(audio.readframes(options["frames"]), dtype="<i2").astype(np.float32) / 32768
            return samples.reshape(-1, audio.getnchannels()), audio.getframerate()

    def write(self, target, samples, rate, **options):
        assert options["subtype"] == "PCM_16"
        with wave.open(target if hasattr(target, "write") else str(target), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    prompts = []
    models = []
    seeds = []

    class Prompt:
        def __init__(self, text, frames=200):
            self.ref_text = text
            self.ref_audio_tokens = TokenShape(frames)
            self.ref_rms = 0.2

        def save(self, path):
            Path(path).write_text(json.dumps({"text": self.ref_text, "frames": self.ref_audio_tokens.size(-1)}), encoding="utf-8")

        @classmethod
        def load(cls, path):
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(value["text"], value["frames"])

    class Model:
        sampling_rate = 24000

        def __init__(self):
            self.calls = []
            self.dtype = None
            self.llm = types.SimpleNamespace(config=types.SimpleNamespace(
                rope_parameters={"rope_type": "default"}, use_cache=True))
            self.asr_loads = []
            self.result = None
            self.transcript = "Recognized reference speech"
            self.audio_tokenizer = types.SimpleNamespace(config=types.SimpleNamespace(frame_rate=25))
            self.text_tokenizer = CpuTokenizer()
            self.duration_estimator = upstream.RuleDurationEstimator()

        @classmethod
        def from_pretrained(cls, path, **options):
            assert options["load_asr"] is False and options["attn_implementation"] == "sdpa"
            instance = cls()
            instance.dtype = options["dtype"]
            models.append(instance)
            return instance

        def eval(self):
            return self

        def generate(self, **options):
            self.calls.append(options)
            if options["text"] == "fail":
                raise RuntimeError("Synthetic model failure")
            if self.result is not None:
                return self.result
            duration = options.get("duration", 1)
            return [(np.sin(np.arange(int(duration * self.sampling_rate)) * 0.05) * 0.25).astype(np.float32)]

        def create_voice_clone_prompt(self, **options):
            prompts.append(options)
            text = options["ref_text"]
            if options.get("preprocess_prompt", True) and not text.endswith("."):
                text += "."
            samples, rate = options["ref_audio"]
            return Prompt(text, int(len(samples) / rate * 25))

        def load_asr_model(self, **options):
            self.asr_loads.append(options)

        def transcribe(self, audio):
            return self.transcript

    def resolve_instruction(value, use_zh=False):
        if not value:
            return None
        tags = [item.strip().lower() for item in re.split("[,，]", value) if item.strip()]
        allowed = {"male", "female", "男", "女", "young adult", "middle-aged", "low pitch", "moderate pitch", "whisper"}
        if set(tags) - allowed or len(set(tags) & {"male", "female", "男", "女"}) > 1:
            raise ValueError("Upstream instruction validation rejected the request")
        return ", ".join(tags)

    sf = FakeSoundFile()
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "fp16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda device: (12, 0))
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.inference_mode = nullcontext
    torch.device = lambda value: DeviceContext()
    torch.random = types.SimpleNamespace(fork_rng=lambda **options: nullcontext())
    torch.manual_seed, torch.from_numpy = seeds.append, lambda value: value
    omni = types.ModuleType("omnivoice")
    omni.OmniVoice, omni.VoiceClonePrompt, omni.OmniVoiceGenerationConfig = Model, Prompt, GenerationConfig
    core = types.ModuleType("omnivoice.models.omnivoice")
    core._resolve_instruct = resolve_instruction
    languages = types.ModuleType("omnivoice.utils.lang_map")
    languages.LANG_NAME_TO_ID = {"uzbek": "uz", "english": "en", "chinese": "zh", "russian": "ru"}
    languages.LANG_NAMES = set(languages.LANG_NAME_TO_ID)
    languages.LANG_IDS = set(languages.LANG_NAME_TO_ID.values())
    languages.lang_display_name = str.title
    attributes = types.ModuleType("omnivoice.utils.voice_design")
    attributes._INSTRUCT_CATEGORIES = [{"male": "男", "female": "女"}, {"young adult": "青年"}, {"low pitch": "低音调"}, {"whisper": "耳语"}, {"british accent"}, {"四川话"}]
    normalizer = types.ModuleType("omnivoice.utils.text")
    normalizer.normalize_text = lambda text, language: text.replace("123", "one hundred twenty-three")
    # Execute only CPU utilities and three preprocessing methods extracted from
    # the pinned upstream wheel. No model module or ML backend is imported.
    upstream = types.ModuleType("engine_test_upstream")
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    with zipfile.ZipFile(Path(__file__).resolve().parents[1] / "offline/wheels/omnivoice-0.2.1-py3-none-any.whl") as wheel:
        exec(compile(wheel.read("omnivoice/utils/duration.py"), "upstream_duration.py", "exec"), upstream.__dict__)
        exec(compile(wheel.read("omnivoice/utils/text.py"), "upstream_text.py", "exec"), upstream.__dict__)
        source = ast.parse(wheel.read("omnivoice/models/omnivoice.py"))
    nodes = []
    method_names = {"_preprocess_all", "_estimate_target_tokens", "_ensure_list"}
    for node in source.body:
        if isinstance(node, ast.ClassDef) and node.name == "GenerationTask":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "OmniVoice":
            nodes.append(ast.ClassDef(name="PreprocessMethods", bases=[], keywords=[], body=[method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in method_names], decorator_list=[]))
        elif isinstance(node, ast.FunctionDef) and node.name == "_combine_text":
            nodes.append(node)
    upstream.__dict__.update(dataclass=dataclass, _resolve_language=lambda value: value, _resolve_instruct=resolve_instruction, _normalize_text=lambda text, language: normalizer.normalize_text(text, language), _ZH_RE=re.compile(r"[\u4e00-\u9fff]"))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "upstream_preprocess.py", "exec"), upstream.__dict__)
    for name in method_names:
        setattr(Model, name, getattr(upstream.PreprocessMethods, name))
    normalizer.chunk_text_punctuation = upstream.chunk_text_punctuation
    core._combine_text = upstream._combine_text
    core._tokenize_with_nonverbal_tags = lambda text, tokenizer: tokenizer(text, return_tensors="pt").input_ids
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **options: "/fake/pinned-snapshot"
    numbers = types.ModuleType("num2words")
    numbers.CONVERTER_CLASSES = {"en": object(), "ru": object()}
    for name, module in {"torch": torch, "soundfile": sf, "huggingface_hub": hub, "omnivoice": omni,
                         "omnivoice.models": types.ModuleType("omnivoice.models"), "omnivoice.models.omnivoice": core,
                         "omnivoice.utils": types.ModuleType("omnivoice.utils"), "omnivoice.utils.lang_map": languages,
                         "omnivoice.utils.voice_design": attributes, "omnivoice.utils.text": normalizer, "num2words": numbers}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(engine_module.importlib.util, "find_spec", lambda name: object() if name in ("tn", "pynini") else None)
    asr = tmp_path / "whisper"
    asr.mkdir()
    for name in ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json"):
        (asr / name).write_text("fake local snapshot")
    settings = dict(model_id="k2-fsa/OmniVoice", model_revision="pinned-revision", cache_dir=tmp_path / "cache", voices_dir=tmp_path / "voices", device="cpu", asr_model_path=asr, auto_download_models=False)
    engine = OmniVoiceEngine(**settings)
    engine.load()
    return types.SimpleNamespace(engine=engine, model=models[-1], models=models, sf=sf, prompts=prompts, settings=settings, seeds=seeds, tmp=tmp_path, normalizer=normalizer, upstream=upstream)


def test_import_has_no_ml_dependencies():
    result = subprocess.run([sys.executable, "-c", "import sys,engine; assert not {'torch','omnivoice','numpy','soundfile'} & set(sys.modules)"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_startup_prepares_pinned_models_before_inference(runtime, monkeypatch):
    import model_assets

    calls = []

    def prepare(name, destination):
        directory = Path(destination)
        required = ("config.json", "model.safetensors", "tokenizer.json", "audio_tokenizer/config.json") if name == "omnivoice" else (
            "config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json")
        for filename in required:
            target = directory / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("prepared test snapshot")
        calls.append((name, directory))
        return directory

    monkeypatch.setattr(model_assets, "ensure_snapshot", prepare)
    model = OmniVoiceEngine(**{**runtime.settings, "model_revision": model_assets.MODELS["omnivoice"]["revision"],
        "asr_model_path": None, "voices_dir": runtime.tmp / "auto-voices", "auto_download_models": True})
    model.load()
    assert [name for name, _ in calls] == ["omnivoice", "whisper"]
    assert model.model_path == runtime.tmp / "cache/models/omnivoice"
    assert model.asr_model_path == runtime.tmp / "cache/models/whisper"
    assert model._ready and model._asr_available()


def test_download_optout_preserves_transcription_free_startup(runtime, monkeypatch):
    import model_assets

    monkeypatch.setattr(model_assets, "ensure_snapshot", lambda *a, **k: pytest.fail("download attempted"))
    model = OmniVoiceEngine(**{**runtime.settings, "asr_model_path": None})
    model.load()
    assert model._ready and not model._asr_available()


def test_custom_model_snapshot_is_not_repaired_as_the_default_checkpoint(runtime, monkeypatch):
    import model_assets

    directory = runtime.tmp / "custom-model"
    for filename in ("config.json", "model.safetensors", "tokenizer.json", "audio_tokenizer/config.json"):
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("custom snapshot")
    def prepare(name, destination):
        assert name == "whisper", "custom TTS snapshot must not be replaced"
        return Path(destination)

    monkeypatch.setattr(model_assets, "ensure_snapshot", prepare)
    model = OmniVoiceEngine(**{**runtime.settings, "model_path": directory, "auto_download_models": True})
    model.load()
    assert model._ready and (directory / "config.json").read_text() == "custom snapshot"


def test_explicit_whisper_path_still_runs_manifest_completeness_checks(runtime, monkeypatch):
    import model_assets

    # All four old capability-check sentinel files exist, but that must not
    # bypass detection/repair of truncated weights or absent tokenizer files.
    calls = []

    def prepare(name, destination):
        calls.append(name)
        return Path(destination)

    monkeypatch.setattr(model_assets, "ensure_snapshot", prepare)
    model = OmniVoiceEngine(**{**runtime.settings, "auto_download_models": True})
    model.load()
    assert calls == ["whisper"]
    assert model._ready


def test_custom_hub_model_honors_transformers_offline_flag(runtime, monkeypatch):
    import model_assets

    calls = []
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(model_assets, "ensure_snapshot", lambda name, destination: Path(destination))

    def snapshot(**options):
        calls.append(options)
        return "/fake/pinned-snapshot"

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", snapshot)
    model = OmniVoiceEngine(**{**runtime.settings, "auto_download_models": True})
    model.load()
    assert calls[0]["local_files_only"] is True


def accelerated_engine(runtime, monkeypatch, apply):
    flash = types.ModuleType("flashinfer")
    flash.norm = types.SimpleNamespace(_USE_CUDA_NORM=True)
    monkeypatch.setitem(sys.modules, "flashinfer", flash)
    module = types.ModuleType("omnivoice.models.omnivoice_flashinfer")
    module.apply_flashinfer = apply
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.0a")
    monkeypatch.setenv("FLASHINFER_USE_CUDA_NORM", "1")
    return OmniVoiceEngine(**{**runtime.settings, "device": "cuda:0", "acceleration": "flashinfer"})


def test_baseline_does_not_import_or_initialize_accelerator(runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice_flashinfer", None)
    engine = OmniVoiceEngine(**runtime.settings)
    engine.load()
    assert engine._ready and engine.capabilities["inference_acceleration"] == "none"
    assert not runtime.models[-1].calls  # Existing profiles need no synthesis.


def test_flashinfer_preserves_profiles_quality_controls_and_loads_once(runtime, monkeypatch):
    before = {str(p.relative_to(runtime.engine.voices_dir)): p.read_bytes()
              for p in runtime.engine.voices_dir.rglob("*") if p.is_file()}
    applied = []

    def apply(model, **options):
        applied.append(options)
        assert not model.calls

    engine = accelerated_engine(runtime, monkeypatch, apply)
    engine.load()
    engine.load()
    assert applied == [{"enable_cuda_graph": False}]
    assert engine._ready and engine.capabilities["inference_acceleration"] == "flashinfer"
    assert len(runtime.models[-1].calls) == 1  # Kernel warmup, no profile creation.
    assert runtime.seeds[-1] == 1729
    engine.execute("speech", {"text": "Davom etish uchun tilni tanlang.", "language": "uz", "voice": "female_1"})
    config = runtime.models[-1].calls[-1]["generation_config"]
    assert config.num_step == 32 and config.denoise and config.preprocess_prompt and config.postprocess_output
    assert (config.audio_chunk_duration, config.audio_chunk_threshold, config.pad_duration, config.fade_duration) == (15, 30, 0.1, 0.1)
    assert {str(p.relative_to(engine.voices_dir)): p.read_bytes()
            for p in engine.voices_dir.rglob("*") if p.is_file()} == before


def test_new_builtin_profiles_are_created_before_accelerator_patch(runtime, monkeypatch):
    def apply(model, **options):
        assert len(model.calls) == 4
        assert all(call["num_step"] == 32 for call in model.calls)
        assert len(list((runtime.tmp / "fresh-voices").glob("*/manifest.json"))) == 4

    engine = accelerated_engine(runtime, monkeypatch, apply)
    engine.voices_dir = runtime.tmp / "fresh-voices"
    engine.load()
    assert engine._ready and len(runtime.models[-1].calls) == 5


@pytest.mark.parametrize("failure", ["import", "partial_patch", "kernel"])
def test_accelerator_failures_never_serve_or_fall_back_on_partially_patched_model(runtime, monkeypatch, failure):
    def apply(model, **options):
        model.partially_patched = True
        if failure == "partial_patch":
            raise RuntimeError("Patch failed")
        if failure == "kernel":
            model.result = [np.zeros(24000, dtype=np.float32)]

    engine = accelerated_engine(runtime, monkeypatch, apply)
    if failure == "import":
        monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice_flashinfer", None)
    with pytest.raises(RuntimeError, match="restart with TTS_ACCELERATION=none"):
        engine.load()
    calls = len(runtime.models[-1].calls)
    assert not engine._ready and engine._active_acceleration == "none"
    with pytest.raises(RuntimeError, match="not loaded"):
        engine.execute("speech", {"text": "Salom", "voice": "female_1"})
    assert len(runtime.models[-1].calls) == calls


@pytest.mark.parametrize("unsupported", ["cpu", "dtype", "capability", "cuda", "target", "norm_setting", "rope"])
def test_acceleration_rejects_unvalidated_runtime_before_patching(runtime, monkeypatch, unsupported):
    applied = []
    engine = accelerated_engine(runtime, monkeypatch, lambda *args, **kwargs: applied.append(True))
    torch = sys.modules["torch"]
    if unsupported == "cpu":
        engine.device = "cpu"
    elif unsupported == "dtype":
        torch.float16 = "inconsistent-fp16"
        original = runtime.model.__class__.from_pretrained.__func__

        def load(cls, *args, **kwargs):
            model = original(cls, *args, **kwargs)
            model.dtype = torch.float32
            return model

        monkeypatch.setattr(runtime.model.__class__, "from_pretrained", classmethod(load))
    elif unsupported == "capability":
        torch.cuda.get_device_capability = lambda device: (9, 0)
    elif unsupported == "cuda":
        torch.version.cuda = "13.0"
    elif unsupported == "target":
        monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.1a")
    elif unsupported == "norm_setting":
        monkeypatch.delenv("FLASHINFER_USE_CUDA_NORM")
    else:
        original = runtime.model.__class__.from_pretrained.__func__

        def load(cls, *args, **kwargs):
            model = original(cls, *args, **kwargs)
            model.llm.config.rope_parameters = {"rope_type": "unsupported"}
            return model

        monkeypatch.setattr(runtime.model.__class__, "from_pretrained", classmethod(load))
    with pytest.raises(RuntimeError, match="validated CUDA"):
        engine.load()
    assert not engine._ready and not applied


def test_preimported_unvalidated_normalization_backend_cannot_patch_or_serve(runtime, monkeypatch):
    applied = []
    engine = accelerated_engine(runtime, monkeypatch, lambda *args, **kwargs: applied.append(True))
    sys.modules["flashinfer"].norm._USE_CUDA_NORM = False
    with pytest.raises(RuntimeError, match="restart with TTS_ACCELERATION=none"):
        engine.load()
    assert not engine._ready and not applied


def test_unknown_acceleration_setting_is_rejected(runtime):
    with pytest.raises(ValueError, match="acceleration"):
        OmniVoiceEngine(**runtime.settings, acceleration="automatic")


@pytest.mark.parametrize("paths", [{}, {"tensorrt_plan": "forward.plan"}, {"tensorrt_manifest": "manifest.json"}])
def test_tensorrt_requires_both_artifact_paths_before_loading(runtime, paths):
    with pytest.raises(ValueError, match="pinned plan and manifest"):
        OmniVoiceEngine(**runtime.settings, acceleration="tensorrt", **paths)


def test_backward_compatibility_controls_and_nullable_language(runtime):
    engine, model = runtime.engine, runtime.model
    result = engine.synthesize("Salom", "uz", "female_1")
    with wave.open(io.BytesIO(result)) as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
    engine.execute("speech", {"text": "Hello", "language": None, "mode": "design", "gender": "female", "instruct": "whisper", "speed": 1.2, "duration": 30, "seed": 42, "generation": {"audio_chunk_duration": 3, "audio_chunk_threshold": 3, "num_step": 16, "postprocess_output": False}})
    options = model.calls[-1]
    assert options["instruct"] == "female, whisper" and options["language"] is None
    assert options["speed"] == 1.2 and options["duration"] == 30
    assert options["generation_config"].audio_chunk_threshold > 30
    assert options["generation_config"].num_step == 16 and not options["generation_config"].postprocess_output
    assert runtime.seeds[-1] == 42
    engine.execute("speech", {"text": "Hello", "mode": "auto"})
    assert "instruct" not in model.calls[-1] and "voice_clone_prompt" not in model.calls[-1]


@pytest.mark.parametrize("settings", [
    {"instruct": "male"}, {"instruct": "unsupported attribute"}, {"gender": "male"},
    {"generation": {"unknown": 1}}, {"generation": {"t_shift": 0}}, {"generation": {"num_step": True}},
    {"generation": {"audio_chunk_threshold": 3}}, {"speed": float("nan")}, {"duration": 31}, {"seed": -1},
    {"normalize_text": True}, {"language": "unknown"},
])
def test_invalid_controls_rejected_before_inference(runtime, settings):
    count = len(runtime.model.calls)
    with pytest.raises(ValueError):
        runtime.engine.execute("speech", {"text": "Salom", "language": "uz", "voice": "female_1", **settings})
    assert len(runtime.model.calls) == count


def test_capabilities_and_preset_instruction_augmentation(runtime):
    caps = runtime.engine.capabilities
    assert caps["asr_available"] and not caps["asr_loaded"]
    assert set(caps["normalization_languages"]) == {"en", "zh", "ru"}
    assert caps["normalization_modes"]["ru"] == "integers"
    runtime.engine.execute("speech", {"text": "Hello", "language": "en", "voice": "female_1", "instruct": "female, whisper", "normalize_text": True})
    assert runtime.model.calls[-1]["normalize_text"] is False
    assert "voice_clone_prompt" in runtime.model.calls[-1]


@pytest.mark.parametrize("audio", [b"not audio", b"", wav_bytes(silent=True), b"x" * (8 * 1024 * 1024 + 1)], ids=["not-audio", "empty", "silence", "large-invalid-file"])
def test_invalid_uploads_do_not_reach_model(runtime, audio):
    count = len(runtime.prompts)
    with pytest.raises(ValueError):
        runtime.engine.execute("clone_speech", {"text": "Salom", "audio": audio, "ref_text": "Reference"})
    assert len(runtime.prompts) == count and not runtime.model.asr_loads


@pytest.mark.parametrize("seconds,rate,channels", [(1, 24000, 1), (12, 24000, 1), (4, 7999, 1), (4, 24000, 3), (12, 192000, 2)])
def test_valid_reference_audio_has_no_old_duration_size_rate_or_channel_rejection(runtime, seconds, rate, channels):
    encoded = wav_bytes(seconds=seconds, rate=rate, channels=channels)
    if rate == 192000:
        assert len(encoded) > 8 * 1024 * 1024
    result = runtime.engine.execute("clone_speech", {
        "text": "Salom.", "audio": encoded, "ref_text": "The exact recording transcript."})
    assert result.startswith(b"RIFF")
    samples, actual_rate = runtime.prompts[-1]["ref_audio"]
    assert actual_rate == rate and samples.ndim == 1
    assert len(samples) == seconds * rate
    assert not runtime.model.asr_loads


@pytest.mark.parametrize("field", ["samplerate", "channels", "frames"])
def test_invalid_nonpositive_audio_header_is_rejected_before_decoding(runtime, field):
    info = types.SimpleNamespace(format="WAV", samplerate=24000, channels=1, frames=24000)
    setattr(info, field, 0)
    runtime.sf.special[b"invalid header"] = (info, np.ones((1, 1), dtype=np.float32))
    reads = runtime.sf.read_calls
    with pytest.raises(ValueError):
        runtime.engine._decode_upload(b"invalid header")
    assert runtime.sf.read_calls == reads


def test_header_must_match_decoded_frames_and_nonfinite_audio_is_rejected(runtime):
    info = types.SimpleNamespace(format="WAV", samplerate=24000, channels=1, frames=24000 * 60)
    runtime.sf.special[b"huge header"] = (info, np.zeros((1, 1)))
    reads = runtime.sf.read_calls
    with pytest.raises(ValueError):
        runtime.engine.execute("transcribe", {"audio": b"huge header"})
    assert runtime.sf.read_calls == reads + 1
    info.frames = 24000 * 4
    runtime.sf.special[b"nan audio"] = (info, np.full((info.frames, 1), np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        runtime.engine.execute("transcribe", {"audio": b"nan audio"})
    assert not runtime.model.asr_loads


def test_staged_wav_uses_bounded_stream_reads_and_closes_temporary_mapping(runtime, monkeypatch):
    source = runtime.tmp / "staged-recording.wav"
    source.write_bytes(wav_bytes(seconds=61, rate=16000, channels=2))
    reads = []

    def info(path):
        with wave.open(str(path), "rb") as stream:
            return types.SimpleNamespace(format="WAV", samplerate=stream.getframerate(),
                                         channels=stream.getnchannels(), frames=stream.getnframes())

    class StreamingSoundFile:
        def __init__(self, path):
            self.stream = wave.open(str(path), "rb")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, count, **options):
            assert count <= 65536 and options == {"dtype": "float32", "always_2d": True}
            reads.append(count)
            return (np.frombuffer(self.stream.readframes(count), dtype="<i2").astype(np.float32) / 32768).reshape(-1, 2)

    original_read = Path.read_bytes
    def forbid_full_source_read(path):
        if path == source:
            pytest.fail("A staged audio upload must be decoded from its file stream")
        return original_read(path)

    monkeypatch.setattr(runtime.sf, "info", info)
    monkeypatch.setattr(runtime.sf, "SoundFile", StreamingSoundFile, raising=False)
    monkeypatch.setattr(Path, "read_bytes", forbid_full_source_read)
    with runtime.engine._uploaded_audio(source) as (audio, rate):
        assert isinstance(audio, np.memmap)
        mapping, target = audio._mmap, Path(audio.filename)
        assert rate == 16000 and len(audio) == 61 * rate
        assert np.isfinite(audio).all() and np.sqrt(np.mean(audio[:rate] ** 2)) > .01
        assert target.is_file() and len(reads) > 1
    assert mapping.closed and not target.parent.exists()
    assert source.is_file()


def test_staged_compressed_audio_keeps_input_on_disk_and_closes_mapping_on_failure(runtime, media, monkeypatch):
    source = runtime.tmp / "staged-phone-recording.m4a"
    source.write_bytes(media.encode("aac", "m4a", source=wav_bytes(seconds=35)))
    original_read = Path.read_bytes
    def forbid_full_source_read(path):
        if path == source:
            pytest.fail("Compressed uploads must be passed to the decoder by path")
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", forbid_full_source_read)
    with pytest.raises(RuntimeError, match="Synthetic caller failure"):
        with runtime.engine._uploaded_audio(source) as (audio, rate):
            assert isinstance(audio, np.memmap) and rate == 24000
            assert abs(len(audio) / rate - 35) < .2
            assert np.isfinite(audio[:rate]).all()
            mapping, target = audio._mmap, Path(audio.filename)
            raise RuntimeError("Synthetic caller failure")
    assert mapping.closed and not target.parent.exists() and source.is_file()


def test_direct_audio_decode_failure_is_an_input_error_without_fallback(runtime, monkeypatch):
    def broken_read(*args, **options):
        raise RuntimeError("Private libsndfile decoding details")

    def unexpected_fallback(encoded):
        pytest.fail("A corrupt WAV must not bypass direct decoding checks")

    monkeypatch.setattr(runtime.sf, "read", broken_read)
    monkeypatch.setattr(runtime.engine, "_decode_media_upload", unexpected_fallback)
    with pytest.raises(ValueError, match="could not be decoded"):
        runtime.engine.execute("transcribe", {"audio": wav_bytes()})
    assert not runtime.model.asr_loads


@pytest.mark.parametrize("codec,extension,options", [
    ("libmp3lame", "mp3", ()),
    ("aac", "m4a", ()),
    ("alac", "m4a", ()),
    ("aac", "aac", ()),
    ("libopus", "ogg", ()),
    ("libvorbis", "ogg", ()),
    ("libopus", "webm", ()),
    ("libopencore_amrnb", "amr", ("-ar", "8000", "-b:a", "12.2k")),
    ("aac", "3gp", ("-ar", "16000")),
    ("wmav2", "wma", ("-ar", "44100")),
    ("pcm_s16be", "aiff", ()),
    ("pcm_s16le", "caf", ()),
], ids=["mp3", "m4a-aac", "m4a-alac", "aac", "ogg-opus", "ogg-vorbis",
        "webm-opus", "amr", "3gp", "wma", "aiff", "caf"])
def test_real_phone_and_audio_formats_decode_to_finite_mono(runtime, media, codec, extension, options):
    encoded = media.encode(codec, extension, options=options)
    calls = len(runtime.model.calls)
    prompts = len(runtime.prompts)
    audio, rate = runtime.engine._decode_upload(encoded)
    assert rate == 24000 and audio.ndim == 1
    assert audio.dtype == np.float32 and audio.flags.c_contiguous
    assert abs(len(audio) / rate - 4) < 0.2
    assert np.isfinite(audio).all() and np.sqrt(np.mean(audio ** 2)) > 0.01
    assert len(runtime.model.calls) == calls and len(runtime.prompts) == prompts
    assert not runtime.model.asr_loads


@pytest.mark.parametrize("extension,codec", [("wav", "pcm_s16le"), ("flac", "flac")])
def test_wav_and_flac_preserve_direct_decoder_metadata(runtime, media, monkeypatch, extension, codec):
    source = wav_bytes(rate=16000, channels=2)
    encoded = media.encode(codec, extension, source=source)
    if extension == "flac":
        # The fixture replaces libsndfile; supply its lossless FLAC metadata and samples.
        with wave.open(io.BytesIO(source)) as original:
            samples = np.frombuffer(original.readframes(original.getnframes()), dtype="<i2").astype(np.float32) / 32768
        info = types.SimpleNamespace(format="FLAC", samplerate=16000, channels=2, frames=64000)
        runtime.sf.special[encoded] = (info, samples.reshape(-1, 2))

    def unexpected_fallback(encoded):
        pytest.fail("Valid WAV/FLAC must retain the direct decoder path")

    monkeypatch.setattr(runtime.engine, "_decode_media_upload", unexpected_fallback)
    audio, rate = runtime.engine._decode_upload(encoded)
    assert rate == 16000 and audio.shape == (64000,)
    assert audio.dtype == np.float32 and np.isfinite(audio).all()


@pytest.mark.parametrize("frames", [1, 23999, 71999, 72000, 240000, 240001, 288000])
def test_real_media_preserves_full_decoded_audio_without_duration_rejection(runtime, media, frames):
    source = wav_bytes(seconds=(frames + 0.1) / 24000)
    if frames == 1:
        source = source[:-2] + b"\x10\x27"
    with wave.open(io.BytesIO(source)) as audio:
        assert audio.getnframes() == frames
    encoded = media.encode("pcm_s16le", "caf", source=source)
    audio, rate = runtime.engine._decode_upload(encoded)
    assert rate == 24000 and len(audio) == frames
    assert not runtime.model.asr_loads


def test_real_media_rejects_truncated_video_only_and_playlist_inputs(runtime, media, tmp_path):
    complete = media.encode("aac", "m4a")
    no_audio = tmp_path / "video-only.mp4"
    result = subprocess.run(
        [media.executable, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=16x16:r=1:d=4", "-an", "-c:v", "mpeg4", str(no_audio)],
        capture_output=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    referenced = tmp_path / "external-reference.wav"
    referenced.write_bytes(wav_bytes())
    playlist = f"ffconcat version 1.0\nfile '{referenced.as_posix()}'\n".encode()
    hls = b"#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXTINF:4,\nhttp://127.0.0.1:1/audio.ts\n#EXT-X-ENDLIST\n"
    prompts, calls = len(runtime.prompts), len(runtime.model.calls)
    for encoded in (complete[:len(complete) // 2], no_audio.read_bytes(), playlist, hls):
        with pytest.raises(ValueError):
            runtime.engine.execute("clone_speech", {"audio": encoded, "text": "Salom.", "ref_text": "Reference."})
    assert len(runtime.prompts) == prompts and len(runtime.model.calls) == calls
    assert not runtime.model.asr_loads


def test_real_media_timestamp_gap_does_not_insert_unbounded_silence(runtime, media):
    encoded = media.encode("pcm_s16le", "mka", options=("-af", "asetpts=PTS+20/TB*gte(T\\,2)"))
    audio, rate = runtime.engine._decode_upload(encoded)
    assert rate == 24000 and len(audio) == 4 * rate
    assert np.sqrt(np.mean(audio ** 2)) > 0.01


@pytest.mark.parametrize("failure,expected", [("timeout", ValueError), ("nonzero", ValueError), ("start", RuntimeError)])
def test_media_subprocess_failures_preserve_isolation_and_clean_temporary_files(runtime, monkeypatch, failure, expected):
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "test-ffmpeg")
    directories = []

    def fail(command, **options):
        directory = Path(command[-1]).parent
        directories.append(directory)
        assert (directory / "input.media").read_bytes() == b"encoded media"
        assert "timeout" not in options
        assert options["stdin"] == options["stdout"] == options["stderr"] == subprocess.DEVNULL
        assert command[command.index("-protocol_whitelist") + 1] == "file"
        formats = set(command[command.index("-format_whitelist") + 1].split(","))
        assert not {"concat", "hls", "image2", "sdp"} & formats
        assert "-t" not in command and "-fs" not in command
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if failure == "start":
            raise OSError("Cannot start decoder")
        return types.SimpleNamespace(returncode=1)

    monkeypatch.setattr(engine_module.subprocess, "run", fail)
    with pytest.raises(expected):
        runtime.engine._decode_upload(b"encoded media")
    assert len(directories) == 1 and not directories[0].exists()
    assert not runtime.model.asr_loads
    # A decoder failure must not prevent the next direct WAV request.
    audio, rate = runtime.engine._decode_upload(wav_bytes())
    assert rate == 24000 and len(audio) == 4 * rate


@pytest.mark.parametrize("output", [
    np.zeros(72000, dtype="<f4").tobytes(),
    np.full(72000, np.nan, dtype="<f4").tobytes(),
    b"",
    b"unaligned",
], ids=["silence", "nonfinite", "empty", "unaligned"])
def test_media_decoder_rejects_invalid_decoded_output_and_cleans_up(runtime, monkeypatch, output):
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "test-ffmpeg")
    targets = []

    def decoded(command, **options):
        target = Path(command[-1])
        targets.append(target)
        target.write_bytes(output)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(engine_module.subprocess, "run", decoded)
    with pytest.raises(ValueError):
        runtime.engine.execute("clone_speech", {"audio": b"encoded media", "text": "Salom.", "ref_text": "Reference."})
    assert len(targets) == 1 and not targets[0].parent.exists()
    assert not runtime.model.asr_loads


def test_local_asr_lazy_once_and_blank_transcript_rejected(runtime):
    audio = wav_bytes(channels=2, rate=16000)
    payload = {"text": "Salom", "audio": audio, "language": "uz"}
    runtime.engine.execute("clone_speech", payload)
    runtime.engine.execute("transcribe", {"audio": audio})
    assert runtime.model.asr_loads == [{"model_name": str(runtime.settings["asr_model_path"].resolve()), "device": "cpu"}]
    assert runtime.engine.capabilities["asr_loaded"]
    assert runtime.prompts[-1]["ref_audio"][0].ndim == 1
    runtime.model.transcript = "   "
    with pytest.raises(ValueError):
        runtime.engine.execute("clone_speech", payload)
    runtime.engine.asr_model_path = None
    with pytest.raises(RuntimeError):
        runtime.engine.execute("transcribe", {"audio": audio})


@pytest.fixture
def reference_asr(runtime, monkeypatch):
    class Pipeline:
        def __init__(self):
            self.calls = []
            self.model = types.SimpleNamespace(generation_config=types.SimpleNamespace(
                lang_to_id={"<|uz|>": 50337, "<|en|>": 50259}, language=None, task=None))

        def __call__(self, audio, **options):
            self.calls.append({"seconds": len(audio["array"]) / audio["sampling_rate"],
                               "rate": audio["sampling_rate"], "dimensions": audio["array"].ndim, **options})
            return {"text": "Selected reference words." if options else "Automatic reference words."}

    pipeline = Pipeline()
    runtime.model._asr_pipe = pipeline
    # Mirror the pinned model's no-hint transcription path. This lets a later
    # autodetection call prove that an earlier forced language did not leak.
    def transcribe(recording):
        samples, rate = recording
        return pipeline({"array": np.squeeze(samples), "sampling_rate": rate})["text"].strip()
    monkeypatch.setattr(runtime.model, "transcribe", transcribe)
    return pipeline


def test_new_clone_uses_supported_recording_language_only_for_automatic_transcription(runtime, reference_asr):
    config_before = dict(reference_asr.model.generation_config.__dict__)
    result = runtime.engine.execute("clone_voice", {
        "audio": wav_bytes(), "name": "Uzbek reference", "gender": "female", "reference_language": "uz"})
    assert result["ref_text"] == "Selected reference words."
    assert reference_asr.calls == [{"seconds": 4, "rate": 24000, "dimensions": 1,
                                    "generate_kwargs": {"language": "uz", "task": "transcribe"}}]
    assert reference_asr.model.generation_config.__dict__ == config_before
    assert len(runtime.model.asr_loads) == 1
    supplied = runtime.engine.execute("clone_voice", {
        "audio": wav_bytes(), "name": "Explicit words", "gender": "male", "reference_language": "uz",
        "ref_text": "These exact supplied words."})
    assert supplied["ref_text"] == "These exact supplied words."
    assert len(reference_asr.calls) == 1, "a valid supplied transcript must not be replaced by ASR"


@pytest.mark.parametrize("reason", ["long_crop", "oversized_transcript"])
def test_recording_language_reaches_automatic_crop_and_transcript_replacement(runtime, reference_asr, reason):
    result = runtime.engine.execute("clone_voice", {
        "audio": wav_bytes(seconds=35 if reason == "long_crop" else 4),
        "name": "Aligned reference", "gender": "female", "reference_language": "uz",
        "ref_text": "Full original recording text." if reason == "long_crop" else "Long transcript. " * 300})
    assert result["ref_text"] == "Selected reference words."
    assert result["reference_preparation"]["automatic_transcript"] is True
    assert len(reference_asr.calls) == 1
    assert reference_asr.calls[0]["generate_kwargs"] == {"language": "uz", "task": "transcribe"}
    if reason == "long_crop":
        assert 10 <= reference_asr.calls[0]["seconds"] <= 15
    else:
        assert result["reference_preparation"]["transcript_replaced"] is True


def test_autodetection_after_forced_reference_and_unsupported_language_remains_unchanged(runtime, reference_asr):
    base = {"audio": wav_bytes(), "name": "Reference", "gender": "female"}
    runtime.engine.execute("clone_voice", {**base, "reference_language": "uz"})
    none = runtime.engine.execute("clone_voice", base)
    unsupported = runtime.engine.execute("clone_voice", {**base, "reference_language": "zh"})
    assert none["reference_language"] is None and unsupported["reference_language"] == "zh"
    assert none["ref_text"] == unsupported["ref_text"] == "Automatic reference words."
    assert [call.get("generate_kwargs") for call in reference_asr.calls] == [
        {"language": "uz", "task": "transcribe"}, None, None]
    assert reference_asr.model.generation_config.language is None
    assert reference_asr.model.generation_config.task is None
    assert len(runtime.model.asr_loads) == 1


def test_one_shot_target_language_does_not_force_reference_language_and_old_transcribe_call_works(runtime, reference_asr):
    runtime.engine.execute("clone_speech", {"audio": wav_bytes(), "text": "Target text.", "language": "uz"})
    assert "generate_kwargs" not in reference_asr.calls[-1]
    assert runtime.prompts[-1]["ref_text"] == "Automatic reference words."
    audio, rate = runtime.engine._decode_upload(wav_bytes())
    assert runtime.engine._transcribe_samples(audio, rate) == "Automatic reference words."
    assert all("generate_kwargs" not in call for call in reference_asr.calls)


def test_explicit_reference_asr_failure_has_no_partially_published_profile(runtime, reference_asr, monkeypatch):
    def fail(*args, **options):
        raise RuntimeError("Private recognizer failure")
    monkeypatch.setattr(type(reference_asr), "__call__", fail)
    with pytest.raises(RuntimeError, match="Reference transcription failed"):
        runtime.engine.execute("clone_voice", {
            "audio": wav_bytes(), "name": "Failed reference", "gender": "female", "reference_language": "uz"})
    assert not runtime.engine._custom_ids
    assert not any(voice.get("source") == "clone" for voice in runtime.engine.voices)


def test_full_recording_transcription_keeps_all_ordered_windows(runtime, monkeypatch):
    lengths = []
    def transcribe(recording):
        samples, rate = recording
        lengths.append(len(samples) / rate)
        return f"Part {len(lengths)}."
    monkeypatch.setattr(runtime.model, "transcribe", transcribe)
    result = runtime.engine.execute("transcribe", {"audio": wav_bytes(seconds=55)})
    assert lengths == [25, 25, 5]
    assert result == {"text": "Part 1. Part 2. Part 3."}
    assert len(runtime.model.asr_loads) == 1


def test_long_reference_uses_voiced_excerpt_and_matching_transcript_that_survives_reload(runtime, monkeypatch):
    encoded = wav_bytes(seconds=35)
    original, rate = runtime.engine._decode_upload(encoded)
    original[:5 * rate] = 0
    encoded = runtime.engine._wav(original, rate)
    transcribed = []
    def transcribe(recording):
        samples, current_rate = recording
        transcribed.append((samples.copy(), current_rate))
        return "Selected reference phrase"
    monkeypatch.setattr(runtime.model, "transcribe", transcribe)
    metadata = runtime.engine.execute("clone_voice", {
        "audio": encoded, "ref_text": "Full recording words must not be attached to a crop.",
        "name": "Long recording", "gender": "female", "reference_language": "en"})
    preparation = metadata["reference_preparation"]
    assert preparation["source_seconds"] == 35
    assert 4 <= preparation["selected_start_seconds"] <= 5
    assert 10 <= preparation["selected_seconds"] <= 15
    assert preparation["automatic_transcript"] is True
    assert metadata["ref_text"] == "Selected reference phrase."
    assert len(transcribed) == 1
    selected, selected_rate = runtime.prompts[-1]["ref_audio"]
    assert selected_rate == rate
    assert np.array_equal(selected, transcribed[0][0])
    with wave.open(io.BytesIO(runtime.engine.execute("voice_sample", {"voice": metadata["id"]}))) as audio:
        assert audio.getnframes() / audio.getframerate() == preparation["selected_seconds"]
    reloaded = OmniVoiceEngine(**runtime.settings)
    reloaded.load()
    assert metadata in reloaded.voices
    assert reloaded._prompts[metadata["id"]].ref_text == metadata["ref_text"]


def test_subframe_reference_is_padded_only_to_the_audio_codec_minimum(runtime):
    source = np.array([0.25], dtype=np.float32)
    prompt = runtime.engine._make_prompt(source, 24000, "One sample", False)
    samples, rate = runtime.prompts[-1]["ref_audio"]
    assert rate == 24000 and len(samples) == 960
    assert samples[0] == source[0] and not np.any(samples[1:])
    assert prompt.ref_audio_tokens.size(-1) == 1


def test_oversized_supplied_reference_transcript_is_aligned_automatically_before_saving(runtime):
    metadata = runtime.engine.execute("clone_voice", {
        "audio": wav_bytes(), "ref_text": "Oversized transcript. " * 600,
        "name": "Aligned recording", "gender": "female", "reference_language": "en"})
    assert metadata["ref_text"] == "Recognized reference speech."
    assert metadata["reference_preparation"]["automatic_transcript"] is True
    assert len(runtime.model.asr_loads) == 1
    runtime.engine.synthesize("This voice is reusable.", "en", metadata["id"])
    reloaded = OmniVoiceEngine(**runtime.settings)
    reloaded.load()
    assert metadata in reloaded.voices
    reloaded.synthesize("The persisted voice is reusable.", "en", metadata["id"])


@pytest.mark.parametrize("compressed", [False, True], ids=["wav", "m4a"])
def test_cloned_voice_persists_and_builtins_are_unchanged(runtime, media, monkeypatch, compressed):
    before = {path: path.read_bytes() for spec in VOICE_SPECS for path in (runtime.settings["voices_dir"] / spec["id"]).iterdir()}
    prompt_count = len(runtime.prompts)
    reference = wav_bytes(rate=16000, channels=2)
    if compressed:
        reference = media.encode("aac", "m4a", source=reference)
    conversions = []
    decode_media = OmniVoiceEngine._decode_media_upload

    def counted_decoder(self, encoded):
        conversions.append(encoded)
        return decode_media(self, encoded)

    monkeypatch.setattr(OmniVoiceEngine, "_decode_media_upload", counted_decoder)
    metadata = runtime.engine.execute("clone_voice", {"audio": reference, "ref_text": "My reference", "name": "My voice", "gender": "female", "reference_language": "en", "preprocess_prompt": False})
    assert metadata["id"].startswith("clone_") and metadata["synthetic"] is False
    assert metadata["ref_text"] == "My reference" and metadata["sample_url"].endswith("/sample")
    assert runtime.prompts[-1]["preprocess_prompt"] is False
    assert not any("path" in key for key in metadata)
    sample = runtime.engine.execute("voice_sample", {"voice": metadata["id"]})
    with wave.open(io.BytesIO(sample)) as audio:
        assert audio.getframerate() == (24000 if compressed else 16000) and audio.getnchannels() == 1
    assert len(conversions) == int(compressed)
    assert len(runtime.prompts) == prompt_count + 1
    saved_prompt = runtime.engine._prompts[metadata["id"]]
    call_count = len(runtime.model.calls)
    texts = ["Davom etish uchun tilni tanlang.", "Yangi matn ham shu saqlangan ovozda o'qiladi."]
    for text in texts:
        runtime.engine.synthesize(text, "uz", metadata["id"])
    calls = runtime.model.calls[call_count:]
    assert [call["text"] for call in calls] == texts
    assert all(call["voice_clone_prompt"] is saved_prompt for call in calls)
    assert len(runtime.prompts) == prompt_count + 1
    reloaded = OmniVoiceEngine(**runtime.settings)
    reloaded.load()
    assert metadata in reloaded.voices and not runtime.models[-1].calls
    reloaded_prompt = reloaded._prompts[metadata["id"]]
    assert reloaded_prompt is not saved_prompt
    reloaded.synthesize(texts[0], "uz", metadata["id"])
    assert runtime.models[-1].calls[-1]["text"] == texts[0]
    assert runtime.models[-1].calls[-1]["voice_clone_prompt"] is reloaded_prompt
    assert len(runtime.prompts) == prompt_count + 1
    assert len(conversions) == int(compressed)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_designed_voice_normalizes_anchor_and_keeps_transcript_exact(runtime):
    metadata = runtime.engine.execute("design_voice", {"name": "Reader", "gender": "female", "language": "en", "instruct": "whisper", "reference_text": "I have 123 apples", "normalize_text": True, "seed": 7, "generation": {"postprocess_output": True, "denoise": False}})
    assert metadata["source"] == "design" and metadata["synthetic"] is True
    assert metadata["ref_text"] == "I have one hundred twenty-three apples."
    options = runtime.model.calls[-1]
    assert options["text"] == "I have one hundred twenty-three apples" and options["normalize_text"] is False
    assert options["duration"] == 8 and options["generation_config"].postprocess_output is True
    manifest = json.loads((runtime.settings["voices_dir"] / "custom" / metadata["id"] / "manifest.json").read_text())
    assert manifest["controls"]["normalize_text"] and manifest["controls"]["seed"] == 7


@pytest.mark.parametrize("normalize", [False, True])
def test_large_design_script_selects_a_matching_anchor_and_remains_reusable(runtime, monkeypatch, normalize):
    original_normalize = runtime.normalizer.normalize_text
    normalizations = []
    def normalize_once(text, language):
        normalizations.append((text, language))
        return original_normalize(text, language)
    monkeypatch.setattr(runtime.normalizer, "normalize_text", normalize_once)
    script = "I have 123 apples in the garden. " * 250
    expected = original_normalize(script.strip(), "en") if normalize else script.strip()
    metadata = runtime.engine.execute("design_voice", {
        "name": "Long design script", "gender": "female", "language": "en",
        "instruct": "female, young adult", "reference_text": script,
        "normalize_text": normalize, "seed": 1234})
    preparation = metadata["reference_text_preparation"]
    options = runtime.model.calls[-1]
    selected = options["text"]
    assert preparation == {"source_characters": len(expected), "selected_characters": len(selected),
                           "automatic_selection": True}
    assert len(expected) > 6000 and 0 < len(selected) < len(expected)
    assert expected.startswith(selected)
    assert runtime.model._estimate_target_tokens(selected, None, None) <= 8 * 25
    assert options["duration"] == 8 and options["normalize_text"] is False
    assert options["generation_config"].num_step == 32
    assert metadata["ref_text"] == (selected if selected.endswith(".") else selected + ".")
    assert len(normalizations) == int(normalize)
    full_speech = "This longer saved-voice sentence must remain complete. " * 80
    start = len(runtime.model.calls)
    assert runtime.engine.synthesize(full_speech, "en", metadata["id"]).startswith(b"RIFF")
    assert "".join(call["text"] for call in runtime.model.calls[start:]).split() == full_speech.split()
    reloaded = OmniVoiceEngine(**runtime.settings)
    reloaded.load()
    assert metadata in reloaded.voices
    assert reloaded.synthesize("Saved designed voice works after restart.", "en", metadata["id"]).startswith(b"RIFF")


def test_custom_cache_corruption_fails_without_replacement(runtime):
    metadata = runtime.engine.execute("clone_voice", {"audio": wav_bytes(), "ref_text": "Reference", "name": "Voice", "gender": "male"})
    prompt = runtime.settings["voices_dir"] / "custom" / metadata["id"] / "prompt.pt"
    prompt.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="identity was preserved"):
        OmniVoiceEngine(**runtime.settings).load()
    assert prompt.read_bytes() == b"corrupt"


def test_custom_publish_failure_leaves_no_visible_profile_and_low_disk_is_reported(runtime, monkeypatch):
    ids = {voice["id"] for voice in runtime.engine.voices}
    def fail_publish(*args):
        raise OSError("Disk write failed")
    monkeypatch.setattr(engine_module.os, "rename", fail_publish)
    with pytest.raises(RuntimeError, match="could not be published"):
        runtime.engine.execute("clone_voice", {"audio": wav_bytes(), "ref_text": "Reference", "name": "Voice", "gender": "male"})
    assert {voice["id"] for voice in runtime.engine.voices} == ids
    assert not list((runtime.settings["voices_dir"] / "custom").iterdir())
    monkeypatch.setattr(engine_module.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=10))
    with pytest.raises(OSError) as error:
        runtime.engine._custom_capacity(11)
    assert error.value.errno == 28


@pytest.fixture
def custom_voice(runtime):
    return runtime.engine.execute("clone_voice", {"audio": wav_bytes(), "ref_text": "Reference", "name": "Deletion test", "gender": "female"})


@pytest.mark.parametrize("operation,payload", [
    ("clone_voice", {"ref_text": "Reference", "name": "Clone to delete", "gender": "male"}),
    ("design_voice", {"name": "Design to delete", "gender": "male", "language": "en", "instruct": "male, young adult"}),
])
def test_delete_custom_voice_removes_disk_cache_and_survives_reload(runtime, custom_voice, operation, payload):
    before = {path: path.read_bytes() for spec in VOICE_SPECS for path in (runtime.settings["voices_dir"] / spec["id"]).iterdir()}
    survivor = runtime.settings["voices_dir"] / "custom" / custom_voice["id"]
    before.update({path: path.read_bytes() for path in survivor.iterdir()})
    if operation == "clone_voice":
        payload = {**payload, "audio": wav_bytes()}
    voice = runtime.engine.execute(operation, payload)
    voice_id = voice["id"]
    root = runtime.settings["voices_dir"] / "custom"
    assert (root / voice_id).is_dir()
    assert runtime.engine.execute("delete_voice", {"voice": voice_id}) == {"id": voice_id, "deleted": True}
    assert not (root / voice_id).exists() and not (root / f".deleted-{voice_id}").exists()
    assert voice_id not in runtime.engine._custom_ids
    assert voice_id not in runtime.engine._prompts and voice_id not in runtime.engine._samples
    assert voice_id not in {item["id"] for item in runtime.engine.voices}
    with pytest.raises(VoiceNotFoundError):
        runtime.engine.execute("voice_sample", {"voice": voice_id})
    with pytest.raises(ValueError, match="Unknown voice"):
        runtime.engine.synthesize("Salom.", "uz", voice_id)
    with pytest.raises(VoiceNotFoundError):
        runtime.engine.execute("delete_voice", {"voice": voice_id})
    reloaded = OmniVoiceEngine(**runtime.settings)
    reloaded.load()
    ids = {item["id"] for item in reloaded.voices}
    assert voice_id not in ids and custom_voice["id"] in ids
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("voice_id", [spec["id"] for spec in VOICE_SPECS])
def test_delete_rejects_builtins_without_changing_them(runtime, voice_id):
    directory = runtime.settings["voices_dir"] / voice_id
    before = {path: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ProtectedVoiceError):
        runtime.engine.execute("delete_voice", {"voice": voice_id})
    assert voice_id in runtime.engine._prompts and voice_id in runtime.engine._samples
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("voice_id", [None, {}, "../female_1", "clone_" + "a" * 32, "clone_" + "a" * 32 + "/.."])
def test_delete_missing_and_malformed_voice_is_not_found(runtime, voice_id):
    ids = {item["id"] for item in runtime.engine.voices}
    with pytest.raises(VoiceNotFoundError):
        runtime.engine.execute("delete_voice", {"voice": voice_id})
    assert {item["id"] for item in runtime.engine.voices} == ids


def test_delete_releases_cached_identity_without_profile_count_or_storage_quota(runtime, custom_voice, monkeypatch):
    voice_id = custom_voice["id"]
    quota = 256 * 1024 * 1024
    monkeypatch.setattr(engine_module.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=2**40))
    runtime.engine._custom_capacity(quota)
    runtime.engine._custom_ids.update(f"reserved-{index}" for index in range(63))
    runtime.engine._custom_capacity()
    runtime.engine.execute("delete_voice", {"voice": voice_id})
    runtime.engine._custom_capacity(quota)
    replacement = runtime.engine.execute("clone_voice", {"audio": wav_bytes(), "ref_text": "Reference", "name": "Replacement", "gender": "female"})
    assert replacement["id"] in runtime.engine._custom_ids
    assert len(runtime.engine._custom_ids) == 64


def test_more_than_sixty_four_custom_voices_can_be_saved(runtime, monkeypatch):
    monkeypatch.setattr(engine_module.shutil, "disk_usage", lambda path: types.SimpleNamespace(free=2**40))
    encoded = wav_bytes()
    voices = [runtime.engine.execute("clone_voice", {
        "audio": encoded, "ref_text": "Reference.", "name": f"Saved voice {number}", "gender": "female"})
        for number in range(65)]
    assert len({voice["id"] for voice in voices}) == 65
    assert len(runtime.engine._custom_ids) == 65
    runtime.engine._custom_capacity(256 * 1024 * 1024 + 1)


def test_delete_rename_failure_preserves_usable_profile(runtime, custom_voice, monkeypatch):
    voice_id = custom_voice["id"]
    directory = runtime.settings["voices_dir"] / "custom" / voice_id
    before = {path: path.read_bytes() for path in directory.iterdir()}
    prompt = runtime.engine._prompts[voice_id]

    def fail_rename(*args):
        raise OSError("Rename denied")

    monkeypatch.setattr(engine_module.os, "rename", fail_rename)
    with pytest.raises(VoiceDeletionError, match="committed"):
        runtime.engine.execute("delete_voice", {"voice": voice_id})
    assert voice_id in runtime.engine._custom_ids
    assert runtime.engine._prompts[voice_id] is prompt
    assert all(path.read_bytes() == contents for path, contents in before.items())
    runtime.engine.synthesize("Still usable.", "en", voice_id)
    assert runtime.model.calls[-1]["voice_clone_prompt"] is prompt
    assert not (directory.parent / f".deleted-{voice_id}").exists()


@pytest.mark.parametrize("retry", ["delete", "startup", "create"])
def test_partial_delete_stays_inactive_and_can_retry(runtime, custom_voice, monkeypatch, retry, caplog):
    voice_id = custom_voice["id"]
    root = runtime.settings["voices_dir"] / "custom"
    tombstone = root / f".deleted-{voice_id}"
    unlink = Path.unlink

    def fail_second_unlink(path, *args, **options):
        if path.parent == tombstone and path.name == "reference.wav":
            raise OSError("Cleanup denied")
        return unlink(path, *args, **options)

    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "unlink", fail_second_unlink)
        with pytest.raises(VoiceDeletionError, match="did not finish"):
            runtime.engine.execute("delete_voice", {"voice": voice_id})
        assert not (root / voice_id).exists() and tombstone.is_dir()
        assert not (tombstone / "prompt.pt").exists()
        assert voice_id not in runtime.engine._custom_ids
        assert voice_id not in runtime.engine._prompts and voice_id not in runtime.engine._samples
        assert voice_id not in {item["id"] for item in runtime.engine.voices}
        with pytest.raises(VoiceNotFoundError):
            runtime.engine.execute("voice_sample", {"voice": voice_id})
        assert (tombstone / "reference.wav").stat().st_size > 0
        failed_cleanup_reload = OmniVoiceEngine(**runtime.settings)
        failed_cleanup_reload.load()
        assert voice_id not in {item["id"] for item in failed_cleanup_reload.voices}
        assert "Pending voice deletion cleanup did not finish" in caplog.text
    if retry == "delete":
        assert runtime.engine.execute("delete_voice", {"voice": voice_id}) == {"id": voice_id, "deleted": True}
    elif retry == "startup":
        reloaded = OmniVoiceEngine(**runtime.settings)
        reloaded.load()
        assert voice_id not in {item["id"] for item in reloaded.voices}
    else:
        created = runtime.engine.execute("clone_voice", {"audio": wav_bytes(), "ref_text": "Reference", "name": "After cleanup", "gender": "male"})
        assert created["id"] in runtime.engine._custom_ids
    assert not tombstone.exists()
    with pytest.raises(VoiceNotFoundError):
        runtime.engine.execute("delete_voice", {"voice": voice_id})


@pytest.mark.parametrize("unsafe", ["extra-file", "extra-directory", "missing-file", "root-link", "directory-link", "file-link", "resolved-escape"])
def test_delete_rejects_unsafe_storage_before_any_removal(runtime, custom_voice, monkeypatch, unsafe):
    voice_id = custom_voice["id"]
    root = runtime.settings["voices_dir"] / "custom"
    directory = root / voice_id
    if unsafe == "extra-file":
        (directory / "unrelated.txt").write_text("keep me")
    elif unsafe == "extra-directory":
        (directory / "unrelated").mkdir()
    elif unsafe == "missing-file":
        (directory / "prompt.pt").unlink()
    elif unsafe.endswith("-link"):
        target = {"root-link": root, "directory-link": directory, "file-link": directory / "reference.wav"}[unsafe]
        is_link = runtime.engine._is_link
        monkeypatch.setattr(runtime.engine, "_is_link", lambda path: path == target or is_link(path))
    else:
        resolve = Path.resolve
        monkeypatch.setattr(Path, "resolve", lambda path, *args, **options: runtime.tmp / "outside" if path == directory else resolve(path, *args, **options))
    before = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    with pytest.raises(VoiceDeletionError):
        runtime.engine.execute("delete_voice", {"voice": voice_id})
    assert directory.is_dir() and not (root / f".deleted-{voice_id}").exists()
    assert voice_id in runtime.engine._custom_ids and voice_id in runtime.engine._prompts
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_startup_cleanup_preserves_unrelated_hidden_staging_files(runtime, custom_voice):
    root = runtime.settings["voices_dir"] / "custom"
    stage = root / ".deleted-clone_not-a-real-id"
    stage.mkdir()
    sentinel = stage / "reference.wav"
    sentinel.write_bytes(b"unrelated staging data")
    reloaded = OmniVoiceEngine(**runtime.settings)
    reloaded.load()
    assert custom_voice["id"] in {item["id"] for item in reloaded.voices}
    assert sentinel.read_bytes() == b"unrelated staging data"


def test_batch_filenames_use_keys_and_keep_numeric_fallback(runtime):
    items = [
        {"key": "assistant.welcome", "text": "Davom etish uchun tilni tanlang.", "language": "uz", "voice": "female_1"},
        {"text": "Hello.", "mode": "auto"},
        {"key": "assistant.next", "text": "Continue.", "voice": "male_1"},
    ]
    archive = runtime.engine.execute("batch", {"items": items})
    with zipfile.ZipFile(io.BytesIO(archive)) as result:
        assert result.namelist() == ["assistant.welcome.wav", "002.wav", "assistant.next.wav", "manifest.json"]
        manifest = json.loads(result.read("manifest.json"))["items"]
        assert manifest[0]["file"] == "assistant.welcome.wav"
        assert manifest[0]["options"]["key"] == "assistant.welcome"
        assert manifest[1]["file"] == "002.wav"
        with wave.open(io.BytesIO(result.read("assistant.welcome.wav"))) as audio:
            assert audio.getframerate() == 24000 and audio.getnframes() > 0


@pytest.mark.parametrize("keys", [
    ("assistant.welcome", "assistant.welcome"),
    ("assistant.Welcome", "Assistant.welcome"),
    ("002", None),
    ("../escape", None),
    ("CON", None),
])
def test_invalid_batch_filenames_are_rejected_before_generation(runtime, keys):
    items = [{"text": "Hello.", "voice": "female_1", "key": key} for key in keys]
    calls = len(runtime.model.calls)
    with pytest.raises(ValueError):
        runtime.engine.execute("batch", {"items": items})
    assert len(runtime.model.calls) == calls


def test_batch_archive_and_failed_batch_do_not_mutate_profiles(runtime):
    archive = runtime.engine.execute("batch", {"items": [{"text": "Hello", "language": "en", "voice": "female_1"}, {"text": "Salom", "mode": "auto", "seed": 1}]})
    with zipfile.ZipFile(io.BytesIO(archive)) as result:
        assert result.namelist() == ["001.wav", "002.wav", "manifest.json"]
        assert json.loads(result.read("manifest.json"))["items"][1]["options"]["seed"] == 1
        with wave.open(io.BytesIO(result.read("001.wav"))) as audio:
            assert audio.getsampwidth() == 2
    voices = list(runtime.engine.voices)
    with pytest.raises(RuntimeError):
        runtime.engine.execute("batch", {"items": [{"text": "Hello"}, {"text": "fail"}]})
    assert runtime.engine.voices == voices
    archive = runtime.engine.execute("batch", {"items": [{"text": "x"}] * 5})
    with zipfile.ZipFile(io.BytesIO(archive)) as result:
        assert len([name for name in result.namelist() if name.endswith(".wav")]) == 5


@pytest.mark.parametrize("samples", [np.zeros(24), np.array([np.nan]), np.zeros((2, 20))])
def test_invalid_model_audio_is_operational_failure(runtime, samples):
    runtime.model.result = [samples]
    with pytest.raises(RuntimeError):
        runtime.engine.synthesize("Salom", "uz", "male_1")


def test_generated_audio_can_exceed_the_old_ten_minute_output_limit(runtime):
    samples = np.full(24000 * 601, 0.25, dtype=np.float32)
    runtime.model.result = [samples]
    result = runtime.engine.synthesize("Salom.", "uz", "female_1")
    with wave.open(io.BytesIO(result), "rb") as audio:
        assert audio.getnframes() == len(samples)
        assert audio.getnframes() / audio.getframerate() == 601


def test_backend_value_error_is_operational_and_invalid_instruction_is_not(runtime, monkeypatch):
    def backend_error(**options):
        raise ValueError("Internal decoder failure")
    monkeypatch.setattr(runtime.model, "generate", backend_error)
    with pytest.raises(RuntimeError, match="inference failed"):
        runtime.engine.synthesize("Salom", "uz", "male_1")
    with pytest.raises(ValueError):
        runtime.engine.execute("speech", {"text": "Salom", "instruct": "unsupported attribute"})


def test_failed_tensorrt_context_keeps_restart_marker_across_processes(runtime, monkeypatch):
    import pickle

    runtime.engine._trt_backend = types.SimpleNamespace(failed=True)

    def fail(**options):
        raise RuntimeError("Private model failure")

    monkeypatch.setattr(runtime.model, "generate", fail)
    with pytest.raises(engine_module.WorkerRestartRequired) as caught:
        runtime.engine.synthesize("Salom", "uz", "male_1")
    restored = pickle.loads(pickle.dumps(caught.value))
    assert restored.recoverable and str(restored) == "OmniVoice inference failed."


def assert_generated_working_sets_fit(engine, calls):
    for call in calls:
        engine._preflight(dict(call))
        assert call["generation_config"].num_step == 32


def test_reference_ratio_explosion_is_generated_in_bounded_segments_without_text_loss(runtime):
    engine, model = runtime.engine, runtime.model
    prompt = engine._prompts["male_1"]
    prompt.ref_text, prompt.ref_audio_tokens = "a.", TokenShape(250)
    assert model._estimate_target_tokens("a" * 1000, "a.", 250) == 166666
    calls = len(model.calls)
    assert engine.synthesize("a" * 1000, "uz", "male_1").startswith(b"RIFF")
    generated = model.calls[calls:]
    assert len(generated) > 1
    assert "".join(call["text"] for call in generated) == "a" * 1000
    assert_generated_working_sets_fit(engine, generated)


def test_actual_upstream_unsplittable_long_sentence_is_split_without_text_loss(runtime):
    text = "a" * 2400
    assert runtime.upstream.chunk_text_punctuation(text, chunk_len=50, min_chunk_len=3) == [text]
    calls = len(runtime.model.calls)
    assert runtime.engine.synthesize(text, "uz", "female_1").startswith(b"RIFF")
    generated = runtime.model.calls[calls:]
    assert len(generated) > 1
    assert "".join(call["text"] for call in generated) == text
    assert_generated_working_sets_fit(runtime.engine, generated)


@pytest.mark.parametrize("mode", ["auto", "design"])
def test_segmented_new_voice_uses_a_small_first_anchor_and_preserves_all_words(runtime, monkeypatch, mode):
    original = runtime.model.generate
    prompt_start = len(runtime.prompts)
    def generate(**options):
        original(**options)
        task = runtime.model._preprocess_all(
            **{key: options[key] for key in ("text", "language", "voice_clone_prompt", "instruct", "speed", "duration", "normalize_text") if key in options},
            preprocess_prompt=options["generation_config"].preprocess_prompt)
        assert task.target_lens[0] <= 15 * 25
        frames = max(1, round(task.target_lens[0] / 25 * runtime.model.sampling_rate))
        return [(np.sin(np.arange(frames) * .05) * .25).astype(np.float32)]
    monkeypatch.setattr(runtime.model, "generate", generate)
    text = "Ordinary words in a long recording " * 300
    settings = {"text": text, "mode": mode, "language": "en", "seed": 1234}
    if mode == "design":
        settings["instruct"] = "female"
    start = len(runtime.model.calls)
    assert runtime.engine.execute("speech", settings).startswith(b"RIFF")
    calls = runtime.model.calls[start:]
    assert len(calls) > 1
    assert "".join(call["text"] for call in calls).split() == text.split()
    for prompt in runtime.prompts[prompt_start:]:
        samples, rate = prompt["ref_audio"]
        assert 0 < len(samples) / rate <= 15
    assert len(runtime.prompts) == prompt_start + 1
    assert_generated_working_sets_fit(runtime.engine, calls)


@pytest.mark.parametrize("mode", ["preset", "auto", "design"])
def test_punctuated_1000_character_long_form_is_preserved(runtime, mode):
    text = "This is an ordinary sentence. " * 33
    assert 900 <= len(text) <= 1000
    settings = {"text": text, "language": "en", "mode": mode}
    if mode == "preset":
        settings["voice"] = "female_1"
    elif mode == "design":
        settings["instruct"] = "female"
    calls = len(runtime.model.calls)
    result = runtime.engine.execute("speech", settings)
    assert result.startswith(b"RIFF") and len(runtime.model.calls) == calls + 1


def test_exact_normalized_target_is_segmented_once_without_losing_normalized_text(runtime, monkeypatch):
    monkeypatch.setattr(runtime.normalizer, "normalize_text", lambda text, language: "a" * 1000)
    calls = len(runtime.model.calls)
    assert runtime.engine.execute("speech", {"text": "123", "language": "en", "voice": "female_1", "normalize_text": True}).startswith(b"RIFF")
    generated = runtime.model.calls[calls:]
    assert len(generated) > 1
    assert "".join(call["text"] for call in generated) == "a" * 1000
    assert all(call["normalize_text"] is False for call in generated)
    assert_generated_working_sets_fit(runtime.engine, generated)


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r", " \n\n\t "])
def test_multiline_speech_preserves_word_boundaries_before_native_estimation_and_tokenization(runtime, separator):
    # The pinned upstream combiner deletes CR/LF. Feeding it the original
    # multiline script would merge words even though the API accepts newlines.
    words = ["O‘zbekcha", "g‘alaba", "[laughter]"]
    source = separator.join(words)
    expected = " ".join(words)
    if separator in ("\n", "\r\n", "\r"):
        assert runtime.upstream._combine_text(source) != expected
    runtime.engine.execute("speech", {"text": source, "voice": "female_1", "language": "uz", "seed": 42})
    actual = runtime.model.calls[-1]
    assert actual["text"] == expected
    assert runtime.upstream._combine_text(actual["text"]) == expected
    prepared = runtime.model._preprocess_all(
        text=actual["text"], voice_clone_prompt=actual["voice_clone_prompt"], language="uz")
    prompt = actual["voice_clone_prompt"]
    assert prepared.target_lens[0] == runtime.model._estimate_target_tokens(
        expected, prompt.ref_text, prompt.ref_audio_tokens.size(-1))
    assert actual["generation_config"].num_step == 32


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("saved", [False, True])
def test_multiline_reference_transcripts_keep_words_separate_in_new_prompts(runtime, automatic, saved):
    runtime.model.transcript = "Bu\r\nmening\novozim."
    payload = {"audio": wav_bytes(), "reference_language": "uz", "name": "Mening  ovozim", "gender": "male"}
    if not automatic:
        payload["ref_text"] = runtime.model.transcript
    if not saved:
        payload.update(text="Birinchi\nikkinchi", language="uz")
    result = runtime.engine.execute("clone_voice" if saved else "clone_speech", payload)
    assert runtime.prompts[-1]["ref_text"] == "Bu mening ovozim."
    if saved:
        assert result["name"] == "Mening  ovozim", "display names are not speech scripts"
        assert result["ref_text"] == "Bu mening ovozim."
        reloaded = OmniVoiceEngine(**runtime.settings)
        reloaded.load()
        assert reloaded._prompts[result["id"]].ref_text == "Bu mening ovozim."
    else:
        assert runtime.model.calls[-1]["text"] == "Birinchi ikkinchi"


def test_new_design_anchor_keeps_linebreak_word_boundaries_and_single_line_scripts_are_unchanged(runtime):
    result = runtime.engine.execute("design_voice", {
        "name": "Designed  voice", "gender": "female", "language": "en", "instruct": "female",
        "reference_text": "First\r\nsecond\nthird."})
    assert result["name"] == "Designed  voice"
    assert runtime.model.calls[-1]["text"] == "First second third."
    assert runtime.prompts[-1]["ref_text"] == "First second third."
    single_line = "O‘zbekcha  g‘alaba\t[laughter]"
    runtime.engine.synthesize(single_line, "uz", "female_1")
    assert runtime.model.calls[-1]["text"] == single_line


def test_legacy_multiline_prompt_uses_request_copy_without_mutating_saved_profile(runtime):
    prompt = runtime.engine._prompts["female_1"]
    prompt.ref_text = "Bu\r\nmening\novozim."
    folder = runtime.settings["voices_dir"] / "female_1"
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir() if path.is_file()}
    runtime.engine.synthesize("Davom eting.", "uz", "female_1")
    actual = runtime.model.calls[-1]["voice_clone_prompt"]
    assert actual is not prompt
    assert actual.ref_text == "Bu mening ovozim."
    assert actual.ref_audio_tokens is prompt.ref_audio_tokens and actual.ref_rms == prompt.ref_rms
    assert runtime.engine._prompts["female_1"] is prompt
    assert prompt.ref_text == "Bu\r\nmening\novozim."
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir() if path.is_file()}


def test_legacy_queued_snapshot_reference_keeps_immutable_bytes_and_cached_transcript(runtime):
    path = runtime.tmp / "legacy-snapshot.pt"
    runtime.engine._prompt_class("Old\nreference.").save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = "clone_" + "a" * 32
    snapshot = {"voice_id": identity, "sha256": digest, "gender": "female", "name": "Legacy queued voice"}
    runtime.engine.execute_pinned_speech({"text": "New text.", "voice": identity, "language": "en"}, path, snapshot)
    actual = runtime.model.calls[-1]["voice_clone_prompt"]
    assert actual.ref_text == "Old reference."
    assert runtime.engine._pinned_prompts[digest].ref_text == "Old\nreference."
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert identity not in runtime.engine._prompts
    assert all(voice["id"] != identity for voice in runtime.engine.voices)


def test_reference_and_combined_sequence_budgets_checked_even_for_fixed_duration(runtime):
    prompt = runtime.engine._prompts["female_1"]
    prompt.ref_audio_tokens = TokenShape(251)
    calls = len(runtime.model.calls)
    assert runtime.engine.execute("speech", {"text": "Hello", "voice": "female_1", "duration": 1}).startswith(b"RIFF")
    assert len(runtime.model.calls) == calls + 1
    prompt.ref_audio_tokens = TokenShape(200)
    runtime.model.text_tokenizer.multiplier = 100
    with pytest.raises(ValueError, match="working memory"):
        runtime.engine.execute("speech", {"text": "Hello", "voice": "female_1", "duration": 1})
    assert len(runtime.model.calls) == calls + 1


def test_direct_path_over_30_seconds_enables_bounded_upstream_chunks_with_high_chunk_threshold(runtime):
    calls = len(runtime.model.calls)
    text = "An ordinary sentence. " * 24
    assert runtime.engine.execute("speech", {"text": text, "voice": "female_1", "generation": {"audio_chunk_threshold": 60}}).startswith(b"RIFF")
    generated = runtime.model.calls[calls:]
    assert generated and all(call["generation_config"].audio_chunk_threshold <= 30 for call in generated)
    assert "".join(call["text"] for call in generated).split() == text.split()
    assert_generated_working_sets_fit(runtime.engine, generated)


def test_actual_chunk_total_may_exceed_ten_minutes_when_each_model_working_set_fits(runtime):
    prompt = runtime.engine._prompts["male_1"]
    prompt.ref_text, prompt.ref_audio_tokens = "a.", TokenShape(250)
    text = "aa." * 36
    assert runtime.model._estimate_target_tokens(text, "a.", 250) == 15000
    chunks = runtime.upstream.chunk_text_punctuation(text, chunk_len=0, min_chunk_len=3)
    assert len(chunks) == 36
    frames = sum(runtime.model._estimate_target_tokens(chunk, "a.", 250) for chunk in chunks)
    assert frames / 25 < 600 < frames / 25 + 2
    calls = len(runtime.model.calls)
    assert runtime.engine.execute("speech", {"text": text, "voice": "male_1", "generation": {"audio_chunk_duration": 3, "pad_duration": 1}}).startswith(b"RIFF")
    assert len(runtime.model.calls) == calls + 1
    assert_generated_working_sets_fit(runtime.engine, runtime.model.calls[calls:])


def queued_voice_snapshot(runtime, voice_id="female_1"):
    descriptor = runtime.engine.profile_snapshot_descriptor(voice_id)
    directory = runtime.tmp / "queued-snapshots"
    directory.mkdir(exist_ok=True)
    path = directory / f"{descriptor['sha256']}.pt"
    path.write_bytes(Path(descriptor["path"]).read_bytes())
    return path, {"voice_id": voice_id, "sha256": descriptor["sha256"],
                  "gender": descriptor["gender"], "name": descriptor["name"]}


@pytest.mark.parametrize("selected", ["builtin", "custom"])
def test_snapshot_descriptor_uses_verified_profile_identity(runtime, custom_voice, selected):
    voice_id = "female_1" if selected == "builtin" else custom_voice["id"]
    descriptor = runtime.engine.profile_snapshot_descriptor(voice_id)
    expected = runtime.settings["voices_dir"] / ("" if selected == "builtin" else "custom") / voice_id / "prompt.pt"
    metadata = next(item for item in runtime.engine.voices if item["id"] == voice_id)
    assert Path(descriptor["path"]) == expected
    assert descriptor["sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert (descriptor["name"], descriptor["gender"]) == (metadata["name"], metadata["gender"])
    expected.write_bytes(b"corrupted persisted identity")
    with pytest.raises(RuntimeError, match="checksum"):
        runtime.engine.profile_snapshot_descriptor(voice_id)


def test_unknown_or_unready_voice_has_no_snapshot_descriptor(runtime):
    with pytest.raises(VoiceNotFoundError):
        runtime.engine.profile_snapshot_descriptor("../not-a-voice")
    not_loaded = OmniVoiceEngine(**runtime.settings)
    with pytest.raises(VoiceNotFoundError):
        not_loaded.profile_snapshot_descriptor("female_1")


def test_pinned_custom_job_survives_public_deletion_and_worker_restart(runtime, custom_voice):
    voice_id = custom_voice["id"]
    path, snapshot = queued_voice_snapshot(runtime, voice_id)
    transcript = runtime.engine._prompts[voice_id].ref_text
    runtime.engine.execute("delete_voice", {"voice": voice_id})
    assert not (runtime.settings["voices_dir"] / "custom" / voice_id).exists()
    for _ in range(2):
        worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
        worker.load()
        before_voices, before_prompts = worker.voices, dict(worker._prompts)
        output = worker.execute_pinned_speech(
            {"text": "Still available.", "voice": voice_id, "language": "en", "seed": 1234},
            path, snapshot)
        with wave.open(io.BytesIO(output)) as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
        assert runtime.models[-1].calls[-1]["voice_clone_prompt"].ref_text == transcript
        assert runtime.models[-1].calls[-1]["generation_config"].num_step == 32
        assert runtime.seeds[-1] == 1234
        assert worker.voices is before_voices and worker._prompts == before_prompts
        assert voice_id not in worker._prompts and voice_id not in worker._custom_ids
    assert voice_id not in {item["id"] for item in runtime.engine.voices}


@pytest.mark.parametrize("voice_kind", ["existing", "deleted_custom"])
def test_failed_pinned_generation_restores_worker_catalog_and_prompt(runtime, custom_voice, voice_kind):
    voice_id = "female_1" if voice_kind == "existing" else custom_voice["id"]
    path, snapshot = queued_voice_snapshot(runtime, voice_id)
    if voice_kind == "deleted_custom":
        runtime.engine.execute("delete_voice", {"voice": voice_id})
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    old_voices, old_prompts = worker.voices, dict(worker._prompts)
    with pytest.raises(RuntimeError, match="inference failed"):
        worker.execute_pinned_speech({"text": "fail", "voice": voice_id}, path, snapshot)
    assert worker.voices is old_voices and worker._prompts == old_prompts
    worker.synthesize("Next request.", "en", "female_1")
    assert runtime.models[-1].calls[-1]["voice_clone_prompt"] is old_prompts["female_1"]


@pytest.mark.parametrize("generation", [{}, {
    "guidance_scale": 2.5, "denoise": False, "postprocess_output": False,
    "audio_chunk_duration": 12.0, "audio_chunk_threshold": 24.0,
}])
def test_pinned_jobs_keep_quality_defaults_and_explicit_controls(runtime, generation):
    path, snapshot = queued_voice_snapshot(runtime)
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    worker.execute_pinned_speech(
        {"text": "Hello again.", "voice": "female_1", "language": "en",
         "gender": "female", "instruct": "female, whisper", "speed": 1.1,
         "seed": 42, "generation": generation}, path, snapshot)
    call = runtime.models[-1].calls[-1]
    config = call["generation_config"]
    assert config.num_step == 32
    assert config.preprocess_prompt and config.position_temperature == 5.0
    assert config.class_temperature == 0 and config.layer_penalty_factor == 5.0
    assert config.t_shift == 0.1 and config.pad_duration == config.fade_duration == 0.1
    assert config.guidance_scale == generation.get("guidance_scale", 2.0)
    assert config.denoise == generation.get("denoise", True)
    assert config.postprocess_output == generation.get("postprocess_output", True)
    assert config.audio_chunk_duration == generation.get("audio_chunk_duration", 15.0)
    assert config.audio_chunk_threshold == generation.get("audio_chunk_threshold", 30.0)
    assert call["language"] == "en" and call["speed"] == 1.1
    assert call["instruct"] == "female, whisper" and runtime.seeds[-1] == 42


def test_pinned_prompt_cache_is_lru_bounded_and_rechecks_evicted_files(runtime, monkeypatch):
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    prompt_class = worker._prompt_class
    original_load = prompt_class.load
    loads = []
    def counted(cls, path):
        loads.append(path)
        return original_load(path)
    monkeypatch.setattr(prompt_class, "load", classmethod(counted))
    directory = runtime.tmp / "queued-snapshots"
    directory.mkdir()
    snapshots = []
    for number in range(65):
        source = directory / f"{number}.pt"
        prompt_class(f"Reference number {number}.", frames=25).save(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        snapshots.append((source, {"voice_id": f"clone_{number:032x}", "sha256": digest,
                                   "gender": "female", "name": f"Voice {number}"}))
    def generate(number):
        source, snapshot = snapshots[number]
        return worker.execute_pinned_speech({"text": "Hi.", "voice": snapshot["voice_id"]}, source, snapshot)
    for number in range(64):
        generate(number)
    assert len(worker._pinned_prompts) == len(loads) == 64
    generate(0)  # Reuse promotes this prompt, rather than re-reading its file.
    generate(64)
    assert len(loads) == 65 and len(worker._pinned_prompts) == 64
    assert snapshots[0][1]["sha256"] in worker._pinned_prompts
    assert snapshots[1][1]["sha256"] not in worker._pinned_prompts
    snapshots[1][0].write_bytes(b"corrupted while no longer cached")
    calls = len(runtime.models[-1].calls)
    with pytest.raises(ValueError, match="checksum"):
        generate(1)
    assert len(loads) == 65 and len(runtime.models[-1].calls) == calls
    assert len(worker._pinned_prompts) == 64


def test_pinned_snapshot_checksum_failure_never_binds_voice_or_calls_model(runtime):
    path, snapshot = queued_voice_snapshot(runtime)
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    previous_voices, previous_prompts = worker.voices, dict(worker._prompts)
    path.write_bytes(b"corrupted queued prompt")
    with pytest.raises(ValueError, match="checksum"):
        worker.execute_pinned_speech({"text": "Hello.", "voice": "female_1"}, path, snapshot)
    assert worker.voices is previous_voices and worker._prompts == previous_prompts
    assert not worker._pinned_prompts and not runtime.models[-1].calls


@pytest.mark.parametrize("change", [
    {"voice_id": "male_1"}, {"gender": "unknown"}, {"sha256": "../bad"}, {"name": None},
])
def test_invalid_pinned_metadata_cannot_mutate_worker_state(runtime, change):
    path, snapshot = queued_voice_snapshot(runtime)
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    previous_voices, previous_prompts = worker.voices, dict(worker._prompts)
    with pytest.raises(ValueError):
        worker.execute_pinned_speech({"text": "Hello.", "voice": "female_1"}, path, dict(snapshot, **change))
    assert worker.voices is previous_voices and worker._prompts == previous_prompts
    assert not runtime.models[-1].calls


def test_missing_pinned_name_cannot_leave_temporary_prompt_binding(runtime):
    path, snapshot = queued_voice_snapshot(runtime)
    del snapshot["name"]
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    previous_voices, previous_prompts = worker.voices, dict(worker._prompts)
    with pytest.raises(ValueError):
        worker.execute_pinned_speech({"text": "Hello.", "voice": "female_1"}, path, snapshot)
    assert worker.voices is previous_voices and worker._prompts == previous_prompts


def test_readonly_worker_startup_does_not_load_custom_catalog_or_cleanup_profiles(runtime, custom_voice, monkeypatch):
    root = runtime.settings["voices_dir"]
    directory = root / "custom" / custom_voice["id"]
    tombstone = directory.parent / f".deleted-{custom_voice['id']}"
    directory.rename(tombstone)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("A synthesis worker attempted profile mutation or custom catalog loading")
    monkeypatch.setattr(OmniVoiceEngine, "_cleanup_deleted_profiles", forbidden)
    monkeypatch.setattr(OmniVoiceEngine, "_load_custom_profiles", forbidden)
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    assert {item["id"] for item in worker.voices} == {item["id"] for item in VOICE_SPECS}
    assert not worker._custom_ids and not runtime.models[-1].calls
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert tombstone.is_dir()


@pytest.mark.parametrize("existing_directory", [False, True])
def test_readonly_worker_never_initializes_missing_voice_assets(runtime, existing_directory):
    directory = runtime.tmp / "uninitialized-worker-voices"
    if existing_directory:
        directory.mkdir()
    worker = OmniVoiceEngine(**{**runtime.settings, "voices_dir": directory}, profile_writer=False)
    with pytest.raises(RuntimeError, match="initialize voices|cannot create"):
        worker.load()
    assert not worker._ready
    assert not directory.exists() or list(directory.iterdir()) == []


@pytest.mark.parametrize("operation", ["clone_voice", "design_voice", "delete_voice"])
def test_readonly_worker_rejects_published_profile_mutations_before_work(runtime, operation):
    root = runtime.settings["voices_dir"]
    worker = OmniVoiceEngine(**runtime.settings, profile_writer=False)
    worker.load()
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(RuntimeError, match="cannot modify"):
        worker.execute(operation, {"voice": "female_1", "audio": wav_bytes(),
                                   "name": "Forbidden", "gender": "female", "ref_text": "Reference."})
    assert not runtime.models[-1].calls and not runtime.models[-1].asr_loads
    assert {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
