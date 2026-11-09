"""CPU tests for artifact identity, routing and restoration; no CUDA claims."""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import threading
from types import SimpleNamespace
from types import MethodType, ModuleType
import sys

import pytest

import trt_backend as module


def manifest():
    return {"schema_version": 1, "kind": "unpadded_cfg_shared_linear", "batch": 2,
            "hidden": 1024, "num_audio_codebook": 8, "audio_vocab_size": 1025,
            "steps": 32, "min_length": 2, "min_target": 2, "max_length": 1024, "max_target": 512,
            "model_id": module.MODEL_ID, "model_revision": module.MODEL_REVISION,
            "versions": dict(module.VERSIONS), "compute_capability": [12, 0],
            "input_names": list(module.INPUT_NAMES), "causal": False, "guidance_scale": 2.0,
            "plan_sha256": "a" * 64}


@pytest.mark.parametrize("change", [{"model_revision": "another"}, {"plan_sha256": "missing"},
                                      {"versions": {}}, {"steps": True}, {"causal": True},
                                      {"max_target": 2048}])
def test_manifest_rejects_incompatible_artifacts_without_loading_cuda(tmp_path, change):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({**manifest(), **change}), encoding="utf-8")
    with pytest.raises(module.TensorRTCompatibilityError):
        module._load_manifest(path, module.MODEL_ID, module.MODEL_REVISION)


def test_checkpoint_hashes_cover_actual_weights_and_detect_replacement(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_bytes(b"configuration")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first weights")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second weights")
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    assert module._verify_model_files(root, files) == files
    with pytest.raises(module.TensorRTCompatibilityError, match="every checkpoint"):
        module._verify_model_files(root, {"config.json": files["config.json"]})
    (root / "model-00002-of-00002.safetensors").write_bytes(b"changed weights")
    with pytest.raises(module.TensorRTCompatibilityError, match="SHA256 differs"):
        module._verify_model_files(root, files)


def test_distribution_version_pin_includes_tensorrt_postrelease(monkeypatch):
    pins = {"torch": module.VERSIONS["torch"], "tensorrt-cu12": module.VERSIONS["tensorrt"],
            "transformers": module.VERSIONS["transformers"], "omnivoice": module.VERSIONS["omnivoice"]}
    monkeypatch.setattr(module.metadata, "version", pins.__getitem__)
    torch = SimpleNamespace(__version__=pins["torch"], version=SimpleNamespace(cuda="12.8"))
    assert module._validate_versions(torch) == module.VERSIONS
    pins["tensorrt-cu12"] = "10.14.1.48.post2"
    with pytest.raises(module.TensorRTCompatibilityError, match="exact validated"):
        module._validate_versions(torch)


@pytest.mark.parametrize("bad_artifact", ["plan", "checkpoint"])
def test_install_checks_actual_artifact_hashes_before_tensorrt_import_or_model_patch(tmp_path, monkeypatch, bad_artifact):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_bytes(b"config")
    (root / "model.safetensors").write_bytes(b"weights")
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
    plan = tmp_path / "forward.plan"
    plan.write_bytes(b"serialized plan")
    spec = {**manifest(), "model_files": files, "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}
    if bad_artifact == "plan":
        plan.write_bytes(b"replaced plan")
    else:
        (root / "model.safetensors").write_bytes(b"replaced weights")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(spec), encoding="utf-8")
    torch = ModuleType("torch")
    torch.float16 = object()
    torch.cuda = SimpleNamespace(get_device_capability=lambda unused: (12, 0))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    helpers = ModuleType("omnivoice.models.omnivoice")
    helpers._combine_text = helpers._tokenize_with_nonverbal_tags = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setattr(module, "_validate_versions", lambda unused: dict(module.VERSIONS))
    model = SimpleNamespace(_fi_runner=object(), dtype=torch.float16, device="cuda:0",
                            llm=SimpleNamespace(config=SimpleNamespace(model_type="qwen3", rope_parameters={"rope_type": "default"}, hidden_size=1024),
                                                has_sliding_layers=False),
                            config=SimpleNamespace(num_audio_codebook=8, audio_vocab_size=1025, audio_mask_id=1024),
                            forward=object())
    native = MethodType(lambda self, task, config: [], model)
    model._generate_iterative = native
    original = model.forward
    with pytest.raises(module.TensorRTCompatibilityError, match="SHA256"):
        module.install(model, plan, manifest_path, native_iterative=native,
                       model_id=module.MODEL_ID, model_revision=module.MODEL_REVISION, model_path=root)
    assert model.forward is original and model._generate_iterative is native and not hasattr(model, "_trt_backend")


@pytest.mark.parametrize("unsafe", ["../outside", "/outside", "directory/../outside", "directory\\file", "C:/outside"])
def test_checkpoint_manifest_cannot_escape_model_directory(tmp_path, unsafe):
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()}
    files[unsafe] = "a" * 64
    with pytest.raises(module.TensorRTCompatibilityError):
        module._verify_model_files(tmp_path, files)


def routing_backend():
    backend = object.__new__(module.TensorRTBackend)
    backend.manifest = manifest()
    backend.profiles = [{"index": 0, "c": (2, 640, 1024), "u": (2, 320, 512)}]
    backend.lock = threading.RLock()
    backend.state, backend.failed, backend.active, backend.forwards = None, False, False, 0
    backend.counts, backend.fallback_reasons = Counter(), Counter()
    calls = []

    @contextmanager
    def cpu_device(device):
        assert device == "cpu"
        yield

    backend.torch = SimpleNamespace(device=cpu_device)

    class Tokens:
        def __init__(self, count):
            self.count = count

        def size(self, dimension):
            assert dimension == -1
            return self.count

    def tokenizer(text, **kwargs):
        calls.append(("style", text))
        return SimpleNamespace(input_ids=Tokens(11))

    def tokenize(text, tokenizer):
        calls.append(("text", text))
        return Tokens(17)

    backend.model = SimpleNamespace(text_tokenizer=tokenizer, forward=object())
    backend.combine_text = lambda ref_text, text: f"{ref_text}:{text}"
    backend.tokenize_text = tokenize
    backend.fallback_iterative = lambda task, config: calls.append(("fallback", task, config)) or ["fi tokens"]

    def prepare(c_len, t_len, profile):
        calls.append(("prepare", c_len, t_len, profile))
        backend.forwards = 0
        backend.state = {"finite": True}

    def forward(unused_model):
        calls.append(("forward",))
        backend.forwards += 1

    def native(task, config):
        calls.append(("native", task, config))
        for _ in range(config.num_step):
            backend.model.forward()
        return ["native tokens"]

    backend._prepare, backend.forward, backend.native_iterative = prepare, forward, native
    task = SimpleNamespace(batch_size=1, target_lens=[100], ref_audio_tokens=[Tokens(200)],
                           ref_texts=["reference"], texts=["target"], langs=["uz"], instructs=[None])
    config = SimpleNamespace(num_step=32, guidance_scale=2.0, denoise=True)
    return backend, task, config, calls


def test_exact_cpu_lengths_include_prompt_style_reference_and_target():
    backend, task, config, calls = routing_backend()
    assert backend.select(task, config) == ((328, 100, 0), None)
    assert calls == [("style", "<|denoise|><|lang_start|>uz<|lang_end|><|instruct_start|>None<|instruct_end|>"),
                     ("text", "<|text_start|>reference:target<|text_end|>")]
    task.ref_audio_tokens = [None]
    task.langs, task.instructs = [None], ["female"]
    assert backend.select(task, config) == ((128, 100, 0), None)
    assert calls[-2][1] == "<|lang_start|>None<|lang_end|><|instruct_start|>female<|instruct_end|>"


@pytest.mark.parametrize("field,value,reason", [("target", 20000, "target_shape"), ("steps", 16, "generation_config"),
                                              ("guidance", 0, "generation_config"), ("batch", 2, "batch"),
                                              ("reference", 20000, "conditional_shape")])
def test_unsupported_requests_fall_back_before_any_gpu_preparation(field, value, reason):
    backend, task, config, calls = routing_backend()
    if field == "target":
        task.target_lens = [value]
    elif field == "steps":
        config.num_step = value
    elif field == "guidance":
        config.guidance_scale = value
    elif field == "batch":
        task.batch_size = value
    else:
        task.ref_audio_tokens = [SimpleNamespace(size=lambda unused: value)]
    original = backend.model.forward
    assert backend.iterative(backend.model, task, config) == ["fi tokens"]
    assert backend.model.forward is original
    assert not any(row[0] in ("prepare", "native", "forward") for row in calls)
    assert backend.fallback_reasons == {reason: 1}
    assert calls[-1] == ("fallback", task, config)


def test_profile_holes_fall_back_and_overlapping_profiles_choose_closest_optimum():
    backend, task, config, _ = routing_backend()
    backend.profiles = [{"index": 0, "c": (2, 50, 100), "u": (2, 40, 100)}]
    assert backend.select(task, config) == (None, "profile_shape")
    backend.profiles += [{"index": 1, "c": (100, 400, 1024), "u": (2, 300, 512)},
                         {"index": 2, "c": (100, 330, 1024), "u": (2, 100, 512)}]
    assert backend.select(task, config) == ((328, 100, 2), None)


def test_native_sampler_keeps_all_32_steps_and_restores_forward():
    backend, task, config, calls = routing_backend()
    original = backend.model.forward
    assert backend.iterative(backend.model, task, config) == ["native tokens"]
    assert sum(row[0] == "forward" for row in calls) == 32
    assert ("native", task, config) in calls
    assert backend.model.forward is original
    assert not backend.active and not backend.failed
    assert backend.counts == {"tensorrt": 1}


@pytest.mark.parametrize("failure", ["native", "short_sampler", "nonfinite"])
def test_failure_restores_forward_without_fi_retry_and_requires_worker_restart(failure):
    backend, task, config, calls = routing_backend()
    original = backend.model.forward
    native = backend.native_iterative

    def fail(task, config):
        if failure == "native":
            backend.model.forward()
            raise RuntimeError("synthetic execution failure")
        if failure == "short_sampler":
            return []
        tokens = native(task, config)
        backend.state["finite"] = False
        return tokens

    backend.native_iterative = fail
    with pytest.raises(RuntimeError):
        backend.iterative(backend.model, task, config)
    assert backend.model.forward is original and not backend.active and backend.failed
    assert not any(row[0] == "fallback" for row in calls)
    config.num_step = 16
    with pytest.raises(RuntimeError, match="restart"):
        backend.iterative(backend.model, task, config)
    assert not any(row[0] == "fallback" for row in calls)
