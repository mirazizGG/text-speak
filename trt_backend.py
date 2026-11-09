"""Optional, checkpoint-specific TensorRT routing over the native sampler.

Call ``install`` after initializing FlashInfer, passing the native iterative
method saved before FlashInfer was applied. TensorRT plans and their manifests
must be deployment-owned artifacts; the digest detects damage, not authorship.
Importing this module does not import CUDA packages or change a model.
"""

from collections import Counter
import hashlib
from importlib import metadata
import json
import mmap
from pathlib import Path, PurePosixPath
import re
import threading
from types import MethodType, SimpleNamespace


MODEL_ID = "k2-fsa/OmniVoice"
MODEL_REVISION = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
VERSIONS = {"torch": "2.10.0+cu128", "cuda": "12.8", "tensorrt": "10.14.1.48.post1",
            "transformers": "5.14.1", "omnivoice": "0.2.1"}
INPUT_NAMES = ("c_embeds", "u_embeds", "c_positions", "u_positions")


class TensorRTCompatibilityError(RuntimeError):
    """An artifact/runtime is incompatible before installing the dispatcher."""


def _load_manifest(path, model_id, model_revision):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise TensorRTCompatibilityError("TensorRT manifest must be a regular deployment file")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TensorRTCompatibilityError("Cannot read TensorRT manifest") from exc
    if not isinstance(manifest, dict):
        raise TensorRTCompatibilityError("TensorRT manifest must be an object")
    required = {"schema_version": 1, "kind": "unpadded_cfg_shared_linear", "batch": 2,
                "hidden": 1024, "num_audio_codebook": 8, "audio_vocab_size": 1025,
                "steps": 32, "min_length": 2, "min_target": 2}
    for name, value in required.items():
        if type(manifest.get(name)) is not type(value) or manifest[name] != value:
            raise TensorRTCompatibilityError(f"Unsupported TensorRT manifest field: {name}")
    if (model_id != MODEL_ID or model_revision != MODEL_REVISION
            or manifest.get("model_id") != model_id or manifest.get("model_revision") != model_revision):
        raise TensorRTCompatibilityError("TensorRT checkpoint identity differs from the pinned model")
    if (manifest.get("causal") is not False or isinstance(manifest.get("guidance_scale"), bool)
            or manifest.get("guidance_scale") != 2.0 or manifest.get("versions") != VERSIONS
            or manifest.get("compute_capability") != [12, 0]
            or any(type(value) is not int for value in manifest.get("compute_capability", []))
            or manifest.get("input_names") != list(INPUT_NAMES)):
        raise TensorRTCompatibilityError("Unsupported TensorRT graph/runtime contract")
    for name in ("max_length", "max_target"):
        if type(manifest.get(name)) is not int or manifest[name] < 2:
            raise TensorRTCompatibilityError(f"Invalid TensorRT profile bound: {name}")
    if manifest["max_target"] > manifest["max_length"]:
        raise TensorRTCompatibilityError("TensorRT target bound exceeds its sequence bound")
    if not isinstance(manifest.get("plan_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["plan_sha256"]):
        raise TensorRTCompatibilityError("TensorRT manifest requires the plan SHA256")
    return manifest


def _validate_versions(torch):
    distributions = {"torch": "torch", "tensorrt": "tensorrt-cu12",
                     "transformers": "transformers", "omnivoice": "omnivoice"}
    try:
        actual = {name: metadata.version(distribution) for name, distribution in distributions.items()}
    except metadata.PackageNotFoundError as exc:
        raise TensorRTCompatibilityError("A pinned TensorRT dependency is absent") from exc
    actual["cuda"] = torch.version.cuda
    if actual != VERSIONS or str(torch.__version__) != VERSIONS["torch"]:
        raise TensorRTCompatibilityError("TensorRT requires its exact validated dependency versions")
    return actual


def _verify_model_files(model_path, files):
    """Bind the compiled weights to the actual read-only checkpoint files."""
    requested = Path(model_path)
    if requested.is_symlink() or not requested.is_dir():
        raise TensorRTCompatibilityError("TensorRT requires a regular checkpoint directory")
    root = requested.resolve(strict=True)
    if not isinstance(files, dict) or "config.json" not in files:
        raise TensorRTCompatibilityError("TensorRT manifest requires checkpoint file digests")
    weight_names = {path.name for path in root.glob("model*.safetensors")}
    if not weight_names or not weight_names.issubset(files):
        raise TensorRTCompatibilityError("TensorRT manifest must cover every checkpoint weight shard")
    verified = {}
    for name, digest in files.items():
        if not isinstance(name, str) or "\\" in name or ":" in name:
            raise TensorRTCompatibilityError("Invalid checkpoint manifest path")
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts) or str(relative) != name:
            raise TensorRTCompatibilityError("Checkpoint manifest path escapes its directory")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise TensorRTCompatibilityError("Invalid checkpoint SHA256")
        path = root.joinpath(*relative.parts)
        if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents)
                or not path.is_file() or not path.resolve(strict=True).is_relative_to(root)):
            raise TensorRTCompatibilityError("Checkpoint files must stay within the model directory without links")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise TensorRTCompatibilityError(f"TensorRT checkpoint SHA256 differs for {name}")
        verified[name] = digest
    return verified


class TensorRTBackend:
    """One model, execution context, CUDA stream and at most one captured graph.

    The owning engine serializes model use; this lock additionally serializes
    calls entering the iterative dispatcher. No CUDA object is shared across
    spawned speech processes. Unexpected execution failures require a restart.
    """

    def __init__(self, model, plan_path, manifest, native_iterative, *, model_path, use_cuda_graph):
        import torch

        self.model, self.torch, self.manifest = model, torch, manifest
        self.native_iterative = native_iterative
        self.fallback_iterative = model._generate_iterative
        self.lock = threading.RLock()
        self.state = None
        self.failed = False
        self.active = False
        self.forwards = 0
        self.use_cuda_graph = use_cuda_graph
        self.counts = Counter()
        self.fallback_reasons = Counter()
        versions = _validate_versions(torch)
        model_files = _verify_model_files(model_path, manifest.get("model_files"))
        if (getattr(native_iterative, "__self__", None) is not model or not hasattr(model, "_fi_runner")
                or hasattr(model, "_trt_backend") or model.dtype != torch.float16
                or not str(model.device).startswith("cuda") or model.llm.config.model_type != "qwen3"
                or model.llm.has_sliding_layers or model.llm.config.rope_parameters.get("rope_type") != "default"
                or model.llm.config.hidden_size != manifest["hidden"]
                or model.config.num_audio_codebook != manifest["num_audio_codebook"]
                or model.config.audio_vocab_size != manifest["audio_vocab_size"]
                or model.config.audio_mask_id != manifest["audio_vocab_size"] - 1):
            raise TensorRTCompatibilityError("TensorRT requires the validated FlashInfer-patched model and pristine native sampler")
        if tuple(torch.cuda.get_device_capability(model.device)) != tuple(manifest["compute_capability"]):
            raise TensorRTCompatibilityError("TensorRT engine was built for another GPU architecture")
        from omnivoice.models.omnivoice import _combine_text, _tokenize_with_nonverbal_tags

        self.combine_text, self.tokenize_text = _combine_text, _tokenize_with_nonverbal_tags
        plan_path = Path(plan_path)
        if plan_path.is_symlink() or not plan_path.is_file():
            raise TensorRTCompatibilityError("TensorRT plan must be a regular deployment file")
        # Validate the exact mapped bytes before TensorRT parses the plan. mmap
        # avoids materializing a second ~900 MB Python bytes object per worker.
        with plan_path.open("rb") as plan_file:
            if plan_path.stat().st_size < 1:
                raise TensorRTCompatibilityError("TensorRT plan is empty")
            with mmap.mmap(plan_file.fileno(), 0, access=mmap.ACCESS_READ) as plan_data:
                if hashlib.sha256(plan_data).hexdigest() != manifest["plan_sha256"]:
                    raise TensorRTCompatibilityError("TensorRT plan SHA256 differs from its manifest")
                import tensorrt as trt

                self.trt = trt
                self.logger = trt.Logger(trt.Logger.WARNING)
                self.runtime = trt.Runtime(self.logger)
                self.runtime.engine_host_code_allowed = False
                self.engine = self.runtime.deserialize_cuda_engine(plan_data)
        if self.engine is None:
            raise RuntimeError("Cannot deserialize the verified TensorRT engine")
        expected = {"c_embeds": trt.float16, "u_embeds": trt.float16,
                    "c_positions": trt.int64, "u_positions": trt.int64, "logits": trt.float16}
        names = {self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)}
        if names != set(expected) or any(self.engine.get_tensor_dtype(name) != dtype for name, dtype in expected.items()):
            raise RuntimeError("TensorRT engine signature differs from its verified manifest")
        self.profiles = self._read_profiles()
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create the TensorRT execution context")
        self.context.nvtx_verbosity = trt.ProfilingVerbosity.NONE
        self.stream = torch.cuda.Stream(device=model.device)
        c_max, t_max = manifest["max_length"], manifest["max_target"]
        codebooks, vocab, hidden = manifest["num_audio_codebook"], manifest["audio_vocab_size"], manifest["hidden"]
        buffers = (2 * c_max * hidden * 2 + (c_max + t_max) * hidden * 2
                   + 2 * codebooks * (c_max + t_max) * vocab * 2
                   + 2 * codebooks * t_max * vocab + 2 * c_max * c_max
                   + 2 * codebooks * c_max * 8 + 2 * (c_max + t_max) * 8)
        context_bytes = int(self.engine.device_memory_size_v2)
        plan_bytes = plan_path.stat().st_size
        self.estimated_extra_gpu_bytes = plan_bytes + context_bytes + buffers
        self.metadata = {"backend": "tensorrt_with_flashinfer_fallback", "model_revision": MODEL_REVISION,
                         "plan_sha256": manifest["plan_sha256"], "versions": versions,
                         "model_files": model_files,
                         "steps": 32, "guidance_scale": 2.0, "max_length": c_max, "max_target": t_max,
                         "cuda_graph": use_cuda_graph, "max_retained_graphs": 1 if use_cuda_graph else 0,
                         "plan_file_bytes": plan_bytes, "context_upper_bound_bytes": context_bytes,
                         "estimated_max_torch_buffer_bytes": buffers,
                         "estimated_extra_gpu_bytes": self.estimated_extra_gpu_bytes,
                         "memory_estimate_note": "Plan size approximates weights; excludes driver/graph allocations and is not a measured VRAM guarantee."}

    def _read_profiles(self):
        profiles = []
        h = self.manifest["hidden"]
        for index in range(self.engine.num_optimization_profiles):
            shapes = {name: [tuple(value) for value in self.engine.get_tensor_profile_shape(name, index)]
                      for name in INPUT_NAMES}
            for prefix in ("c", "u"):
                embeds, positions = shapes[prefix + "_embeds"], shapes[prefix + "_positions"]
                if (len(embeds) != 3 or len(positions) != 3
                        or any(len(value) != 3 or value[0] != 1 or value[2] != h or value[1] < 2 for value in embeds)
                        or any(position != embed[:2] for position, embed in zip(positions, embeds))):
                    raise RuntimeError("TensorRT profile dimensions differ from the checkpoint")
            profiles.append({"index": index, "c": tuple(shape[1] for shape in shapes["c_embeds"]),
                             "u": tuple(shape[1] for shape in shapes["u_embeds"])})
        if not profiles:
            raise RuntimeError("TensorRT engine contains no input profiles")
        return profiles

    def select(self, task, config):
        """Select on CPU, before the native sampler allocates any GPU inputs."""
        if task.batch_size != 1 or len(task.target_lens) != 1:
            return None, "batch"
        if config.num_step != 32 or config.guidance_scale != 2.0:
            return None, "generation_config"
        t_len = task.target_lens[0]
        if type(t_len) is not int or not self.manifest["min_target"] <= t_len <= self.manifest["max_target"]:
            return None, "target_shape"
        reference = task.ref_audio_tokens[0]
        reference_frames = int(reference.size(-1)) if reference is not None else 0
        # This is the exact token construction in pinned OmniVoice 0.2.1
        # _prepare_inference_inputs, without its .to(cuda), masks or audio IDs.
        style = "<|denoise|>" if config.denoise and reference is not None else ""
        style += f"<|lang_start|>{task.langs[0] or 'None'}<|lang_end|>"
        style += f"<|instruct_start|>{task.instructs[0] or 'None'}<|instruct_end|>"
        text = self.combine_text(ref_text=task.ref_texts[0], text=task.texts[0])
        with self.torch.device("cpu"):
            style_length = self.model.text_tokenizer(style, return_tensors="pt").input_ids.size(-1)
            text_length = self.tokenize_text(f"<|text_start|>{text}<|text_end|>", self.model.text_tokenizer).size(-1)
        c_len = int(style_length + text_length + reference_frames + t_len)
        if not t_len <= c_len <= self.manifest["max_length"] or c_len < self.manifest["min_length"]:
            return None, "conditional_shape"
        available = [profile for profile in self.profiles
                     if profile["c"][0] <= c_len <= profile["c"][2] and profile["u"][0] <= t_len <= profile["u"][2]]
        if not available:
            return None, "profile_shape"
        profile = min(available, key=lambda value: abs(value["c"][1] - c_len) + abs(value["u"][1] - t_len))
        return (c_len, t_len, profile["index"]), None

    def iterative(self, unused_model, task, config):
        with self.lock:
            if self.failed:
                raise RuntimeError("TensorRT execution previously failed; restart this speech worker")
            selection, reason = self.select(task, config)
            if selection is None:
                tokens = self.fallback_iterative(task, config)
                self.counts["flashinfer"] += 1
                self.fallback_reasons[reason] += 1
                return tokens
            previous_forward = self.model.forward
            self.active = True
            try:
                self._prepare(*selection)
                self.model.forward = MethodType(self.forward, self.model)
                tokens = self.native_iterative(task, config)
                if self.forwards != 32:
                    raise RuntimeError("Native TensorRT sampler did not complete all 32 forwards")
                if not bool(self.state["finite"]):
                    raise RuntimeError("TensorRT produced nonfinite logits")
                self.counts["tensorrt"] += 1
                return tokens
            except BaseException:
                # No fallback after entering this path: work or RNG may have
                # advanced, and a CUDA failure may invalidate the context.
                self.failed = True
                raise
            finally:
                self.model.forward = previous_forward
                self.active = False

    def _prepare(self, c_len, t_len, profile):
        torch = self.torch
        self.forwards = 0
        if self.state is not None and self.state["shape"] == (c_len, t_len, profile):
            self.state["finite"].fill_(True)
            return
        # Finish all graph/buffer use before replacing the one retained shape.
        if self.state is not None:
            torch.cuda.current_stream(self.model.device).synchronize()
            self.stream.synchronize()
            self.state = None
        if self.context.active_optimization_profile != profile:
            if not self.context.set_optimization_profile_async(profile, self.stream.cuda_stream):
                raise RuntimeError("Cannot select TensorRT input profile")
        hidden, codebooks, vocab = (self.manifest[name] for name in ("hidden", "num_audio_codebook", "audio_vocab_size"))
        values = {"c_embeds": torch.empty(1, c_len, hidden, device=self.model.device, dtype=torch.float16),
                  "u_embeds": torch.empty(1, t_len, hidden, device=self.model.device, dtype=torch.float16),
                  "c_positions": torch.arange(c_len, device=self.model.device).unsqueeze(0),
                  "u_positions": torch.arange(t_len, device=self.model.device).unsqueeze(0)}
        for name, value in values.items():
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise RuntimeError(f"TensorRT rejected its selected profile for {name}")
        output_shape = (2, codebooks, t_len, vocab)
        if tuple(self.context.get_tensor_shape("logits")) != output_shape:
            raise RuntimeError("TensorRT output dimensions differ from the checkpoint")
        values["logits"] = torch.empty(output_shape, device=self.model.device, dtype=torch.float16)
        for name, value in values.items():
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"Cannot bind TensorRT input/output {name}")
        self.state = {"shape": (c_len, t_len, profile), "values": values, "graph": None,
                      "finite": torch.ones((), device=self.model.device, dtype=torch.bool),
                      "full_logits": torch.zeros(2, codebooks, c_len, vocab, device=self.model.device, dtype=torch.float16)}

    def _execute(self):
        def enqueue():
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")

        if not self.use_cuda_graph:
            enqueue()
            return
        if self.state["graph"] is None:
            enqueue()
            enqueue()
            self.stream.synchronize()
            graph = self.torch.cuda.CUDAGraph()
            with self.torch.cuda.graph(graph, stream=self.stream):
                enqueue()
            self.state["graph"] = graph
        with self.torch.cuda.stream(self.stream):
            self.state["graph"].replay()

    def forward(self, unused_model, input_ids, audio_mask, attention_mask=None, **kwargs):
        torch, state = self.torch, self.state
        if not self.active or state is None or kwargs:
            raise RuntimeError("Unexpected call outside the native TensorRT sampler")
        c_len, t_len, _ = state["shape"]
        if (tuple(input_ids.shape) != (2, self.manifest["num_audio_codebook"], c_len)
                or tuple(audio_mask.shape) != (2, c_len) or attention_mask is None
                or attention_mask.dtype != torch.bool or tuple(attention_mask.shape) != (2, 1, c_len, c_len)
                or self.forwards >= 32):
            raise RuntimeError("Native input dimensions changed after CPU TensorRT preflight")
        embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)
        state["values"]["c_embeds"].copy_(embeds[0:1, :c_len])
        state["values"]["u_embeds"].copy_(embeds[1:2, :t_len])
        caller = torch.cuda.current_stream(self.model.device)
        self.stream.wait_stream(caller)
        self._execute()
        caller.wait_stream(self.stream)
        result = state["values"]["logits"]
        state["finite"].logical_and_(result.isfinite().all())
        state["full_logits"][0, :, c_len - t_len:c_len].copy_(result[0])
        state["full_logits"][1, :, :t_len].copy_(result[1])
        self.forwards += 1
        return SimpleNamespace(logits=state["full_logits"])

    def describe(self):
        return {**self.metadata, "segments": dict(self.counts), "fallback_reasons": dict(self.fallback_reasons),
                "failed": self.failed}


def install(model, plan_path, manifest_path, *, native_iterative, model_id, model_revision, model_path, use_cuda_graph=True):
    """Install explicitly; no application imports or environment default changes."""
    if not isinstance(use_cuda_graph, bool):
        raise ValueError("use_cuda_graph must be a boolean")
    manifest = _load_manifest(manifest_path, model_id, model_revision)
    backend = TensorRTBackend(model, plan_path, manifest, native_iterative, model_path=model_path, use_cuda_graph=use_cuda_graph)
    model._generate_iterative = MethodType(backend.iterative, model)
    model._trt_backend = backend
    return backend
