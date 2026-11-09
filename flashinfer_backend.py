"""Bounded exact-shape graphs over the pinned OmniVoice FlashInfer sampler.

The native sampler and its math remain unchanged. Install after apply_flashinfer
with graphs disabled, in a serialized synthesis worker. CUDA failures poison the
controller; the caller must replace that worker, never retry eagerly in it.
"""

from collections import Counter, OrderedDict
from importlib import metadata
import threading


_MIB = 1024 * 1024
_MISSING = object()


class FlashInferGraphError(RuntimeError):
    """The worker must be replaced before any further inference."""


class _GraphCache(OrderedDict):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def get(self, key, default=None):
        owner = self.owner
        if key != owner._admitted_key:
            raise RuntimeError("Native FlashInfer graph shape differs from admission.")
        if key in self:
            self.move_to_end(key)
            owner._counts["hits"] += 1
            return super().get(key)
        while self and (len(self) >= owner.max_entries
                        or owner._bytes + owner._estimate > owner.max_bytes):
            owner._evict()
        owner._capture_start = owner.torch.cuda.memory_allocated(owner.model.device)
        return default

    def __setitem__(self, key, entry):
        owner = self.owner
        if key != owner._admitted_key or key in self or owner._capture_start is None:
            raise RuntimeError("Unexpected native FlashInfer graph insertion.")
        used = max(0, owner.torch.cuda.memory_allocated(owner.model.device) - owner._capture_start)
        owner._capture_start = None
        owner._sizes[key] = used
        owner._bytes += used
        owner._estimate = max(owner._estimate, used)
        owner._counts["captures"] += 1
        super().__setitem__(key, entry)
        # Native code holds this entry locally until the iterative call ends.
        # An unexpectedly large capture is retired only after that call.


class FlashInferGraphBackend:
    def __init__(self, model, torch_module, native_module, *, max_entries,
                 max_bytes, max_packed_length, min_free_bytes):
        self.model = model
        self.torch = torch_module
        self.native_module = native_module
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_packed_length = max_packed_length
        self.min_free_bytes = min_free_bytes
        self.failed = False
        self._native = model._generate_iterative
        self._lock = threading.RLock()
        self._counts = Counter()
        self._sizes = {}
        self._bytes = 0
        self._estimate = min(96 * _MIB, max_bytes)
        self._admitted_key = None
        self._capture_start = None
        self.cache = _GraphCache(self)
        model._fi_graph_cache = self.cache
        model._generate_iterative = self.generate_iterative

    def _evict(self):
        # FlashInfer has custom kernels whose inputs must remain alive until
        # every queued replay finishes, including the last call's outputs.
        self.torch.cuda.synchronize(self.model.device)
        key, entry = self.cache.popitem(last=False)
        context = self.native_module._CTX
        runner = entry.get("runner")
        if runner is not None and context.get("wrapper") is runner.wrapper:
            context["wrapper"] = None
        if context.get("pos_ids") is entry.get("pos_ids_i32"):
            context["pos_ids"] = None
        entry["graph"].reset()
        entry.clear()
        self._bytes -= self._sizes.pop(key)
        self._counts["evictions"] += 1

    def _admit(self, key):
        if (len(key) != 2 or any(type(length) is not int or length <= 0 for length in key)
                or key[0] < key[1] or sum(key) > self.max_packed_length):
            self._counts["shape_fallbacks"] += 1
            return False
        # Reusing an existing graph does not allocate another graph pool.
        if key not in self.cache:
            free, _ = self.torch.cuda.mem_get_info(self.model.device)
            if self._estimate > self.max_bytes or free < self.min_free_bytes + self._estimate:
                self._counts["memory_fallbacks"] += 1
                return False
        self._admitted_key = key
        return True

    def generate_iterative(self, task, gen_config):
        with self._lock:
            if self.failed:
                raise FlashInferGraphError("FlashInfer graph worker requires replacement.")
            model = self.model
            previous_flag = model._fi_enable_cuda_graph
            previous_prepare = model.__dict__.get("_prepare_inference_inputs", _MISSING)
            prepare = model._prepare_inference_inputs
            self._admitted_key = None
            self._capture_start = None
            model._fi_enable_cuda_graph = False
            observed = 0

            def observe(*args, **kwargs):
                nonlocal observed
                inputs = prepare(*args, **kwargs)
                observed += 1
                if observed != 1:
                    raise RuntimeError("Unexpected native FlashInfer input preparation count.")
                key = (inputs["input_ids"].size(2), task.target_lens[0])
                model._fi_enable_cuda_graph = self._admit(key)
                return inputs

            try:
                if task.batch_size == 1 and len(task.target_lens) == 1:
                    # OmniVoice 0.2.1 reads the graph flag AFTER preparing all
                    # inputs. Observe those results without tokenizing twice.
                    model._prepare_inference_inputs = observe
                else:
                    self._counts["batch_fallbacks"] += 1
                result = self._native(task, gen_config)
                self._counts["graph_segments" if self._admitted_key is not None else "eager_segments"] += 1
                while self.cache and (len(self.cache) > self.max_entries or self._bytes > self.max_bytes):
                    self._evict()
                return result
            except Exception as exc:
                self.failed = True
                raise FlashInferGraphError("FlashInfer inference failed; replace this worker.") from exc
            finally:
                if previous_prepare is _MISSING:
                    model.__dict__.pop("_prepare_inference_inputs", None)
                else:
                    model._prepare_inference_inputs = previous_prepare
                model._fi_enable_cuda_graph = previous_flag
                self._admitted_key = None
                self._capture_start = None

    def describe(self):
        with self._lock:
            return {
                "kind": "native_flashinfer_exact_shape_graphs", "failed": self.failed,
                "cache_count": len(self.cache), "cache_bytes_estimate": self._bytes,
                "memory_accounting": "positive_memory_allocated_delta_per_capture",
                "max_entries": self.max_entries, "max_bytes_estimate": self.max_bytes,
                "max_packed_length": self.max_packed_length,
                "min_free_bytes": self.min_free_bytes,
                **{key: self._counts[key] for key in ("graph_segments", "eager_segments", "hits",
                    "captures", "evictions", "shape_fallbacks", "memory_fallbacks", "batch_fallbacks")},
            }


def install(model, *, max_entries=8, max_bytes=768 * _MIB,
            max_packed_length=2048, min_free_bytes=4096 * _MIB,
            torch_module=None, native_module=None):
    """Install once on the pinned sampler; import has no CUDA side effects.

The byte limit estimates persistent allocation, not CUDA-reserved memory or a
hard capture peak limit. The size guard and free-memory reserve bound admission.
"""
    for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes),
                        ("max_packed_length", max_packed_length), ("min_free_bytes", min_free_bytes)):
        if type(value) is not int or value < (0 if name == "min_free_bytes" else 1):
            raise ValueError(f"Invalid FlashInfer graph limit: {name}.")
    if native_module is None:
        if metadata.version("omnivoice") != "0.2.1":
            raise RuntimeError("Exact-shape graphs require pinned OmniVoice 0.2.1.")
        from omnivoice.models import omnivoice_flashinfer as native_module
    if torch_module is None:
        import torch as torch_module
    if (getattr(model._generate_iterative, "__func__", None) is not native_module._generate_iterative_packed
            or getattr(model, "_fi_graph_buckets", None) is not None
            or getattr(model, "_fi_enable_cuda_graph", True)
            or getattr(model, "_fi_graph_cache", None) != {}):
        raise RuntimeError("Install graphs immediately after native FlashInfer with empty, disabled graphs.")
    return FlashInferGraphBackend(model, torch_module, native_module, max_entries=max_entries,
                                 max_bytes=max_bytes, max_packed_length=max_packed_length,
                                 min_free_bytes=min_free_bytes)
