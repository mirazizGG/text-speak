"""Controller lifecycle tests without Torch, a GPU, or model downloads."""

from types import SimpleNamespace

import pytest

from flashinfer_backend import FlashInferGraphError, install

MIB = 1024 * 1024

class FakeCUDA:
    def __init__(self):
        self.allocated = 1000
        self.free = 10**10
        self.events = []

    def memory_allocated(self, device):
        return self.allocated

    def mem_get_info(self, device):
        return self.free, 2 * self.free

    def synchronize(self, device):
        self.events.append("synchronize")


class FakeGraph:
    def __init__(self, cuda, size):
        self.cuda, self.size = cuda, size
        self.fail = False

    def reset(self):
        self.cuda.events.append("reset")
        self.cuda.allocated -= self.size

    def replay(self):
        if self.fail:
            raise RuntimeError("GPU replay failed")


class FakeModel:
    def __init__(self, cuda, context):
        self.device = "cuda:0"
        self.cuda, self.context = cuda, context
        self._fi_enable_cuda_graph = False
        self._fi_graph_buckets = None
        self._fi_graph_cache = {}
        self.capture_size = 100 * MIB
        self.preparations = 0
        self.forward_calls = 0
        self.retired = []
        self.fail_capture = False

    def _prepare_inference_inputs(self, length):
        self.preparations += 1
        return {"input_ids": SimpleNamespace(size=lambda dim: length)}

    def _generate_iterative(self, task, config):
        self.forward_calls += 1
        for length in task.conditional_lens:
            self._prepare_inference_inputs(length)
        if self._fi_enable_cuda_graph:
            key = (task.conditional_lens[0], task.target_lens[0])
            entry = self._fi_graph_cache.get(key)
            if entry is None:
                if self.fail_capture:
                    raise RuntimeError("GPU capture failed")
                self.cuda.allocated += self.capture_size
                entry = {"graph": FakeGraph(self.cuda, self.capture_size),
                         "runner": SimpleNamespace(wrapper=object()), "pos_ids_i32": object(),
                         "logits": object(), "input_ids": object()}
                self.context.update(wrapper=entry["runner"].wrapper, pos_ids=entry["pos_ids_i32"])
                self._fi_graph_cache[key] = entry
                self.retired.append(entry)
            entry["graph"].replay()
        return (tuple(task.conditional_lens), tuple(task.target_lens))


def setup_backend(**options):
    cuda = FakeCUDA()
    context = {"wrapper": None, "pos_ids": None, "doc_slots": None}
    model = FakeModel(cuda, context)
    native = SimpleNamespace(_CTX=context, _generate_iterative_packed=FakeModel._generate_iterative)
    backend = install(model, torch_module=SimpleNamespace(cuda=cuda), native_module=native,
                      **{"max_entries": 2, "max_bytes": 1000 * MIB, "min_free_bytes": 100, **options})
    return model, backend, cuda, native


def run(model, conditional=20, target=10):
    task = SimpleNamespace(batch_size=1, conditional_lens=[conditional], target_lens=[target])
    return model._generate_iterative(task, None)


def test_exact_shape_reuse_prepares_once_and_restores_descriptor():
    model, backend, _, _ = setup_backend()
    assert run(model) == ((20,), (10,))
    assert run(model) == ((20,), (10,))
    assert model.preparations == 2
    assert "_prepare_inference_inputs" not in model.__dict__
    assert model._fi_enable_cuda_graph is False
    status = backend.describe()
    assert (status["captures"], status["hits"], status["graph_segments"]) == (1, 1, 2)
    assert status["cache_bytes_estimate"] == 100 * MIB


def test_lru_eviction_precedes_new_capture_and_releases_all_entry_references():
    model, backend, cuda, native = setup_backend()
    run(model, 20)
    run(model, 30)
    run(model, 20)
    victim = model.retired[1]
    # The global native context still retains the most recently captured entry.
    assert native._CTX["wrapper"] is victim["runner"].wrapper
    run(model, 40)
    assert victim == {}
    assert list(backend.cache) == [(20, 10), (40, 10)]
    assert cuda.events == ["synchronize", "reset"]
    assert backend.describe()["evictions"] == 1
    assert backend.describe()["cache_bytes_estimate"] == 200 * MIB


def test_estimated_byte_budget_evicts_after_oversized_capture_finishes():
    model, backend, cuda, native = setup_backend(max_bytes=150 * MIB)
    model.capture_size = 200 * MIB
    assert run(model) == ((20,), (10,))
    assert not backend.cache
    assert model.retired[0] == {}
    assert native._CTX["wrapper"] is None
    assert native._CTX["pos_ids"] is None
    assert cuda.events == ["synchronize", "reset"]
    # An observed capture larger than the budget disables future admissions.
    run(model)
    assert backend.describe()["memory_fallbacks"] == 1


def test_estimated_byte_budget_evicts_before_count_limit_is_reached():
    model, backend, cuda, _ = setup_backend(max_entries=8, max_bytes=250 * MIB)
    run(model, 20)
    run(model, 30)
    run(model, 40)
    assert list(backend.cache) == [(30, 10), (40, 10)]
    assert backend.describe()["cache_bytes_estimate"] == 200 * MIB
    assert backend.describe()["evictions"] == 1
    assert cuda.allocated == 1000 + 200 * MIB


@pytest.mark.parametrize("conditional,target", [(2040, 10), (10, 20), (20, 0)])
def test_unsupported_shape_remains_eager_without_capture(conditional, target):
    model, backend, _, _ = setup_backend()
    run(model, conditional, target)
    assert backend.describe()["shape_fallbacks"] == 1
    assert backend.describe()["eager_segments"] == 1
    assert not backend.cache


def test_low_free_memory_falls_back_before_capture_but_cached_graph_can_replay():
    model, backend, cuda, _ = setup_backend()
    run(model)
    cuda.free = 0
    run(model)
    run(model, 30)
    assert backend.describe()["hits"] == 1
    assert backend.describe()["captures"] == 1
    assert backend.describe()["memory_fallbacks"] == 1


def test_batch_fallback_preserves_original_preparation():
    model, backend, _, _ = setup_backend()
    task = SimpleNamespace(batch_size=2, conditional_lens=[20, 30], target_lens=[10, 10])
    assert model._generate_iterative(task, None) == ((20, 30), (10, 10))
    assert backend.describe()["batch_fallbacks"] == 1
    assert model.preparations == 2
    assert not backend.cache


@pytest.mark.parametrize("failure", ["capture", "replay"])
def test_gpu_error_poisoning_restores_guards_and_never_retries_eagerly(failure):
    model, backend, _, _ = setup_backend()
    original = model._prepare_inference_inputs
    model._prepare_inference_inputs = original
    if failure == "capture":
        model.fail_capture = True
    else:
        run(model)
        next(iter(backend.cache.values()))["graph"].fail = True
    before = model.forward_calls
    with pytest.raises(FlashInferGraphError, match="replace this worker"):
        run(model)
    assert model._prepare_inference_inputs is original
    assert model._fi_enable_cuda_graph is False
    assert backend.failed
    with pytest.raises(FlashInferGraphError, match="requires replacement"):
        run(model)
    assert model.forward_calls == before + 1


@pytest.mark.parametrize("options", [{"max_entries": 0}, {"max_bytes": True},
                                     {"max_packed_length": -1}, {"min_free_bytes": -1}])
def test_invalid_limits_fail_before_modifying_model(options):
    with pytest.raises(ValueError):
        setup_backend(**options)


def test_install_rejects_buckets_and_reinstallation():
    model, _, cuda, native = setup_backend()
    with pytest.raises(RuntimeError, match="immediately after"):
        install(model, torch_module=SimpleNamespace(cuda=cuda), native_module=native)
    fresh = FakeModel(cuda, native._CTX)
    fresh._fi_graph_buckets = [5]
    with pytest.raises(RuntimeError, match="immediately after"):
        install(fresh, torch_module=SimpleNamespace(cuda=cuda), native_module=native)


def test_eviction_synchronization_failure_poisoning_restores_methods():
    model, backend, cuda, _ = setup_backend(max_entries=1)
    run(model)

    def failed_synchronize(device):
        raise RuntimeError("asynchronous CUDA error")

    cuda.synchronize = failed_synchronize
    with pytest.raises(FlashInferGraphError):
        run(model, 30)
    assert backend.failed
    assert "_prepare_inference_inputs" not in model.__dict__
    assert model._fi_enable_cuda_graph is False
    assert backend.describe()["captures"] == 1
