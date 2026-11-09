"""Graph configuration and failed-worker isolation at application boundaries."""
import sys
import types

import pytest

from api import Settings, create_app
from engine import OmniVoiceEngine, WorkerRestartRequired, _SegmentBudgetError
from test_api import API_KEY, FakeEngine
from test_engine import accelerated_engine, runtime


def test_worker_graphs_are_opt_in_and_parent_model_always_stays_eager(tmp_path, monkeypatch):
    constructed = []

    def factory(**options):
        engine = FakeEngine()
        engine.options = options
        engine.model_path = tmp_path / "resolved-tts"
        engine.asr_model_path = tmp_path / "resolved-asr"
        constructed.append(engine)
        return engine

    monkeypatch.setattr("engine.OmniVoiceEngine", factory)
    monkeypatch.setenv("TTS_ACCELERATION", "flashinfer")
    for enabled in ["0", "1"]:
        monkeypatch.setenv("TTS_FLASHINFER_GRAPHS", enabled)
        app = create_app(Settings(api_key=API_KEY, jobs_dir=tmp_path/enabled))
        try:
            assert constructed[-1].options["flashinfer_graphs"] is False
            assert app.state.worker.backend.config["flashinfer_graphs"] is (enabled == "1")
            assert app.state.worker.backend.config["num_steps"] == 32
            assert app.state.worker.backend.config["auto_download_models"] is False
            assert app.state.worker.backend.config["model_path"] == constructed[-1].model_path
            assert app.state.worker.backend.config["asr_model_path"] == constructed[-1].asr_model_path
        finally:
            app.state.jobs.close()
            app.state.admin_worker.executor.shutdown()
            app.state.worker._storage_executor.shutdown()


def test_invalid_graph_flag_fails_before_loading_or_starting_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_FLASHINFER_GRAPHS", "sometimes")
    with pytest.raises(ValueError, match="TTS_FLASHINFER_GRAPHS"):
        create_app(Settings(api_key=API_KEY, jobs_dir=tmp_path))


def test_invalid_download_flag_fails_before_loading_or_starting_workers(tmp_path, monkeypatch):
    monkeypatch.setenv("TTS_AUTO_DOWNLOAD_MODELS", "sometimes")
    with pytest.raises(ValueError, match="TTS_AUTO_DOWNLOAD_MODELS"):
        create_app(Settings(api_key=API_KEY, jobs_dir=tmp_path))


def test_graph_backend_is_installed_before_readiness_warmup(runtime, monkeypatch):
    installed = []
    backend = types.SimpleNamespace(failed=False)
    monkeypatch.setitem(sys.modules, "flashinfer_backend", types.SimpleNamespace(install=lambda model: installed.append(model) or backend))
    engine = accelerated_engine(runtime, monkeypatch, lambda *args, **kwargs: None)
    engine.flashinfer_graphs = True
    engine.load()
    assert installed == [engine._model]
    assert engine._flashinfer_backend is backend
    assert engine.capabilities["inference_acceleration"] == "flashinfer+cuda-graphs"
    assert all(call.get("generation_config").num_step == 32 for call in engine._model.calls if call.get("generation_config"))


@pytest.mark.parametrize("segmented", [False, True])
def test_failed_graph_backend_requires_worker_replacement_in_every_generation_path(runtime, monkeypatch, segmented):
    engine = runtime.engine
    engine._flashinfer_backend = types.SimpleNamespace(failed=True)
    if segmented:
        calls = 0

        def preflight(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _SegmentBudgetError("Needs segmentation")

        monkeypatch.setattr(engine, "_preflight", preflight)
    with pytest.raises(WorkerRestartRequired) as error:
        engine.execute("speech", {"text": "fail", "voice": "female_1"})
    assert error.value.recoverable is True


@pytest.mark.parametrize("name", ["_trt_backend", "_flashinfer_backend"])
def test_process_boundary_keeps_recovery_when_rng_cleanup_replaces_backend_error(monkeypatch, name):
    import inference_pool

    def fail(*args):
        raise RuntimeError("CUDA RNG restoration failed")

    monkeypatch.setattr(inference_pool, "_MODEL", types.SimpleNamespace(**{name: types.SimpleNamespace(failed=True)}, execute_pinned_speech=fail))
    with pytest.raises(WorkerRestartRequired) as error:
        inference_pool._synthesize_job({"id": "a"*32, "payload": {}})
    assert error.value.recoverable is True


def test_graph_setting_requires_a_boolean(runtime):
    with pytest.raises(ValueError, match="flashinfer_graphs"):
        OmniVoiceEngine(**runtime.settings, flashinfer_graphs="1")
