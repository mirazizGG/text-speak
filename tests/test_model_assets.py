"""Model provisioning boundaries without downloading real checkpoints."""

import hashlib
import io
from pathlib import Path

import pytest

import model_assets


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    data = b"pinned-model-data"
    record = {"path": "nested/model.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(model_assets, "MODELS", {"test": {
        "repo_id": "owner/model", "revision": "fixed-revision", "files": [record],
    }})
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    return tmp_path / "model", data


def test_complete_snapshot_does_not_touch_network_or_create_files(snapshot, monkeypatch):
    destination, data = snapshot
    target = destination / "nested/model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    monkeypatch.setattr(model_assets, "urlopen", lambda *a, **k: pytest.fail("network opened"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("read-only snapshot modified"))
    assert model_assets.ensure_snapshot("test", destination) == destination
    assert not (destination / ".download.lock").exists()


@pytest.mark.parametrize("existing", [None, b"truncated", b"wrong-model-data"])
def test_missing_or_invalid_file_downloads_pinned_content_atomically(snapshot, monkeypatch, existing):
    destination, data = snapshot
    target = destination / "nested/model.bin"
    if existing is not None:
        target.parent.mkdir(parents=True)
        target.write_bytes(existing)
    requests = []

    def open_model(request, **options):
        requests.append((request.full_url, options))
        return io.BytesIO(data)

    monkeypatch.setattr(model_assets, "urlopen", open_model)
    model_assets.ensure_snapshot("test", destination)
    assert target.read_bytes() == data
    assert requests == [("https://huggingface.co/owner/model/resolve/fixed-revision/nested/model.bin", {"timeout": 60})]
    assert not list(destination.rglob("*.part"))
    model_assets.ensure_snapshot("test", destination)
    assert len(requests) == 1


@pytest.mark.parametrize("content", [b"wrong-model-data", b"short"])
def test_failed_checksum_keeps_original_and_removes_own_partial(snapshot, monkeypatch, content):
    destination, _data = snapshot
    target = destination / "nested/model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing-content")
    unrelated = target.parent / ".unrelated.part"
    unrelated.write_bytes(b"other startup")
    monkeypatch.setattr(model_assets, "urlopen", lambda *a, **k: io.BytesIO(content))
    with pytest.raises(RuntimeError, match="checksum/size mismatch"):
        model_assets.ensure_snapshot("test", destination)
    assert target.read_bytes() == b"existing-content"
    assert list(target.parent.glob("*.part")) == [unrelated]


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
@pytest.mark.parametrize("value", ["1", "TRUE", "yes", "On"])
def test_offline_mode_never_creates_destination_or_opens_network(snapshot, monkeypatch, variable, value):
    destination, _data = snapshot
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(model_assets, "urlopen", lambda *a, **k: pytest.fail("network opened"))
    with pytest.raises(RuntimeError, match="automatic download is disabled"):
        model_assets.ensure_snapshot("test", destination)
    assert not destination.exists()


def test_explicit_download_optout_never_opens_network(snapshot, monkeypatch):
    destination, _data = snapshot
    monkeypatch.setattr(model_assets, "urlopen", lambda *a, **k: pytest.fail("network opened"))
    with pytest.raises(RuntimeError, match="automatic download is disabled"):
        model_assets.ensure_snapshot("test", destination, allow_download=False)
    assert not destination.exists()


def test_rechecks_snapshot_after_acquiring_process_lock(snapshot, monkeypatch):
    import filelock

    destination, data = snapshot

    class OtherStartupFinished:
        def __init__(self, *args, **options):
            pass

        def __enter__(self):
            target = destination / "nested/model.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(data)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(filelock, "FileLock", OtherStartupFinished)
    monkeypatch.setattr(model_assets, "urlopen", lambda *a, **k: pytest.fail("duplicate download"))
    model_assets.ensure_snapshot("test", destination)


def test_connection_failure_leaves_no_final_or_partial_model(snapshot, monkeypatch):
    destination, _data = snapshot

    def disconnected(*args, **options):
        raise OSError("connection lost")

    monkeypatch.setattr(model_assets, "urlopen", disconnected)
    with pytest.raises(RuntimeError, match="internet access, free disk space and write permissions"):
        model_assets.ensure_snapshot("test", destination)
    assert not (destination / "nested/model.bin").exists()
    assert not list(destination.rglob("*.part"))
