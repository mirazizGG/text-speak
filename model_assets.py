"""Acquire the pinned public model files before inference starts."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

MODELS = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
_SMALL_FILE = 16 * 1024 * 1024


def offline_mode():
    return any(os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
               for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def _complete(path, record):
    try:
        if not path.is_file() or path.stat().st_size != record["size"]:
            return False
        # Check metadata/tokenizers fully, without rereading multi-GB weights
        # in every spawned speech process. Downloads always get a full SHA256.
        return record["size"] > _SMALL_FILE or hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    except OSError:
        return False


def _missing(destination, specification):
    return [record for record in specification["files"]
            if not _complete(destination / record["path"], record)]


def _download(destination, specification, record):
    target = destination / record["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (f'https://huggingface.co/{specification["repo_id"]}/resolve/'
           f'{specification["revision"]}/{quote(record["path"], safe="/")}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            received = 0
            reported = 0
            print(f'Downloading {specification["repo_id"]}: {record["path"]} ({record["size"] / 2**20:.1f} MiB)', flush=True)
            with urlopen(Request(url, headers={"User-Agent": "text-speak-model-loader"}), timeout=60) as response:
                while chunk := response.read(8 * 1024 * 1024):
                    received += len(chunk)
                    if received > record["size"]:
                        raise RuntimeError(f'Model download checksum/size mismatch: {record["path"]}')
                    output.write(chunk)
                    digest.update(chunk)
                    if received - reported >= 64 * 1024 * 1024:
                        print(f'{record["path"]}: {received / 2**20:.0f} / {record["size"] / 2**20:.0f} MiB', flush=True)
                        reported = received
            if received != record["size"] or digest.hexdigest() != record["sha256"]:
                raise RuntimeError(f'Model download checksum/size mismatch: {record["path"]}')
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_snapshot(name, destination, *, allow_download=True):
    """Reuse complete local files, downloading and verifying missing files only."""
    specification = MODELS[name]
    destination = Path(destination).expanduser().resolve()
    missing = _missing(destination, specification)
    if not missing:
        print(f'Using local {specification["repo_id"]} model files: {destination}', flush=True)
        return destination
    if not allow_download or offline_mode():
        raise RuntimeError(
            f'{specification["repo_id"]} model files are missing/incomplete in {destination}; '
            'automatic download is disabled. Prepare the model files or enable downloads '
            '(TTS_AUTO_DOWNLOAD_MODELS=1, HF_HUB_OFFLINE=0, TRANSFORMERS_OFFLINE=0).'
        )

    from filelock import FileLock

    try:
        destination.mkdir(parents=True, exist_ok=True)
        # OS locks are released on a crash. Recheck after another startup may
        # have finished provisioning the same shared directory.
        with FileLock(str(destination / ".download.lock"), timeout=1800):
            for record in _missing(destination, specification):
                _download(destination, specification, record)
    except OSError as exc:
        raise RuntimeError(
            f'Cannot prepare {specification["repo_id"]} in {destination}. '
            'Check internet access, free disk space and write permissions, then restart.'
        ) from exc
    return destination
