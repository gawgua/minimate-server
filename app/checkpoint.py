import json
import logging
from pathlib import Path

from .config import MODEL_ID

log = logging.getLogger(__name__)

ASR_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken"]
# Config, weights, and tokenizer files only: skip notebooks and training_args.bin.
INTENT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def latest_snapshot(repo_id: str, allow_patterns, cache_dir=None, progress=None) -> Path:
    """Download the newest Hub commit, or fall back to the cache when the Hub is unreachable."""
    import requests
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

    options = {
        "repo_id": repo_id,
        "cache_dir": cache_dir,
        "allow_patterns": allow_patterns,
        "ignore_patterns": ["*/*"],
    }
    try:
        # Probe first so an unreachable Hub is logged instead of silently using the cache.
        HfApi().model_info(repo_id)
    except OfflineModeIsEnabled:
        log.info("HF_HUB_OFFLINE is set. Using the cached %s without checking for updates", repo_id)
        path = Path(snapshot_download(**options, local_files_only=True))
    except (requests.ConnectionError, requests.Timeout, HfHubHTTPError) as exc:
        # Auth and missing-repo errors are configuration mistakes, so only network and 5xx fall back.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(exc, HfHubHTTPError) and (status is None or status < 500):
            raise
        log.warning("Could not check %s for new commits (%s). Using the cached copy", repo_id, exc)
        path = Path(snapshot_download(**options, local_files_only=True))
    else:
        # Download by branch, not commit hash: only a branch download updates refs/main,
        # which the offline fallback above needs to find the newest cached snapshot.
        if progress is not None:
            options["tqdm_class"] = file_progress(progress)
        path = Path(snapshot_download(**options))
    log.info("Using %s at commit %s", repo_id, path.name)
    return path


def file_progress(callback):
    """A tqdm class that reports the finished-file fraction of a snapshot download."""
    import io

    from tqdm.std import tqdm

    class Reporter(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(file=io.StringIO(), disable=False, mininterval=0, miniters=1)
            super().__init__(*args, **kwargs)

        def display(self, msg=None, pos=None):
            if self.total:
                callback(self.n / self.total)
            return True

    return Reporter


def checkpoint(model_id=MODEL_ID, cache_dir=None) -> Path:
    path = Path(model_id).expanduser()
    if not path.is_dir():
        if model_id != MODEL_ID:
            raise ValueError("MODEL_ID must identify the ViMD checkpoint or its local directory")
        path = latest_snapshot(model_id, ASR_PATTERNS, cache_dir)
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("model_type") != "qwen3_asr"
        or config.get("quantization_config") is not None
        or path.name in {"q4", "q8"}
        or not any(path.glob("*.safetensors"))
    ):
        raise ValueError("Expected original ViMD root Safetensors checkpoint")
    return path


def intent_checkpoint(model_id: str, cache_dir=None, progress=None) -> str:
    path = Path(model_id).expanduser()
    if path.is_dir():
        return str(path)
    return str(latest_snapshot(model_id, INTENT_PATTERNS, cache_dir, progress))
