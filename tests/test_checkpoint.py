import json
import sys
from contextlib import nullcontext
from fnmatch import fnmatch
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from huggingface_hub.errors import OfflineModeIsEnabled

from app.checkpoint import ASR_PATTERNS, checkpoint, intent_checkpoint, latest_snapshot
from app.config import INTENT_MODEL_ID, MODEL_ID
from app.intent import IntentClassifier
from app.main import app


@pytest.fixture
def hub(monkeypatch, tmp_path):
    import huggingface_hub

    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))
    (snapshot / "model.safetensors").touch()
    info = Mock(return_value=SimpleNamespace(sha="abc123"))
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=Mock(model_info=info)))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return SimpleNamespace(path=snapshot, info=info, download=download)


def http_error(status):
    from huggingface_hub.errors import HfHubHTTPError

    return HfHubHTTPError("Hub error", response=Mock(status_code=status, headers={}))


def test_root_download(hub, caplog):
    caplog.set_level("INFO", "app.checkpoint")
    assert checkpoint(cache_dir="cache") == hub.path
    hub.info.assert_called_once_with(MODEL_ID)
    args = hub.download.call_args.kwargs
    assert args["repo_id"] == MODEL_ID and args["cache_dir"] == "cache"
    assert "revision" not in args and "local_files_only" not in args
    assert f"Using {MODEL_ID} at commit abc123" in caplog.text
    for name in (
        "model-00001-of-00002.safetensors",
        "model.safetensors.index.json",
        "merges.txt",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        assert any(fnmatch(name, pattern) for pattern in args["allow_patterns"])
    for name in ("q4/model.safetensors", "q8/config.json"):
        assert any(fnmatch(name, pattern) for pattern in args["ignore_patterns"])


@pytest.mark.parametrize(
    "error",
    [
        lambda: requests.ConnectionError("no route"),
        lambda: requests.Timeout("slow"),
        lambda: http_error(503),
        lambda: OfflineModeIsEnabled("offline"),
    ],
)
def test_unreachable_hub_uses_cache(hub, error, caplog):
    caplog.set_level("INFO", "app.checkpoint")
    hub.info.side_effect = error()
    assert latest_snapshot(MODEL_ID, ASR_PATTERNS) == hub.path
    args = hub.download.call_args.kwargs
    assert args["local_files_only"] is True and "revision" not in args
    assert "cached" in caplog.text


@pytest.mark.parametrize("status", [401, 404])
def test_hub_config_errors_abort(hub, status):
    hub.info.side_effect = http_error(status)
    with pytest.raises(requests.HTTPError):
        latest_snapshot(MODEL_ID, ASR_PATTERNS)
    hub.download.assert_not_called()


def test_intent_checkpoint(hub, tmp_path):
    assert intent_checkpoint(str(tmp_path)) == str(tmp_path)
    hub.info.assert_not_called()
    assert intent_checkpoint(INTENT_MODEL_ID) == str(hub.path)
    args = hub.download.call_args.kwargs
    assert args["repo_id"] == INTENT_MODEL_ID and "revision" not in args
    for name in ("config.json", "model.safetensors", "tokenizer.json", "special_tokens_map.json"):
        assert any(fnmatch(name, pattern) for pattern in args["allow_patterns"])
    for name in ("training_args.bin", "bamibert-intents-finetune.ipynb"):
        assert not any(fnmatch(name, pattern) for pattern in args["allow_patterns"])


def test_quantized_config(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr","quantization_config":{}}')
    (tmp_path / "model.safetensors").touch()
    with pytest.raises(ValueError, match="original"):
        checkpoint(str(tmp_path))


@pytest.mark.parametrize("enabled", [True, False])
def test_original_loader(monkeypatch, tmp_path, enabled, ws_reply):
    if not enabled:
        monkeypatch.setenv("INTENT_ENABLED", "false")
    torch = Mock()
    torch.inference_mode = nullcontext
    torch.cuda.is_available.return_value = False
    loader = Mock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=loader))
    resolve = Mock(return_value=tmp_path)
    monkeypatch.setattr("app.asr.checkpoint", resolve)
    load_intent = Mock()
    monkeypatch.setattr(IntentClassifier, "load", load_intent)
    model = loader.from_pretrained.return_value
    model.transcribe.return_value = [SimpleNamespace(text="Xin chào")]
    load_intent.return_value.threshold = None
    load_intent.return_value.classify.return_value = {"id": 2, "label": "greeting", "score": 0.9}
    with TestClient(app) as client:
        assert client.get("/ready").json() == {"ready": True}
        assert app.state.service.model is model
        model.transcribe.assert_called_once()
        audio, rate = model.transcribe.call_args.kwargs["audio"]
        assert rate == 16000 and audio.shape == (16000,) and audio.dtype == np.float32
        assert not audio.any()
        if enabled:
            load_intent.return_value.classify.assert_called_once_with("Xin chào.")
        with client.websocket_connect("/ws/asr", subprotocols=["asr-json"]) as ws:
            for _ in range(2):
                ws.send_bytes(bytes(12800))
                ws.send_text("END")
                assert json.loads(ws_reply(ws)[0])["text"] == "Xin chào"
        assert model.transcribe.call_count == 3
    if enabled:
        load_intent.assert_called_once()
        assert load_intent.call_args.args[0] == INTENT_MODEL_ID
        assert load_intent.return_value.classify.call_count == 3
    else:
        load_intent.assert_not_called()
    resolve.assert_called_once_with(MODEL_ID)
    loader.from_pretrained.assert_called_once_with(
        str(tmp_path),
        device_map="cpu",
        dtype=torch.float32,
        max_inference_batch_size=1,
        max_new_tokens=256,
        use_safetensors=True,
    )


def test_failed_intent_load_aborts_startup(monkeypatch):
    torch = Mock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=Mock()))
    failure = OSError("missing model files")
    monkeypatch.setattr(IntentClassifier, "load", Mock(side_effect=failure))
    with (
        pytest.raises(RuntimeError, match="Could not load intent classifier") as caught,
        TestClient(app),
    ):
        pytest.fail("App should not be ready")
    assert caught.value.__cause__ is failure
