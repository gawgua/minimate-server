from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.asr import ASRService
from app.main import app


@pytest.fixture
def ws_reply():
    def receive(ws):
        audio = bytearray()
        while True:
            message = ws.receive()
            assert message["type"] == "websocket.send", message
            if "text" in message:
                return message["text"], bytes(audio)
            chunk = message["bytes"]
            assert 0 < len(chunk) <= 1600 and len(chunk) % 2 == 0
            audio.extend(chunk)

    return receive


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in [
        "HOST",
        "PORT",
        "MODEL_ID",
        "MODEL_DEVICE",
        "MODEL_DTYPE",
        "MAX_NEW_TOKENS",
        "SAMPLE_RATE",
        "MIN_AUDIO_SECONDS",
        "MAX_AUDIO_SECONDS",
        "ASR_LANGUAGE",
        "MAX_CONCURRENT_INFERENCES",
        "LOG_LEVEL",
        "HIGHPASS_HZ",
        "INTENT_DEFAULT_MODEL",
        "INTENT_TFIDF_DIR",
        "INTENT_ENABLED",
        "INTENT_MODEL_ID",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture
def service():
    model = Mock()
    model.transcribe.return_value = [SimpleNamespace(text="language Vietnamese<asr_text>Xin chào")]
    return ASRService(model, SimpleNamespace(inference_mode=nullcontext), "cpu")


@pytest.fixture
def client(monkeypatch, service):
    loader = Mock(return_value=service)
    monkeypatch.setattr(ASRService, "load", loader)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000)) as client:
        yield client
    loader.assert_called_once()
    assert service.model is None
    assert service.classifier is None
