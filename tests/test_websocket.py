import json
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.asr import ASRService
from app.config import MODEL_ID
from app.main import app

PCM = bytes(12800)


def test_http(client):
    assert client.get("/").json()["websocket"] == "/ws/asr"
    assert client.get("/health").json() == {"status": "ok", "model": MODEL_ID}
    assert client.get("/ready").json() == {"ready": True}


def test_unavailable(monkeypatch):
    with closing(TestClient(app)) as client:
        assert client.get("/ready").status_code == 503

    def fail(settings):
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(ASRService, "load", fail)
    with pytest.raises(RuntimeError, match="bad checkpoint"), TestClient(app):
        pass


def test_interleaved(client, service, ws_reply):
    with (
        client.websocket_connect("/ws/asr", subprotocols=["asr-json"]) as a,
        client.websocket_connect("/ws/asr", subprotocols=["asr-json"]) as b,
    ):
        a.send_bytes(PCM[:6000])
        b.send_bytes(bytes(32000))
        b.send_text("END")
        text, audio = ws_reply(b)
        assert json.loads(text)["duration"] == 1 and audio == bytes(32000)
        a.send_text("unknown")
        a.send_bytes(PCM[6000:])
        a.send_text("END")
        text, audio = ws_reply(a)
        reply = json.loads(text)
        assert audio == PCM
        assert (reply["text"], reply["type"], reply["language"], reply["duration"]) == (
            "Xin chào",
            "transcription",
            "vi",
            0.4,
        )
        assert reply["intent"] is None
        a.send_bytes(PCM)
        a.send_text("END")
        assert json.loads(ws_reply(a)[0])["text"] == "Xin chào"
        b.send_bytes(PCM)
    assert service.model.transcribe.call_count == 3


@pytest.mark.parametrize(
    ("bad", "message"),
    [(b"x", "PCM16 requires even byte count"), (bytes(640002), "Audio too long")],
)
def test_recover(client, bad, message, ws_reply):
    with client.websocket_connect("/ws/asr", subprotocols=["asr-json"]) as ws:
        ws.send_bytes(bad)
        assert ws.receive_json() == {"error": message}
        ws.send_bytes(PCM)
        ws.send_text("END")
        ws.send_text("END")
        assert ws.receive_json() == {"error": "Audio too short"}
        ws.send_bytes(PCM)
        ws.send_text("END")
        assert json.loads(ws_reply(ws)[0])["duration"] == 0.4


def test_failed_inference(client, service, ws_reply):
    with client.websocket_connect("/ws/asr", subprotocols=["asr-json"]) as ws:
        service.model.transcribe.side_effect = MemoryError("private error details")
        ws.send_bytes(PCM)
        ws.send_text("END")
        assert ws.receive_json() == {"error": "Inference failed"}
        service.model.transcribe.side_effect = None
        ws.send_bytes(PCM)
        ws.send_text("END")
        assert json.loads(ws_reply(ws)[0])["text"] == "Xin chào"


def test_intent_classifies_clean_transcript_and_recovers(client, service, ws_reply):
    prediction = {"label": "greeting", "score": 0.97}
    classifier = Mock(threshold=None)
    classifier.classify.side_effect = [
        RuntimeError("private classifier error"),
        {"id": 2, **prediction},
    ]
    service.classifier = classifier
    with client.websocket_connect("/ws/asr", subprotocols=["asr-json"]) as ws:
        for expected in (None, {**prediction, "model": "bamibert", "accepted": True}):
            ws.send_bytes(PCM)
            ws.send_text("END")
            reply = json.loads(ws_reply(ws)[0])
            assert reply["text"] == "Xin chào"
            assert reply["intent"] == expected
            assert "error" not in reply

        service.model.transcribe.return_value = [SimpleNamespace(text=" ")]
        ws.send_bytes(PCM)
        ws.send_text("END")
        reply = json.loads(ws_reply(ws)[0])
        assert reply["text"] == "" and reply["intent"] is None

    assert classifier.classify.call_count == 2
    assert all(call.args == ("Xin chào",) for call in classifier.classify.call_args_list)
