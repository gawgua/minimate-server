import io
import struct
import wave
from contextlib import closing
from unittest.mock import Mock

import numpy as np
from fastapi.testclient import TestClient

from app.dashboard import HISTORY_LIMIT, LOG_LIMIT
from app.main import app
from app.protocol import MAX_LOG_CHARS

PCM = b"\x00\x80\xff\x7f" * 3200


def test_dashboard_pairs_original_pcm_and_transcript(client, service, ws_reply):
    page = client.get("/dashboard")
    assert page.status_code == 200
    assert "Minimate recordings" in page.text
    assert client.get("/dashboard/recordings").json() == []

    service.classifier = Mock(threshold=None)
    intent = {"label": "check_focus_session", "score": 0.92}
    service.classifier.classify.return_value = {"id": 0, **intent}
    clean = bytes(12800)
    noisy = clean[:1600] + struct.pack("<h", 32767) + clean[1602:]
    with client.websocket_connect("/ws/asr") as ws:
        ws.send_bytes(noisy)
        ws.send_text("END")
        assert ws_reply(ws) == ("0", clean)
        ws.send_text("END")
        assert ws.receive_text() == "-1"
    samples, rate = service.model.transcribe.call_args.kwargs["audio"]
    assert rate == 16000
    np.testing.assert_array_equal(samples, np.zeros(6400, dtype=np.float32))
    history = client.get("/dashboard/recordings")
    assert history.headers["cache-control"] == "no-store"
    (clip,) = history.json()
    assert clip["result"]["text"] == "Xin chào"
    assert clip["result"]["intent"] == {**intent, "model": "bamibert", "accepted": True}
    assert "intent_id" not in clip["result"]
    assert clip["duration"] == 0.4
    for suffix, expected in (("", noisy), ("?filtered=true", clean)):
        response = client.get(f"/dashboard/recordings/{clip['id']}.wav{suffix}")
        assert response.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(response.content), "rb") as audio:
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
                1,
                2,
                16000,
            )
            assert audio.readframes(audio.getnframes()) == expected


def test_failed_recordings_age_out(client, service, ws_reply):
    service.model.transcribe.side_effect = RuntimeError("private exception")
    with client.websocket_connect("/ws/asr") as ws:
        ws.send_bytes(PCM)
        ws.send_text("END")
        assert ws.receive_text() == "-1"
        (failed,) = client.get("/dashboard/recordings").json()
        assert failed["error"] == "Inference failed" and failed["result"] is None
        failed_audio = f"/dashboard/recordings/{failed['id']}.wav"
        assert client.get(failed_audio).status_code == 200

        service.model.transcribe.side_effect = None
        for _ in range(HISTORY_LIMIT):
            ws.send_bytes(PCM)
            ws.send_text("END")
            assert ws_reply(ws)[0] == "-1"

    assert len(client.get("/dashboard/recordings").json()) == HISTORY_LIMIT
    assert client.get(failed_audio).status_code == 404


def test_dashboard_routes_are_local_only(client):
    assert client.get("/dashboard", headers={"Host": "example.com"}).status_code == 403
    remote = TestClient(app, base_url="http://localhost", client=("192.168.1.20", 50000))
    with closing(remote):
        for path in (
            "/dashboard",
            "/dashboard/recordings",
            "/dashboard/logs",
            "/dashboard/recordings/sample.wav",
        ):
            assert remote.get(path).status_code == 403


def test_device_logs_reach_dashboard_without_touching_recording(client, ws_reply):
    assert client.get("/dashboard/logs").json() == []
    with client.websocket_connect("/ws/asr") as ws:
        ws.send_text("LOG [1.250s] ASR connected")
        ws.send_bytes(PCM[:6400])
        ws.send_text("LOG [2.000s] Recording")
        ws.send_bytes(PCM[6400:])
        ws.send_text("LOG " + "x" * 1000)
        ws.send_text("END")
        text, audio = ws_reply(ws)
        assert text == "-1" and len(audio) == len(PCM)
    logs = client.get("/dashboard/logs")
    assert logs.headers["cache-control"] == "no-store"
    first, second, long = logs.json()
    assert first["message"] == "[1.250s] ASR connected"
    assert second["message"] == "[2.000s] Recording"
    assert long["message"] == "x" * MAX_LOG_CHARS
    assert first["device"] == "127.0.0.1"
    assert first["seq"] < second["seq"] < long["seq"]
    assert client.get("/dashboard/recordings").json()[0]["duration"] == len(PCM) / 32000


def test_device_logs_are_capped(client):
    with client.websocket_connect("/ws/asr") as ws:
        for index in range(LOG_LIMIT + 5):
            ws.send_text(f"LOG line {index}")
        ws.send_text("END")
        assert ws.receive_text() == "-1"
    logs = client.get("/dashboard/logs").json()
    assert len(logs) == LOG_LIMIT and logs[-1]["message"] == f"line {LOG_LIMIT + 4}"
