import asyncio
import csv
import socket
import threading
import time
import wave
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import uvicorn

from app.asr import ASRService
from app.main import app
from scripts import benchmark, test_client


def write_wav(path, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", ""))
        handle.writeframes(bytes(12800))


def test_invalid_wav(tmp_path):
    path = tmp_path / "bad.wav"
    write_wav(path, rate=8000)
    with pytest.raises(ValueError, match="16 kHz"):
        test_client.read_wav(path)


def test_benchmark_asgi(client, monkeypatch, tmp_path):
    @asynccontextmanager
    async def connect(url, **kwargs):
        with client.websocket_connect("/ws/asr", subprotocols=kwargs["subprotocols"]) as session:

            async def send(data):
                if isinstance(data, bytes):
                    session.send_bytes(data)
                else:
                    session.send_text(data)

            async def recv():
                message = session.receive()
                return message["text"] if "text" in message else message["bytes"]

            yield SimpleNamespace(send=send, recv=recv)

    monkeypatch.setattr(test_client, "connect", connect)
    write_wav(tmp_path / "speech.wav")
    output = tmp_path / "result.csv"
    args = SimpleNamespace(input=tmp_path, server="ws://test/ws/asr", csv=output, references=None)
    asyncio.run(benchmark.benchmark(args))
    with output.open() as handle:
        (row,) = csv.DictReader(handle)
    assert row["text"] == "Xin chào" and row["file"] == "speech.wav"
    assert float(row["rtf"]) == float(row["inference_seconds"]) / 0.4
    filtered = tmp_path / "filtered.wav"
    asyncio.run(test_client.request(tmp_path / "speech.wav", args.server, filtered_wav=filtered))
    assert test_client.read_wav(filtered) == bytes(12800)


def test_client_loopback(service, monkeypatch, tmp_path):
    try:
        listener = socket.socket()
    except PermissionError:
        pytest.skip("Host denies TCP sockets")
    with listener:
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Host denies TCP listeners")
        monkeypatch.setattr(ASRService, "load", lambda settings: service)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", loop="asyncio"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            path = tmp_path / "speech.wav"
            write_wav(path)
            result = asyncio.run(test_client.request(path, f"ws://127.0.0.1:{port}/ws/asr"))
            assert result["text"] == "Xin chào"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
        assert not thread.is_alive()
