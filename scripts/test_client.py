import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

from websockets.asyncio.client import connect

DEFAULT_SERVER = "ws://127.0.0.1:8000/ws/asr"


def read_wav(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (
            16000,
            1,
            2,
            "NONE",
        ):
            raise ValueError("Expected uncompressed 16 kHz mono PCM16 WAV")
        count = wav.getnframes()
        pcm = wav.readframes(count)
        if not count or len(pcm) != count * 2:
            raise ValueError("Empty or truncated WAV")
        return pcm


async def request(path, server, chunk_ms=100, realtime=False, filtered_wav=None):
    if chunk_ms <= 0:
        raise ValueError("--chunk-ms must be positive")
    pcm = read_wav(path)
    size = chunk_ms * 32
    received_bytes = 0
    filtered = bytearray() if filtered_wav is not None else None
    async with connect(server, compression=None, subprotocols=["asr-json"]) as ws:
        start = time.perf_counter()
        for pos in range(0, len(pcm), size):
            chunk = pcm[pos : pos + size]
            await ws.send(chunk)
            if realtime:
                await asyncio.sleep(len(chunk) / 32000)
        end = time.perf_counter()
        await ws.send("END")
        frame = await ws.recv()
        while isinstance(frame, bytes):
            received_bytes += len(frame)
            if received_bytes > len(pcm):
                raise ValueError("Server returned too much audio")
            if filtered is not None:
                filtered.extend(frame)
            frame = await ws.recv()
        reply = json.loads(frame)
        stop = time.perf_counter()
    if "error" in reply:
        raise ValueError(reply["error"])
    if reply.get("type") != "transcription":
        raise ValueError("Unexpected reply")
    if received_bytes != len(pcm):
        raise ValueError("Server returned incomplete audio")
    if filtered_wav is not None:
        with wave.open(str(filtered_wav), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", ""))
            output.writeframes(filtered)
    return {**reply, "request_seconds": stop - start, "end_to_reply_seconds": stop - end}


def main():
    parser = argparse.ArgumentParser(description="Transcribe a PCM16 WAV over WebSocket")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--filtered-wav", type=Path, help="Save the returned filtered audio")
    args = parser.parse_args()
    try:
        result = asyncio.run(
            request(args.wav, args.server, args.chunk_ms, args.realtime, args.filtered_wav)
        )
        print(result["text"])
        if intent := result.get("intent"):
            print(f"Intent {intent['label']} ({intent['score']:.3f})")
        print(
            f"Inference {result['inference_seconds']:.3f}s, END to reply {result['end_to_reply_seconds']:.3f}s"
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report any failure cleanly
        sys.exit(f"Transcription failed: {exc}")


if __name__ == "__main__":
    main()
