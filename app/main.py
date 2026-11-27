import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from itertools import count

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .asr import ASRService
from .audio import clean_pcm16, pcm16_to_float32
from .config import MODEL_ID, RATE, Settings
from .dashboard import HISTORY_LIMIT, LOG_LIMIT, Clip, DeviceLog
from .dashboard import router as dashboard_router
from .logging_config import configure_logging
from .protocol import DIAGNOSTIC_SUBPROTOCOL, LOG_PREFIX, MAX_LOG_CHARS, NO_INTENT, Recording

log = logging.getLogger(__name__)
RETURN_CHUNK_BYTES = 1600


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = Settings.load()
        configure_logging(settings.log_level)
        app.state.settings = settings
        service = await asyncio.to_thread(ASRService.load, settings)
    except Exception:
        log.exception("ASR startup failed")
        raise
    app.state.service = service
    app.state.recordings.clear()
    app.state.device_logs.clear()
    try:
        yield
    finally:
        await service.close()
        app.state.recordings.clear()
        app.state.device_logs.clear()
        app.state.service = None


app = FastAPI(title="ESP32 Vietnamese ASR", lifespan=lifespan)
app.state.service = None
app.state.recordings = deque(maxlen=HISTORY_LIMIT)
app.state.device_logs = deque(maxlen=LOG_LIMIT)
log_seq = count(1)
app.include_router(dashboard_router)


@app.get("/")
async def info():
    return {"service": app.title, "websocket": "/ws/asr", "sample_rate": RATE}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": getattr(getattr(app.state, "settings", None), "model_id", MODEL_ID),
    }


@app.get("/ready")
async def ready():
    service = app.state.service
    available = service is not None and service.ready
    return JSONResponse({"ready": available}, status_code=200 if available else 503)


@app.websocket("/ws/asr")
async def asr_socket(ws: WebSocket):
    diagnostics = DIAGNOSTIC_SUBPROTOCOL in ws.scope.get("subprotocols", ())
    await ws.accept(subprotocol=DIAGNOSTIC_SUBPROTOCOL if diagnostics else None)
    service = app.state.service
    if service is None or not service.ready:
        await ws.close(code=1013)
        return
    settings = app.state.settings
    device = ws.client.host if ws.client else "unknown"
    recording = Recording(settings.min_samples, settings.max_samples)

    async def send_error(message: str):
        if diagnostics:
            await ws.send_json({"error": message})
        else:
            await ws.send_text(str(NO_INTENT))

    log.info("Client connected")
    try:
        while service.ready:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            try:
                if msg.get("bytes") is not None:
                    recording.append(msg["bytes"])
                    continue
                text = msg.get("text")
                if text is not None and text.startswith(LOG_PREFIX):
                    # Device logs never touch the current recording.
                    message = text[len(LOG_PREFIX) :][:MAX_LOG_CHARS]
                    app.state.device_logs.append(DeviceLog(next(log_seq), device, message))
                    log.info("Device %s: %s", device, message)
                    continue
                if text != "END":
                    continue
                pcm = recording.finish()
            except ValueError as exc:
                await send_error(str(exc))
                continue
            if pcm is None:
                continue
            filtered_pcm = await asyncio.to_thread(clean_pcm16, pcm, settings.highpass_hz)
            duration = len(pcm) / (2 * RATE)
            clip = Clip(pcm, filtered_pcm)
            app.state.recordings.append(clip)
            log.info("Recording received: %.3fs", duration)
            if log.isEnabledFor(logging.DEBUG):
                original = pcm16_to_float32(pcm)
                magnitudes = np.abs(original)
                peak = np.max(magnitudes)
                signal_rms = np.std(original)
                clipped = np.count_nonzero(magnitudes >= 32767 / 32768)
                log.debug(
                    "PCM levels peak=%.6f signal_rms=%.6f clipped=%d/%d",
                    peak,
                    signal_rms,
                    clipped,
                    len(original),
                )
            try:
                result = await service.transcribe(pcm16_to_float32(filtered_pcm))
            except asyncio.CancelledError:
                clip.error = "Transcription interrupted"
                raise
            except Exception:
                clip.error = "Inference failed"
                log.exception("Inference failed")
                await send_error(clip.error)
                continue
            intent_id = result.pop("intent_id")
            clip.result = {
                "type": "transcription",
                "language": "vi",
                "duration": duration,
                **result,
            }
            for start in range(0, len(filtered_pcm), RETURN_CHUNK_BYTES):
                await ws.send_bytes(filtered_pcm[start : start + RETURN_CHUNK_BYTES])
            if diagnostics:
                await ws.send_json(clip.result)
            else:
                await ws.send_text(str(intent_id))
    except WebSocketDisconnect:
        pass
    finally:
        log.info("Client disconnected")
