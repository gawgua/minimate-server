from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .audio import pcm16_to_wav
from .config import RATE

HISTORY_LIMIT = 10
LOG_LIMIT = 200
NO_CACHE = {"Cache-Control": "no-store"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class Clip:
    pcm: bytes
    filtered_pcm: bytes
    id: str = field(default_factory=lambda: uuid4().hex)
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    result: dict | None = None
    error: str | None = None

    def summary(self):
        return {
            "id": self.id,
            "recorded_at": self.recorded_at,
            "duration": len(self.pcm) / (2 * RATE),
            "error": self.error,
            "result": self.result,
        }


@dataclass
class DeviceLog:
    seq: int
    device: str
    message: str
    received_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def require_local(request: Request):
    try:
        address = ip_address(request.client.host) if request.client else None
    except ValueError:
        address = None
    if address is None or not address.is_loopback or request.url.hostname not in LOCAL_HOSTS:
        raise HTTPException(403, "Open the recording dashboard on the server laptop")


router = APIRouter(prefix="/dashboard", dependencies=[Depends(require_local)])


@router.get("", include_in_schema=False)
async def dashboard():
    page = Path(__file__).with_name("dashboard.html")
    return FileResponse(page, media_type="text/html", headers=NO_CACHE)


@router.get("/recordings")
async def recordings(request: Request):
    return JSONResponse(
        [clip.summary() for clip in reversed(request.app.state.recordings)], headers=NO_CACHE
    )


@router.get("/logs")
async def device_logs(request: Request):
    return JSONResponse([vars(entry) for entry in request.app.state.device_logs], headers=NO_CACHE)


def intent_switcher(request: Request):
    service = request.app.state.service
    if service is None:
        raise HTTPException(503, "Server is not ready")
    return service.intents


@router.get("/intent-model")
async def intent_model(request: Request):
    return JSONResponse(intent_switcher(request).status(), headers=NO_CACHE)


@router.post("/intent-model")
async def choose_intent_model(request: Request):
    # A JSON body forces a CORS preflight, and the Origin check refuses other sites, so a page
    # elsewhere in the same browser cannot switch models through 127.0.0.1.
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "Send JSON")
    origin = request.headers.get("origin")
    if origin is not None and urlsplit(origin).hostname not in LOCAL_HOSTS:
        raise HTTPException(403, "Switch models from the local dashboard")
    try:
        key = (await request.json())["model"]
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, 'Expected {"model": "<key>"}') from None
    switcher = intent_switcher(request)
    status = switcher.status()
    if status["active"] == key and status["loading"] is None:
        return JSONResponse(status, headers=NO_CACHE)
    try:
        started = switcher.start(key)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if not started:
        raise HTTPException(409, "Another intent model is loading. Wait for it to finish.")
    return JSONResponse(switcher.status(), status_code=202, headers=NO_CACHE)


@router.get("/recordings/{clip_id}.wav")
async def recording_audio(clip_id: str, request: Request, filtered: bool = False):
    for clip in request.app.state.recordings:
        if clip.id == clip_id:
            pcm = clip.filtered_pcm if filtered else clip.pcm
            return Response(pcm16_to_wav(pcm), media_type="audio/wav", headers=NO_CACHE)
    raise HTTPException(404, "Recording expired or not found")
