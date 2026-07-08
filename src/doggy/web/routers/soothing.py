from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi import status as http_status

from doggy.core.config import Settings

# Starlette renamed HTTP_413_REQUEST_ENTITY_TOO_LARGE -> _CONTENT_TOO_LARGE (0.47);
# accept either (prefer the new name so current Starlette stays warning-free).
_HTTP_413 = getattr(http_status, "HTTP_413_CONTENT_TOO_LARGE", None) or getattr(
    http_status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413
)

# Calm audio the library lists and accepts as uploads.
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg"}
_CHUNK = 1024 * 1024          # stream uploads a MiB at a time (1 GB files won't fit RAM)
# Multipart framing (boundary lines + per-part headers) makes Content-Length a bit
# larger than the raw file bytes; allow this slack before rejecting on it up front.
_MULTIPART_OVERHEAD = 4096
_OVER_LIMIT = "That would go over the 1 GB limit. Delete a track first."


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    def _dir() -> Path:
        return Path(settings.soothing_dir)

    def _tracks(soothing: Path) -> list[Path]:
        if not soothing.is_dir():
            return []
        return sorted(
            (p for p in soothing.glob("*")
             if p.is_file() and not p.name.startswith(".")
             and p.suffix.lower() in _AUDIO_EXTS),
            key=lambda p: p.name,
        )

    @router.get("/api/soothing")
    def api_soothing() -> dict:
        tracks = _tracks(_dir())
        sizes = [(p.name, p.stat().st_size) for p in tracks]
        return {
            "tracks": [{"name": n, "size": s} for n, s in sizes],
            "total_bytes": sum(s for _, s in sizes),
            "limit_bytes": settings.soothing_limit_bytes,
        }

    @router.post("/api/soothing")
    async def api_upload_soothing(
        request: Request, file: UploadFile = File(...)
    ) -> dict:
        # Path(...).name strips any directory components → no path traversal.
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in _AUDIO_EXTS:
            raise HTTPException(
                status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="unsupported audio type")
        soothing = _dir()
        limit = settings.soothing_limit_bytes

        # A track of the same name is overwritten by os.replace at the end, so its
        # current size frees up — credit it back instead of double-counting the cap.
        existing = sum(p.stat().st_size for p in _tracks(soothing))
        target = soothing / name
        if target.is_file():
            existing -= target.stat().st_size

        # Early reject on the declared size. Starlette has already spooled the whole
        # multipart body by the time we run, so this can't stop that first spool — it
        # only spares us writing a doomed part file into the library.
        declared = request.headers.get("content-length")
        if (
            declared is not None
            and declared.isdigit()
            and int(declared) > (limit - existing) + _MULTIPART_OVERHEAD
        ):
            raise HTTPException(status_code=_HTTP_413, detail=_OVER_LIMIT)

        soothing.mkdir(parents=True, exist_ok=True)
        # Unique per request so concurrent uploads never clobber each other's temp
        # file; dot-prefixed so it's never listed as a track (see _tracks filter).
        part = soothing / f".upload.{uuid4().hex}.part"
        written = 0
        try:
            with part.open("wb") as out:
                while True:
                    chunk = await file.read(_CHUNK)
                    if not chunk:
                        break
                    if existing + written + len(chunk) > limit:
                        # Chunked uploads carry no Content-Length and so skip the early
                        # check above; abort mid-stream here. Residual limitation: an
                        # oversize chunked body still lands once in Starlette's spool
                        # before we reject it — accepted for a single-user appliance.
                        raise HTTPException(status_code=_HTTP_413, detail=_OVER_LIMIT)
                    out.write(chunk)
                    written += len(chunk)
            os.replace(part, target)
        finally:
            # On success os.replace already renamed the part away (no-op here); on any
            # abort or a failed replace this clears the leftover temp file.
            part.unlink(missing_ok=True)
        return {"name": name}

    @router.delete("/api/soothing/{name}")
    def api_delete_soothing(name: str) -> dict:
        # Path(name).name strips any directory components → no path traversal.
        path = _dir() / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="not found")
        path.unlink()
        return {"ok": True}

    return router
