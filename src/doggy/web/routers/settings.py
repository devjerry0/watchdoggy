from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi import status as http_status
from pydantic import ValidationError

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.core.settings_store import read_changelog

# Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY -> _CONTENT (0.47); accept either
# (prefer the new name so current Starlette doesn't emit a deprecation warning).
_HTTP_422 = getattr(http_status, "HTTP_422_UNPROCESSABLE_CONTENT", None) or getattr(
    http_status, "HTTP_422_UNPROCESSABLE_ENTITY", 422
)


def build_router(runtime: RuntimeSettings,
                 save_env: Callable[..., object]) -> APIRouter:
    """``save_env(tunable, changed_by=..., old=...)`` persists tunables --
    settings_store.save_tunables in production (settings.json + change log);
    the keyword name is historical, from when this rewrote .env."""
    router = APIRouter()

    @router.patch("/api/settings")
    def api_patch(patch: dict, request: Request) -> dict:
        current = runtime.get()
        merged = {**current.model_dump(), **patch}
        try:
            updated = TunableSettings(**merged)
        except ValidationError as exc:
            raise HTTPException(status_code=_HTTP_422, detail=str(exc)) from exc
        runtime.update(updated)
        # Persist immediately: the appliance restarts ITSELF now (self-
        # updates), and a live-only change was silently reverted by the next
        # restart. A toggle the user flipped must survive the machine's own
        # maintenance. Attributed to the client so "who slid certainty to
        # 0.95" is answerable (Sep 2026: it wasn't).
        who = request.client.host if request.client else "unknown"
        save_env(updated, changed_by=who, old=current)
        return updated.model_dump(mode="json")

    @router.post("/api/settings/save")
    def api_save(request: Request) -> dict:
        # Kept for compatibility (the dashboard still calls it); every patch
        # already persisted itself, so this logs nothing new (old == new).
        current = runtime.get()
        who = request.client.host if request.client else "unknown"
        save_env(current, changed_by=who, old=current)
        return {"ok": True}

    @router.get("/api/settings/history")
    def api_history(limit: int = 50) -> list[dict]:
        """Recent settings changes, newest first: ts, key, old, new, by."""
        return read_changelog(limit=max(1, min(limit, 500)))

    return router
