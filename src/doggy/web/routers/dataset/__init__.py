"""Dataset routes, split by concern and composed behind one build_router:

- `sidecars`: the shared vocabulary (verdicts, filters, exam hash) and
  traversal-guarded sidecar access every route uses
- `browse`:   read-only views (stats, filmstrip, single frame, queue),
  served from the shared SidecarIndex
- `labeling`: every write path (verdicts, boxes, prelabels, auto-labels,
  disputes, the catch log's "Not a dog" tap)
- `images`:   frame + thumbnail file serving
"""
from __future__ import annotations

from fastapi import APIRouter

from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.web.routers.dataset import browse, images, labeling
from doggy.web.sidecar_index import SidecarIndex


def build_router(settings: Settings, event_store: EventStore,
                 index: SidecarIndex) -> APIRouter:
    router = APIRouter()
    router.include_router(browse.build_router(settings, index))
    router.include_router(labeling.build_router(settings, event_store))
    router.include_router(images.build_router(settings))
    return router
