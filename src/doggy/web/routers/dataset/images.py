"""Serving dataset frames and their filmstrip thumbnails."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status as http_status
from fastapi.responses import FileResponse

from doggy.core.config import Settings
from doggy.web.routers.dataset.sidecars import not_found

THUMB_WIDTH = 160
THUMB_JPEG_QUALITY = 70


def _write_thumb(source: Path, thumb: Path) -> bool:
    import cv2  # deferred: OpenCV import is heavy; only pay it on a miss
    image = cv2.imread(str(source))
    if image is None:
        return False
    height = max(1, round(image.shape[0] * THUMB_WIDTH / image.shape[1]))
    small = cv2.resize(image, (THUMB_WIDTH, height),
                       interpolation=cv2.INTER_AREA)
    thumb.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(thumb), small, [cv2.IMWRITE_JPEG_QUALITY,
                                    THUMB_JPEG_QUALITY])
    return True


def build_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/dataset/thumb/{name}")
    def dataset_thumb(name: str) -> FileResponse:
        """Small cached thumbnail for the filmstrip. Full frames are ~100KB
        each; a 400-frame strip of them is megabytes through a Pi -- thumbs
        are ~5KB, generated once and cached beside the dataset."""
        safe = Path(name).name  # traversal guard
        source = Path(settings.dataset_dir) / safe
        if not safe.startswith("sample_") or not safe.endswith(".jpg"):
            raise not_found()
        thumb = Path(settings.dataset_dir) / "thumbs" / safe
        if not source.is_file():
            thumb.unlink(missing_ok=True)  # source pruned: drop the stale thumb
            raise not_found()
        if not thumb.is_file() and not _write_thumb(source, thumb):
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND,
                                detail="unreadable")
        return FileResponse(thumb)

    @router.get("/dataset/{name}")
    def dataset_image(name: str) -> FileResponse:
        # Path(name).name strips any directory components -> no path traversal.
        safe = Path(name).name
        path = Path(settings.dataset_dir) / safe
        if not safe.startswith("sample_") or not safe.endswith(".jpg") \
                or not path.is_file():
            raise not_found()
        return FileResponse(path)

    return router
