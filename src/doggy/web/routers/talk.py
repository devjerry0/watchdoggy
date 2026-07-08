from __future__ import annotations

import logging
import shutil
import subprocess
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from doggy.core.runtime import RuntimeSettings

log = logging.getLogger("doggy")

# One talker at a time: the speaker is a single mono pipe, so a second caller is
# turned away rather than mixed in. Module-level because the appliance runs one
# process and the lock must outlive any single websocket.
_busy = threading.Lock()


def _spawn_player(volume: float) -> subprocess.Popen | None:
    # Raw PCM straight into PipeWire; on a dev Mac there is no pw-cat, so we
    # return None and the handler discards audio (the UI still works).
    exe = shutil.which("pw-cat") or shutil.which("pw-play")
    if not exe:
        log.info("push-to-talk: no pw-cat on this host; discarding audio")
        return None
    # --raw is REQUIRED: without it pw-cat opens the stream through libsndfile,
    # which needs a file header (WAV/FLAC) and rejects the browser's headerless
    # PCM with "Format not recognised" -- so no voice is ever heard.
    return subprocess.Popen(
        [exe, "--playback", "--raw", "--volume", f"{max(0.0, min(1.0, volume)):.3f}",
         "--rate", "16000", "--channels", "1", "--format", "s16", "-"],
        stdin=subprocess.PIPE)


def build_router(runtime: RuntimeSettings) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/talk")
    async def talk(ws: WebSocket) -> None:
        if not _busy.acquire(blocking=False):
            # Accept first so the browser gets a clean close code rather than a
            # bare handshake failure it can't distinguish from a network fault.
            await ws.accept()
            await ws.close(code=1013)  # try again later: someone is talking
            return
        # Everything past here must reach the finally so the lock is released:
        # spawning inside the try (Popen can fail) and any pipe write failing
        # (the speaker dropping) must not strand the lock and brick push-to-talk.
        proc = None
        try:
            proc = _spawn_player(runtime.get().talk_volume)
            await ws.accept()
            while True:
                data = await ws.receive_bytes()
                if proc and proc.stdin:
                    # Blocking write on the event loop: frames are tiny (~8 KB)
                    # and go to a local pipe, so it never meaningfully stalls.
                    try:
                        proc.stdin.write(data)
                        proc.stdin.flush()
                    except OSError:
                        # The player went away mid-stream (e.g. the Bluetooth
                        # speaker dropped): end the session, don't propagate.
                        log.info("push-to-talk: player went away mid-stream")
                        break
        except WebSocketDisconnect:
            pass
        finally:
            if proc is not None:
                # close() re-flushes buffered bytes and can itself raise on a
                # dead pipe; swallow that so terminate() and release() still run.
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except OSError:
                    pass
                try:
                    proc.terminate()
                except OSError:
                    pass
            _busy.release()

    return router
