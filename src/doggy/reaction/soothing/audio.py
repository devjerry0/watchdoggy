"""Platform audio adapter for soothing playback: the library listing, the
player process (pw-play/pw-cat on the Pi, afplay on macOS), and live
PipeWire volume. Best effort everywhere: a missing tool degrades, never
raises."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

log = logging.getLogger("doggy")

SOOTHING_EXTS = {".mp3", ".wav", ".flac", ".ogg"}

Spawn = Callable[[Path, float], "subprocess.Popen | None"]
# Apply a volume [0,1] to a running player's audio stream in place; returns True
# if applied live (no restart needed), False if the caller must re-spawn.
SetVolume = Callable[["subprocess.Popen", float], bool]

# The stream node registers a beat after the player process starts: retry
# the pw-dump lookup a few times before giving up on live volume.
PW_NODE_RETRIES = 5
PW_NODE_RETRY_WAIT_S = 0.2
PW_TOOL_TIMEOUT_S = 3


def list_tracks(library_dir: Path) -> list[Path]:
    if not library_dir.is_dir():
        return []
    # Filter by audio extension: this also drops the upload's .upload.part
    # temp file (Path.glob does match dotfiles).
    return sorted(
        (p for p in library_dir.glob("*")
         if p.is_file() and p.suffix.lower() in SOOTHING_EXTS),
        key=lambda p: p.name,
    )


class AudioBackend:
    """Spawns audio players and applies live volume for the soothing loop."""

    def __init__(self, wait: Callable[[float], bool]) -> None:
        # Event.wait-shaped: True means shutdown, cutting retries short.
        self._wait = wait
        self._warned_no_player = False

    def _command(self, path: Path, volume: float) -> list[str] | None:
        # Backend fallback pw-play -> pw-cat on the Pi (so a Bluetooth sink
        # works), afplay on macOS. Unlike CommandAlerter (pw-play -> paplay ->
        # aplay) we deliberately do NOT fall back to paplay/aplay: those decode
        # only PCM/WAV and the soothing library is mostly mp3, which pw-play and
        # pw-cat decode on-device. pw-play and pw-cat are the same PipeWire
        # binary but bare pw-cat needs an explicit --playback ("one of the
        # playback/record options must be provided"); pw-play defaults to it.
        player = shutil.which("pw-play") or shutil.which("pw-cat")
        if player:
            cmd = [player, "--volume", str(volume)]
            if Path(player).name == "pw-cat":
                cmd.append("--playback")
            return [*cmd, str(path)]
        if sys.platform == "darwin":
            return ["afplay", "-v", str(volume), str(path)]
        if not self._warned_no_player:
            log.info("soothing: no audio player (pw-play/pw-cat/afplay) found; "
                     "soothing playback idle")
            self._warned_no_player = True
        return None

    def spawn(self, path: Path, volume: float) -> subprocess.Popen | None:
        cmd = self._command(path, volume)
        if cmd is None:
            return None
        try:
            return subprocess.Popen(cmd)
        except OSError:
            log.info("soothing: failed to launch %s", cmd[0])
            return None

    def set_live_volume(self, proc: subprocess.Popen, volume: float) -> bool:
        """Set the running player's PipeWire stream volume in place, so a
        loudness change is heard without restarting the track, and so the wanted
        volume beats the session manager's remembered per-stream value. Returns
        False (the caller re-spawns / keeps --volume) when the PipeWire tools or
        the stream node are not found -- e.g. on macOS/afplay or a non-PipeWire
        host."""
        node = self._stream_node(proc.pid)
        if node is None:
            return False
        level = f"{max(0.0, min(1.0, volume)):.3f}"
        try:
            done = subprocess.run(["wpctl", "set-volume", str(node), level],
                                  capture_output=True,
                                  timeout=PW_TOOL_TIMEOUT_S)
            return done.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _stream_node(self, pid: int) -> int | None:
        """The PipeWire Stream/Output node id for a player process. The pid
        maps to a Client object; the audio node links to it via client.id."""
        for _ in range(PW_NODE_RETRIES):
            node = self._stream_node_once(pid)
            if node is not None:
                return node
            if self._wait(PW_NODE_RETRY_WAIT_S):  # shutdown cuts this short
                return None
        return None

    @staticmethod
    def _stream_node_once(pid: int) -> int | None:
        objs = _pw_dump()
        if objs is None:
            return None
        client = _client_for_pid(objs, pid)
        if client is None:
            return None
        return _node_for_client(objs, client)


def _pw_dump() -> list | None:
    try:
        dump = subprocess.run(["pw-dump"], capture_output=True,
                              timeout=PW_TOOL_TIMEOUT_S, text=True)
        if dump.returncode != 0:
            return None
        return json.loads(dump.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _props(obj: dict) -> dict:
    return (obj.get("info") or {}).get("props") or {}


def _client_for_pid(objs: list, pid: int) -> int | None:
    return next((o.get("id") for o in objs
                 if _props(o).get("application.process.id") == pid), None)


def _node_for_client(objs: list, client: int) -> int | None:
    for o in objs:
        if not str(o.get("type", "")).endswith("Node"):
            continue
        props = _props(o)
        if (props.get("client.id") == client
                and str(props.get("media.class", "")).startswith("Stream/Output")):
            return o.get("id")
    return None
