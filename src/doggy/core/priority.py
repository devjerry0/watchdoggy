"""Per-thread CPU priority (Linux). The detector's NCNN threads sync at every
layer, so any other thread that grabs a core costs FPS; bulk web work
(index scans, 50k-sidecar aggregates) runs at a higher nice instead.

Linux applies setpriority() to a single thread when given its native TID;
lowering priority never needs privileges, and it sticks for the thread's
life (raising it back would), which is exactly what a pooled web worker
should do once it has proven it does heavy work."""
from __future__ import annotations

import os
import sys
import threading

WEB_WORKER_NICE = 10
BACKGROUND_NICE = 15


def lower_thread_priority(nice: int) -> None:
    """Raise the current thread's nice to at least ``nice``. No-op off Linux
    (macOS applies setpriority to the whole process)."""
    if sys.platform != "linux":
        return
    try:
        tid = threading.get_native_id()
        if os.getpriority(os.PRIO_PROCESS, tid) < nice:
            os.setpriority(os.PRIO_PROCESS, tid, nice)
    except (OSError, AttributeError):
        pass
