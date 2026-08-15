"""Keep the Pi's clock honest over the trainer's 443 exception.

The appliance has no battery clock and the egress firewall blocks public
NTP (UDP 123 is not in the trainer's exception). But every HTTPS response
carries a Date header accurate to ~a second -- the htpdate trick -- and
the trainer already talks to api.github.com. Each daemon pass compares
that time to the local clock and steps it via a fixed-path root helper
when the drift exceeds the threshold. Router NTP (timesyncd) stays
configured as the primary; this is the layer that survives a router that
doesn't serve time and a power loss that lands the clock in the past."""
from __future__ import annotations

import subprocess
import time
import urllib.request
from email.utils import parsedate_to_datetime

from trainer_daemon.env import log

TIME_SOURCE = "https://api.github.com/"
REQUEST_TIMEOUT_S = 15
# Below this the clock is fine (Date has 1s granularity; don't chase jitter).
STEP_THRESHOLD_S = 3.0
SET_CLOCK_HELPER = "/usr/local/bin/doggy-set-clock"


def _http_date_now() -> float | None:
    """Wall-clock time from the Date header, RTT-midpoint corrected."""
    request = urllib.request.Request(
        TIME_SOURCE, method="HEAD",
        headers={"User-Agent": "watchdoggy-clock"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(request,
                                    timeout=REQUEST_TIMEOUT_S) as resp:
            rtt = time.monotonic() - t0
            header = resp.headers.get("Date")
            if not header:
                return None
            return parsedate_to_datetime(header).timestamp() + rtt / 2
    except Exception:
        return None


def sync_clock() -> None:
    """Best effort, never raises: a failed sync must not cost the pass."""
    remote = _http_date_now()
    if remote is None:
        log("clock: time source unreachable; skipping sync")
        return
    delta = remote - time.time()
    if abs(delta) < STEP_THRESHOLD_S:
        return
    try:
        subprocess.run(["sudo", SET_CLOCK_HELPER, str(int(remote))],
                       check=True, timeout=30)
        log(f"clock: stepped {delta:+.0f}s (HTTPS Date header)")
    except Exception as exc:
        log(f"WARNING: clock step failed: {exc}")
