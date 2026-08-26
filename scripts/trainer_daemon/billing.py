"""Workspace billing and the monthly credit gate for cloud jobs.

The daemon refreshes jobs/billing.json from Modal at the start of every
pass, so the gate always judges numbers at most one pass old. Missing or
unreadable billing fails OPEN (this is a cost guard, not a lock); real
money already billed, or metered spend at the configured monthly credits,
refuses cloud jobs until the Modal cycle resets or the budget is raised.
"""
from __future__ import annotations

import json
import subprocess
import time

from trainer_daemon import env

BILLING_CLI_TIMEOUT_S = 60
CLOUD_KINDS = ("train", "prelabel")  # job kinds that spend Modal compute


def billing_summary() -> dict | None:
    """Workspace spend this month, straight from Modal. The workspace runs
    only this appliance, so before/after deltas attribute cost per run."""
    try:
        proc = subprocess.run([str(env.MODAL), "billing", "summary", "--json"],
                              capture_output=True,
                              timeout=BILLING_CLI_TIMEOUT_S, check=True)
        summary = json.loads(proc.stdout)
        return {"metered_cost": float(summary.get("metered_cost", 0)),
                "billed_cost": float(summary.get("billed_cost", 0)),
                "credits_used": -float(summary.get("adjustments", {})
                                       .get("credits", 0)),
                "fetched_at": time.time()}
    except Exception as exc:
        env.log(f"WARNING: billing summary unavailable: {exc}")
        return None


def write_billing() -> dict | None:
    summary = billing_summary()
    if summary:
        (env.JOBS_DIR / "billing.json").write_text(json.dumps(summary))
    return summary


def run_cost(before: dict | None) -> float | None:
    after = write_billing()
    if not (before and after):
        return None
    return round(max(0.0, after["metered_cost"] - before["metered_cost"]), 2)


def _read_billing() -> dict | None:
    path = env.JOBS_DIR / "billing.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def credit_refusal(kind: str) -> str | None:
    """Why a job of this kind must not start right now, or None to allow."""
    if kind not in CLOUD_KINDS:
        return None
    summary = _read_billing()
    if summary is None:
        return None
    billed = float(summary.get("billed_cost", 0.0))
    if billed > 0:
        return (f"refused: ${billed:.2f} already billed beyond credits "
                "this month -- cloud jobs paused")
    used = float(summary.get("metered_cost", 0.0))
    budget = float(env.settings().get("monthly_credits", 0))
    if used < budget:
        return None
    return (f"refused: no Modal credits left (${used:.2f} used of "
            f"${budget:.0f}/month) -- cloud jobs resume when the cycle "
            "resets or monthly credits is raised in settings")
