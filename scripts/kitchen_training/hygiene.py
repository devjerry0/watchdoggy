"""Volume hygiene and run-budget guards for cloud runs.

Born from the Sep 2026 outage: four timed-out training runs each left a
full built dataset (~100k files) on the Volume, which then hit Modal's
500,000-inode limit -- after which every run died at mkdir(ENOSPC) while
the daemon re-queued it every 30 minutes. These helpers make a run clean
up after its predecessors BEFORE it writes anything (a crashed container
can never brick the Volume for good), keep finished runs small, and scale
epochs so a corpus that grew 8x overnight cannot blow the time budget.

Pure path/arithmetic helpers: no ML imports, all roots passed in.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

# A run dir carrying this marker finished its pipeline. Anything without it
# is the corpse of a crashed or timed-out container: the next run's janitor
# deletes it.
COMPLETED_MARKER = ".completed"

# Total images-seen budget per training run, anchored to the proven recipe
# (80 epochs x ~2.7k train images). Overnight auto-labeling can grow the
# corpus by an order of magnitude; holding the step budget roughly constant
# keeps runs inside the function timeout and the monthly credits.
EPOCH_IMAGE_BUDGET = 216_000
MIN_EPOCHS = 15

# An unmarked run dir younger than this is spared: it may belong to a run
# that is still executing in another container (e.g. a Mac-launched kickoff
# overlapping the Pi's pass). 11h = the 10h cloud-job ceiling
# (modal_pipeline.CLOUD_JOB_CEILING) + 1h -- past it, no container can
# still be writing. Real corpses are hours old by the next attempt, so
# self-heal is delayed at most one backoff cycle.
CORPSE_GRACE_SECONDS = 11 * 3600


def scale_epochs(requested: int, corpus_images: int) -> int:
    """Epochs that spend ~EPOCH_IMAGE_BUDGET on this corpus: never more than
    requested, and never below MIN_EPOCHS unless the request itself was."""
    scaled = round(EPOCH_IMAGE_BUDGET / max(corpus_images, 1))
    return min(requested, max(MIN_EPOCHS, scaled))


def janitor(runs_dir: Path, keep: int, current_run: str,
            kickoff_root: Path | None = None) -> list[str]:
    """Start-of-run cleanup; returns what was deleted, for the run log.

    Run dirs: corpses (no completion marker) go unconditionally; completed
    runs are pruned to the newest keep-1 by mtime (the current run becomes
    the keep-th). Mtime, not name order: the Pi's clock can regress, so
    timestamp names are not trusted to sort by age. Kickoff upload dirs
    (per-job deployed-bundle copies) are pruned to the newest keep. The
    current run's dirs are never touched.
    """
    now = time.time()
    deleted: list[str] = []

    def _remove(path: Path, label: str) -> None:
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():  # report only what is actually gone
            deleted.append(label)

    def _mtime(path: Path) -> float | None:
        try:  # another client may delete a dir between iterdir and stat
            return path.stat().st_mtime
        except OSError:
            return None

    if runs_dir.is_dir():
        completed: list[tuple[float, Path]] = []
        corpses: list[Path] = []
        for path in runs_dir.iterdir():
            if not path.is_dir() or path.name == current_run:
                continue
            mtime = _mtime(path)
            if mtime is None:
                continue
            if (path / COMPLETED_MARKER).is_file():
                completed.append((mtime, path))
            elif now - mtime >= CORPSE_GRACE_SECONDS:
                corpses.append(path)
        completed.sort()
        keep_completed = max(keep - 1, 0)
        stale = corpses + [p for _, p in
                           (completed[:-keep_completed] if keep_completed
                            else completed)]
        for path in stale:
            _remove(path, path.name)
    if kickoff_root is not None and kickoff_root.is_dir():
        uploads = sorted(
            ((m, p) for p in kickoff_root.iterdir()
             if p.is_dir() and p.name != current_run
             and (m := _mtime(p)) is not None))
        for _, path in uploads[:-keep] if keep else uploads:
            _remove(path, f"{kickoff_root.name}/{path.name}")
    return deleted


def strip_dataset(run_dir: Path) -> None:
    """Delete the built dataset tree (2+ files per corpus frame -- what blew
    the inode limit). Weights, report, and the NCNN bundle stay; the dataset
    is rebuilt from the mirror on every run anyway."""
    shutil.rmtree(run_dir / "dataset", ignore_errors=True)


def mark_completed(run_dir: Path) -> None:
    (run_dir / COMPLETED_MARKER).touch()
