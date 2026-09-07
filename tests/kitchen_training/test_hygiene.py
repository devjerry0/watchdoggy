"""Volume hygiene: the janitor must make a full Volume heal itself at the
START of a run (corpses from timed-out containers deleted before mkdir),
and epoch scaling must keep an exploded corpus inside the step budget.

hygiene.py is loaded directly by file path: kitchen_training/__init__ eagerly
imports the ML stack (cv2, ultralytics), which these path/arithmetic tests
must not depend on."""
import importlib.util
import os
import time
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "kt_hygiene",
    Path(__file__).resolve().parents[2]
    / "scripts/kitchen_training/hygiene.py")
hygiene = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hygiene)


def _run_dir(root, name, completed, age_minutes):
    d = root / name
    d.mkdir(parents=True)
    (d / "weights.pt").write_text("x")
    if completed:
        (d / hygiene.COMPLETED_MARKER).touch()
    stamp = time.time() - age_minutes * 60
    os.utime(d, (stamp, stamp))
    return d


def test_corpses_die_completed_runs_kept_to_quota(tmp_path):
    runs = tmp_path / "training-runs"
    corpse = _run_dir(runs, "20260831-200315", completed=False, age_minutes=900)
    oldest = _run_dir(runs, "20260825-145945", completed=True, age_minutes=400)
    kept = [_run_dir(runs, f"2026090{i}-000000", completed=True,
                     age_minutes=300 - i) for i in range(1, 5)]
    (runs / "prelabel-cache.json").write_text("{}")

    deleted = hygiene.janitor(runs, keep=5, current_run="20260906-120000")

    assert corpse.name in deleted and not corpse.exists()
    # quota is keep-1 completed runs: the current run becomes the keep-th
    assert oldest.name in deleted and not oldest.exists()
    assert all(d.exists() for d in kept)
    assert (runs / "prelabel-cache.json").exists()  # files are never touched


def test_current_run_is_never_deleted_even_without_marker(tmp_path):
    runs = tmp_path / "training-runs"
    current = _run_dir(runs, "20260906-120000", completed=False, age_minutes=0)
    assert hygiene.janitor(runs, keep=5, current_run=current.name) == []
    assert current.exists()


def test_young_unmarked_dir_survives_grace_window(tmp_path):
    # A concurrent run from another client (Mac kickoff overlapping the Pi's
    # pass) has no marker yet but a recent mtime: it must NOT be janitored.
    runs = tmp_path / "training-runs"
    live = _run_dir(runs, "20260906-110000", completed=False, age_minutes=30)
    old_corpse = _run_dir(runs, "20260901-054129", completed=False,
                          age_minutes=int(hygiene.CORPSE_GRACE_SECONDS / 60) + 60)
    deleted = hygiene.janitor(runs, keep=5, current_run="20260906-120000")
    assert live.exists()
    assert not old_corpse.exists() and old_corpse.name in deleted


def test_janitor_reports_only_what_is_gone(tmp_path):
    # deleted list must reflect reality, not intent (rmtree can fail).
    runs = tmp_path / "training-runs"
    corpse = _run_dir(runs, "20260831-200315", completed=False, age_minutes=900)
    deleted = hygiene.janitor(runs, keep=5, current_run="20260906-120000")
    assert deleted == [corpse.name] and not corpse.exists()


def test_clock_regression_prunes_by_mtime_not_name(tmp_path):
    runs = tmp_path / "training-runs"
    # Name says old, mtime says newest (Pi clock regressed): must survive.
    misdated = _run_dir(runs, "20200101-000000", completed=True, age_minutes=1)
    victims = [_run_dir(runs, f"2026090{i}-000000", completed=True,
                        age_minutes=1000 + i) for i in range(1, 6)]
    hygiene.janitor(runs, keep=5, current_run="20260906-120000")
    assert misdated.exists()
    assert not victims[-1].exists()  # oldest by mtime went instead


def test_kickoff_uploads_pruned_to_quota(tmp_path):
    runs = tmp_path / "training-runs"
    runs.mkdir()
    kickoffs = tmp_path / "runs"
    stale = [_run_dir(kickoffs, f"2026090{i}-000000", completed=False,
                      age_minutes=100 + i) for i in range(1, 7)]
    current = _run_dir(kickoffs, "20260906-120000", completed=False,
                       age_minutes=0)
    hygiene.janitor(runs, keep=5, current_run=current.name,
                    kickoff_root=kickoffs)
    assert current.exists()
    survivors = [d for d in stale if d.exists()]
    assert len(survivors) == 5


def test_strip_dataset_spares_bundle_and_report(tmp_path):
    run = tmp_path / "run"
    (run / "dataset/images/train").mkdir(parents=True)
    (run / "dataset/images/train/f.jpg").write_text("x")
    (run / "kitchen_ncnn_model").mkdir()
    (run / "report.md").write_text("x")
    hygiene.strip_dataset(run)
    assert not (run / "dataset").exists()
    assert (run / "kitchen_ncnn_model").exists()
    assert (run / "report.md").exists()


def test_epoch_scaling_holds_the_step_budget():
    # The proven anchor stays untouched...
    assert hygiene.scale_epochs(80, 2_700) == 80
    # ...an exploded corpus scales down to the floor...
    assert hygiene.scale_epochs(80, 22_000) == hygiene.MIN_EPOCHS
    # ...a mid-size corpus lands proportionally...
    assert hygiene.scale_epochs(80, 8_000) == 27
    # ...an explicit low request is never raised...
    assert hygiene.scale_epochs(10, 22_000) == 10
    # ...and a legacy 200-epoch request is capped at the proven 80 anchor.
    assert hygiene.scale_epochs(200, 2_700) == 80
