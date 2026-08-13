"""The run report: leaderboard, changed frames, deploy-truth, deploy commands."""
from __future__ import annotations

import json
from pathlib import Path

from kitchen_training.config import (DATASET_MIRROR, FIRE_CONF, THRESHOLDS, log)
from kitchen_training.evaluation import rank_key


def _leaderboard_md(metrics: dict, winner: str) -> list[str]:
    def cell(model_metrics, threshold, scope):
        s = model_metrics["curve"][threshold][scope]
        return f"{s['caught']}/{s['caught'] + s['missed']} · {s['false_fires']} FP"

    def tag(name):
        if name == winner:
            return " **(winner)**"
        if name == "baseline":
            return " (baseline)"
        return ""

    ranked = sorted(metrics["models"],
                    key=lambda name: rank_key(metrics["models"][name]))
    lines = ["| model | held-out @0.6 | held-out @0.7 | held-out @0.8 | "
             "all frames @0.7 (most seen in training) | val mAP50 |",
             "|---|---|---|---|---|---|"]
    for name in ranked:
        model_metrics = metrics["models"][name]
        map50 = model_metrics["map50_val"]
        map50_cell = map50 if map50 is not None else "--"
        lines.append(f"| {name}{tag(name)} | {cell(model_metrics, '0.6', 'heldout')} | "
                     f"{cell(model_metrics, '0.7', 'heldout')} | "
                     f"{cell(model_metrics, '0.8', 'heldout')} | "
                     f"{cell(model_metrics, '0.7', 'all')} | {map50_cell} |")
    return lines


def _truth_by_stem(known_stems: dict) -> dict[str, bool]:
    truth = {}
    for sidecar in sorted(DATASET_MIRROR.glob("sample_*.json")):
        if sidecar.stem not in known_stems:
            continue
        verdict = json.loads(sidecar.read_text()).get("human_label")
        truth[sidecar.stem] = verdict in ("dog", "dog_mixed")
    return truth


def _changed_md(metrics: dict, winner: str) -> list[str]:
    baseline_confs = metrics["models"]["baseline"]["confs"]
    winner_confs = metrics["models"][winner]["confs"]
    lines = ["", f"Frames where {winner} changed the verdict vs baseline "
                 f"(@{FIRE_CONF}):"]
    changed = []
    for stem, is_dog in _truth_by_stem(winner_confs).items():
        fired_before = baseline_confs[stem] >= FIRE_CONF
        fires_now = winner_confs[stem] >= FIRE_CONF
        if fired_before == fires_now:
            continue
        changed.append(f"- {stem}: truth={'dog' if is_dog else 'no-dog'}, now "
                       f"{'fires' if fires_now else 'quiet'} "
                       f"({'improvement' if fires_now == is_dog else 'REGRESSION'})")
    return lines + (changed or ["- none"])


def _dropped_md(dataset_stats: dict) -> list[str]:
    if not dataset_stats.get("dropped"):
        return []
    return (["", "Frames excluded from training -- open /review, tap them in "
                 "the history rail, and draw their boxes:"]
            + [f"- {stem}" for stem, _ in dataset_stats["dropped"]])


def _ncnn_md(metrics: dict, winner: str) -> list[str]:
    deploy_truth = metrics.get("ncnn_truth")
    if not deploy_truth:
        return []
    lines = ["", f"## Deploy-truth: the NCNN bundle itself ({winner})",
             "The Pi runs this bundle, whose NMS head scores differently "
             "than the .pt above.", "",
             "| threshold | all frames | held-out |", "|---|---|---|"]
    for threshold in (f"{t:.1f}" for t in THRESHOLDS):
        a, h = deploy_truth[threshold]["all"], deploy_truth[threshold]["heldout"]
        lines.append(f"| {threshold} | {a['caught']}/{a['caught']+a['missed']} · "
                     f"{a['false_fires']} FP | "
                     f"{h['caught']}/{h['caught']+h['missed']} · "
                     f"{h['false_fires']} FP |")
    lines.append(f"\nHighest dog-confidence on any non-dog frame: "
                 f"{deploy_truth['max_nondog_conf']}")
    return lines


def _deploy_md(ncnn_bundle: Path | None) -> list[str]:
    if not ncnn_bundle:
        return []
    return ["", "## Deploy", "```",
            f"rsync -az {ncnn_bundle}/ doggy@doggypi.local:doggy/models/kitchen_ncnn_model/",
            "ssh doggy@doggypi.local \"sed -i "
            "'s|DOGGY_MODEL_PATH=.*|DOGGY_MODEL_PATH=models/kitchen_ncnn_model|' "
            "~/doggy/.env && sudo systemctl restart doggy\"", "```"]


def report(run_dir: Path, dataset_stats: dict, metrics: dict, winner: str,
           ncnn_bundle: Path | None) -> None:
    (run_dir / "report.json").write_text(json.dumps(
        {"dataset": dataset_stats, "metrics": metrics, "winner": winner,
         "run_dir": str(run_dir)}, indent=1, default=str))

    heldout_dogs = metrics["heldout_dogs"]
    heldout_nondogs = metrics["heldout_nondogs"]
    lines = [f"# Training run {run_dir.name}", "",
             f"Dataset: {dataset_stats['train']} train / {dataset_stats['val']} "
             f"val frames ({dataset_stats['verdicts']})", "",
             f"Held-out = {heldout_dogs} dog + {heldout_nondogs} non-dog frames "
             "the models never trained on. One frame there is worth "
             f"~{100 // max(heldout_dogs, 1)} points -- "
             "treat single-frame gaps as noise, not signal.", ""]
    lines += _leaderboard_md(metrics, winner)
    lines += _changed_md(metrics, winner)
    lines += _dropped_md(dataset_stats)
    lines += _ncnn_md(metrics, winner)
    lines += _deploy_md(ncnn_bundle)
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    log(f"report: {run_dir / 'report.md'}")
