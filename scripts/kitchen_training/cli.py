"""Argument parsing and the linear stage runner (the dashboard button's entry)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import kitchen_training
from kitchen_training.build import build
from kitchen_training.config import BASE_MODEL, RUNS, log
from kitchen_training.dataset import prelabel_push, pull
from kitchen_training.evaluation import evaluate, ncnn_truth, pick_winner
from kitchen_training.export import export
from kitchen_training.report import report
from kitchen_training.training import sweep, train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=kitchen_training.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="doggy@doggypi.local")
    parser.add_argument("--backend", choices=("local", "modal"), default="local",
                        help="where the train stage runs: this machine's GPU, "
                             "or a Modal cloud GPU (scripts/modal_train.py)")
    parser.add_argument("--sweep", action="store_true",
                        help="train every SWEEP_CONFIGS variant on Modal in "
                             "parallel and leaderboard them (implies "
                             "--backend modal)")
    parser.add_argument("--gpu", default="L4",
                        help="Modal GPU type (T4, L4, A10G, A100, H100)")
    parser.add_argument("--augment", action="store_true",
                        help="add motion-blur/defocus/dark copies of every "
                             "TRAIN frame (labels carry over; the held-out "
                             "exam stays pristine)")
    parser.add_argument("--push-prelabels", action="store_true",
                        help="no training: big-model boxes for every frame, "
                             "pushed into the Pi's sidecars for the review page")
    parser.add_argument("--pi-url", default="https://doggypi.local:8443")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--freeze", type=int, default=10,
                        help="backbone layers to freeze (small datasets "
                             "need this)")
    return parser.parse_args()


def _candidate_weights(args, run_dir: Path) -> dict[str, Path]:
    if args.sweep:
        return sweep(run_dir, args.gpu)
    best = train(run_dir, args.backend, args.epochs, args.batch, args.freeze)
    return {"fine-tune": best}


def _export_winner(run_dir: Path, best: Path, metrics: dict) -> Path | None:
    # A broken exporter must never cost us the training + eval results.
    try:
        ncnn_bundle = export(run_dir, best)
        metrics["ncnn_truth"] = ncnn_truth(run_dir, ncnn_bundle)
        return ncnn_bundle
    except Exception as exc:
        log(f"WARNING: NCNN export failed ({exc}); report continues without it")
        return None


def main() -> None:
    args = _parse_args()
    os.environ["DOGGY_TRAIN_GPU"] = args.gpu  # read by scripts/modal_train.py

    if args.push_prelabels:
        if not args.skip_pull:
            pull(args.host)
        prelabel_push(args.pi_url)
        return

    run_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    log(f"run dir: {run_dir}")

    if not args.skip_pull:
        pull(args.host)
    dataset_stats = build(run_dir, augment=args.augment)
    weights_by_name = _candidate_weights(args, run_dir)
    weights_by_name["baseline"] = BASE_MODEL
    metrics = evaluate(run_dir, weights_by_name)
    winner = pick_winner(metrics)
    log(f"winner: {winner}")
    ncnn_bundle = None
    if not args.skip_export:
        ncnn_bundle = _export_winner(run_dir, weights_by_name[winner], metrics)
    report(run_dir, dataset_stats, metrics, winner, ncnn_bundle)
    log("done.")
