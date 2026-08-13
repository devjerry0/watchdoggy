"""Fine-tune the kitchen dog detector on the labeled dataset, end to end.

One command runs the whole training day:

    uv run python scripts/train_kitchen_model.py --host doggy@doggypi.local

Stages (each skippable):
  pull       rsync the labeled dataset from the Pi
  build      fuse human verdicts + big-model prelabels into a YOLO dataset
  train      fine-tune the nano model (frozen backbone, early stopping);
             --backend modal sends this one stage to a Modal cloud GPU
  evaluate   baseline vs fine-tune, scored the way the appliance fires
  export     NCNN bundle ready to rsync to the Pi
  report     dataset stats, metrics, and changed frames, in the run dir

Everything lands in training-runs/<timestamp>/ so runs are comparable.
This package IS the future dashboard button's backend: keep it
argument-driven and side-effect free outside its run directory (and the
dataset mirror).

Label fusion rules (the heart of it):
  hand boxes drawn on /review are the complete truth for a frame and
             override everything below.
  dog        keep big-model dog+person boxes; if the big model missed the dog,
             fall back to the nano's sidecar box; drop the frame if neither.
  dog_mixed  big-model dog boxes that heavily overlap a person box are the
             misclassified person -- dropped; the rest are real.
  person     every dog box is a lie (that is the point); keep person boxes.
  no_dog     legacy coarse verdict: same as person.
  empty      background: empty label file (a hard negative).
  skip       excluded.

The split is STABLE: a frame's stem hash decides train/val forever, so growing
the dataset never moves old frames across the split (no quiet contamination).
"""
from kitchen_training.build import build
from kitchen_training.cli import main
from kitchen_training.evaluation import (evaluate, ncnn_truth, pick_winner,
                                         rank_key, robustness)
from kitchen_training.export import export
from kitchen_training.report import report
from kitchen_training.training import sweep, train

__all__ = ["build", "evaluate", "export", "main", "ncnn_truth", "pick_winner",
           "rank_key", "report", "robustness", "sweep", "train"]
