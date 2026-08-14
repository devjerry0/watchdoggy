"""Shared helpers for tests/events/: a bare frame factory and a "catch"
helper that plays a sound and (optionally) attaches an outcome."""
import numpy as np


def _img():
    return np.zeros((8, 8, 3), np.uint8)


def _catch(s, sound, wall_time, mono_ts, clear=None, taken=(), outcome=True):
    r = s.add(_img(), 0.9, 1.0, wall_time, mono_ts)
    s.attach_sound(r.id, sound)
    if outcome:
        s.attach_outcome(r.id, clear_seconds=clear, taken=list(taken),
                         wall_time=wall_time + 30)
    return r
