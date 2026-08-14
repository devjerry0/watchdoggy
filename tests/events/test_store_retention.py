"""EventStore retention: pruning by event count and by age."""
from doggy.events.store import EventStore

from tests.events.conftest import _img


def test_prune_by_count_deletes_files(tmp_path):
    s = EventStore(tmp_path, max_events=2, max_age_days=0)
    r0 = s.add(_img(), 0.5, None, 1.0, 1.0)
    s.add(_img(), 0.5, None, 2.0, 2.0)
    s.add(_img(), 0.5, None, 3.0, 3.0)
    assert len(s.list()) == 2
    assert not (tmp_path / r0.thumb).exists()   # oldest file removed


def test_prune_by_age(tmp_path):
    s = EventStore(tmp_path, max_events=100, max_age_days=1, clock=lambda: 1_000_000.0)
    old = s.add(_img(), 0.5, None, 1_000_000.0 - 2 * 86400, 1.0)  # 2 days old by wall_time
    fresh = s.add(_img(), 0.5, None, 1_000_000.0, 2.0)
    s.prune()
    ids = [e.id for e in s.list()]
    assert fresh.id in ids and old.id not in ids
