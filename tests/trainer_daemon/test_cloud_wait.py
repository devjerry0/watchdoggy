"""modal_pipeline._wait: a cloud call that died without a result fails the
wait instead of blocking the trainer. On 2026-10-13 a labeler container
vanished mid-epoch, Modal marked its input FAILURE, call.get() never
returned, and the Pi waited 12 h with every other job queued behind it."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import modal_pipeline  # noqa: E402


def node(status: str, *children) -> SimpleNamespace:
    """One call-graph input, shaped like modal's InputInfo."""
    return SimpleNamespace(status=SimpleNamespace(name=status),
                           children=list(children))


class FakeCall:
    """get() times out `timeouts` times, then returns `result`; each call
    graph check returns the next graph (a list of root nodes)."""
    object_id = "fc-test"

    def __init__(self, timeouts: int, graphs: list, result=None):
        self.timeouts, self.graphs, self.result = timeouts, graphs, result
        self.gets = 0

    def get(self, timeout):
        self.gets += 1
        if self.gets <= self.timeouts:
            raise TimeoutError
        return self.result

    def get_call_graph(self):
        return self.graphs.pop(0)


def test_a_call_that_died_without_a_result_fails_the_wait():
    call = FakeCall(timeouts=10_000, graphs=[[node("FAILURE")]] * 5)
    with pytest.raises(RuntimeError, match="FAILURE"):
        modal_pipeline._wait(call)
    assert call.gets == modal_pipeline.DEAD_CHECKS


def test_a_running_call_keeps_waiting_through_one_dead_looking_check():
    call = FakeCall(timeouts=3, result="results", graphs=[
        [node("FAILURE")], [node("PENDING")], [node("PENDING")]])
    assert modal_pipeline._wait(call) == "results"


def test_a_dead_mapped_child_under_a_pending_parent_fails_the_wait():
    # run_label_pass maps run_consensus_chunk: a chunk that died without a
    # result leaves the parent PENDING until its own 10 h timeout.
    graph = [node("PENDING", node("SUCCESS"), node("FAILURE"))]
    call = FakeCall(timeouts=10_000, graphs=[graph] * 5)
    with pytest.raises(RuntimeError, match="FAILURE"):
        modal_pipeline._wait(call)
