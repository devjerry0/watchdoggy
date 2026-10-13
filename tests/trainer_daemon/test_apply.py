"""Cloud results reach the appliance in bounded batch requests."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from trainer_daemon import apply  # noqa: E402


def test_big_pass_applies_in_chunks_and_falls_back_per_chunk(tmp_path, monkeypatch):
    verdicts = {f"sample_{i:05d}": "empty" for i in range(2500)}
    (tmp_path / "v.json").write_text(json.dumps(verdicts))
    (tmp_path / "p.json").write_text(json.dumps({"sample_00007": []}))
    batches, singles = [], []

    def fake_post(path, payload, timeout=None):
        if path == "/api/dataset/apply-cloud-results":
            batches.append(len(payload["auto_verdicts"]))
            if len(batches) == 2:
                raise TimeoutError("The read operation timed out")
            return {"prelabels": len(payload["prelabels"]),
                    "auto_verdicts": len(payload["auto_verdicts"]),
                    "disputes": 0, "clear_auto": 0}
        singles.append(payload["name"])
        return {"ok": True}

    monkeypatch.setattr(apply, "api_post", fake_post)
    counts = apply.apply_cloud_results(tmp_path / "p.json", tmp_path / "v.json",
                                       tmp_path / "missing.json")
    assert batches == [1000, 1000, 500]
    assert len(singles) == 1000             # only the failed chunk went per-frame
    assert counts == (1, 2500, 0, 0)


def test_boxes_carry_the_tag_of_the_labeler_that_drew_them(tmp_path, monkeypatch):
    import modal_pipeline
    tags = []

    def fake_post(path, payload, timeout=None):
        tags.append(payload["model"])
        if path == "/api/dataset/apply-cloud-results":
            raise TimeoutError  # the per-frame fallback sends the same tag
        return {"ok": True}

    monkeypatch.setattr(apply, "api_post", fake_post)
    modal_pipeline._write_label_results(tmp_path, {
        "prelabels": {"sample_1": []}, "labeler": "yolo26x-kitchen-abcd1234"})
    apply.apply_cloud_results(tmp_path / "prelabels.json",
                              model=apply.labeler_tag(tmp_path))
    assert tags == ["yolo26x-kitchen-abcd1234"] * 2
    # Results from before labeler swaps name no labeler: stock yolo26x.
    modal_pipeline._write_label_results(tmp_path, {"prelabels": {}})
    assert apply.labeler_tag(tmp_path) == "yolo26x"
    assert apply.labeler_tag(tmp_path / "missing") == "yolo26x"


def test_clears_ride_the_same_chunks_and_fall_back_per_frame(tmp_path, monkeypatch):
    import modal_pipeline
    from trainer_daemon import runs
    clears = [f"sample_{i:05d}" for i in range(1500)]
    batches, singles = [], []

    def fake_post(path, payload, timeout=None):
        if path == "/api/dataset/apply-cloud-results":
            batches.append(sorted(payload["clear_auto"]))
            if len(batches) == 2:
                raise TimeoutError("The read operation timed out")
            return {"prelabels": 0, "auto_verdicts": len(payload["auto_verdicts"]),
                    "disputes": 0, "clear_auto": len(payload["clear_auto"])}
        singles.append((path, payload["name"]))
        return {"ok": True}

    def fake_modal_run(entrypoint, arguments, job_id):
        out = Path(arguments[arguments.index("--out-dir") + 1])
        modal_pipeline._write_label_results(out, {
            "prelabels": {}, "auto_verdicts": {"sample_00000": "dog"},
            "clear_auto": clears[1:]})

    monkeypatch.setattr(apply, "api_post", fake_post)
    monkeypatch.setattr(runs, "modal_run", fake_modal_run)
    monkeypatch.setattr(runs, "billing_summary", lambda: None)
    monkeypatch.setattr(runs, "run_cost", lambda before: None)
    detail = runs.run_prelabel_job({"id": "job-1"})
    assert [len(b) for b in batches] == [999, 500]     # one chunk of 1,000 frames each
    assert singles == [("/api/dataset/autolabel/clear", stem) for stem in clears[1000:]]
    assert detail.endswith("1 auto-labeled, 0 labels disputed, "
                           "1499 auto-labels cleared")
