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
                    "disputes": 0}
        singles.append(payload["name"])
        return {"ok": True}

    monkeypatch.setattr(apply, "api_post", fake_post)
    counts = apply.apply_cloud_results(tmp_path / "p.json", tmp_path / "v.json",
                                       tmp_path / "missing.json")
    assert batches == [1000, 1000, 500]
    assert len(singles) == 1000             # only the failed chunk went per-frame
    assert counts == (1, 2500, 0)
