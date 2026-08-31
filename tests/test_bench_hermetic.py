"""The hermetic benchmark environment must actually stop the cheats it names."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from oncoscope.bench.hermetic import (assert_no_byte_leakage, build_bench,
                                      near_duplicate_scan, run_hermetic,
                                      verify_bench)
from oncoscope.bench.mias import parse_info


@dataclass(frozen=True)
class _Case:
    case_id: str
    patient_id: str
    label: int
    image_path: str = ""


def _mk_bench(tmp_path, n=6):
    rng = np.random.default_rng(0)
    cases = [_Case(f"c{i:02d}", f"p{i // 2:02d}", i % 2) for i in range(n)]
    out = tmp_path / "bench"
    build_bench(cases, lambda c: rng.random((32, 32), dtype=np.float32), out, "t1")
    return out


def test_seal_round_trip_and_tamper_detection(tmp_path):
    out = _mk_bench(tmp_path)
    gold = verify_bench(out)                      # clean verify passes
    assert len(gold["rows"]) == 6

    raw = json.loads((out / "gold.json").read_text())
    raw["rows"][0]["label"] ^= 1                  # flip one label
    (out / "gold.json").write_text(json.dumps(raw, indent=1))
    with pytest.raises(ValueError, match="seal verification FAILED"):
        verify_bench(out)


def test_staged_dir_contains_no_gold_and_opaque_ids(tmp_path):
    out = _mk_bench(tmp_path)
    gold = json.loads((out / "gold.json").read_text())
    for row in gold["rows"]:
        assert row["case_id"] not in row["opaque_id"]   # id leaks nothing
    # the adapter view is cases/*.npy only — bare arrays, no headers
    staged_names = {p.name for p in (out / "cases").iterdir()}
    assert all(n.endswith(".npy") for n in staged_names)
    assert len(staged_names) == 6


def test_byte_leakage_check_fires(tmp_path):
    f = tmp_path / "img.bin"
    f.write_bytes(b"SAME BYTES")
    import hashlib
    train_hashes = {hashlib.sha256(b"SAME BYTES").hexdigest()}
    with pytest.raises(ValueError, match="LEAKAGE"):
        assert_no_byte_leakage([f], train_hashes)
    assert_no_byte_leakage([f], {"deadbeef"})     # disjoint set passes


def test_near_duplicate_scan_fires():
    a = np.eye(4)[:2]
    with pytest.raises(ValueError, match="near-duplicate"):
        near_duplicate_scan(a, a)
    assert near_duplicate_scan(a, np.eye(4)[2:]) == 0.0


def _write_adapter(tmp_path, body):
    p = tmp_path / "adapter.py"
    p.write_text(
        "import argparse, json, numpy as np\nfrom pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--input'); ap.add_argument('--output')\n"
        "a = ap.parse_args()\n"
        "ids = json.loads((Path(a.input)/'manifest.json').read_text())['case_ids']\n"
        + body)
    return p


def test_honest_adapter_passes_and_scores(tmp_path):
    out = _mk_bench(tmp_path)
    adapter = _write_adapter(tmp_path,
        "with open(a.output,'w') as f:\n"
        "    for c in ids:\n"
        "        px = np.load(Path(a.input)/'cases'/f'{c}.npy')\n"
        "        f.write(json.dumps({'case_id': c, 'score': float(px.mean())})+'\\n')\n")
    result = run_hermetic(out, adapter)
    assert set(result.scores) == {p.stem for p in (out / "cases").iterdir()}
    assert result.determinism_max_delta <= 1e-6


def test_order_dependent_cheat_is_caught(tmp_path):
    out = _mk_bench(tmp_path)
    adapter = _write_adapter(tmp_path,
        "with open(a.output,'w') as f:\n"
        "    for i, c in enumerate(ids):\n"   # score depends on presentation order
        "        f.write(json.dumps({'case_id': c, 'score': i/len(ids)})+'\\n')\n")
    with pytest.raises(ValueError, match="DETERMINISM FAILURE"):
        run_hermetic(out, adapter)


def test_incomplete_answers_are_void(tmp_path):
    out = _mk_bench(tmp_path)
    adapter = _write_adapter(tmp_path,
        "with open(a.output,'w') as f:\n"
        "    for c in ids[:-1]:\n"            # skips one case
        "        f.write(json.dumps({'case_id': c, 'score': 0.5})+'\\n')\n")
    with pytest.raises(ValueError, match="did not score every case"):
        run_hermetic(out, adapter)


def test_mias_info_parser():
    info = ("mdb001 G CIRC B 535 425 197\n"
            "mdb003 D NORM\n"
            "mdb005 F CIRC B 477 133 30\n"
            "mdb005 F CIRC B 500 168 26\n"
            "mdb023 G CIRC M 538 681 29\n"
            "garbage line\n")
    rows = parse_info(info)
    assert ("mdb023", "CIRC", "M") in rows
    assert ("mdb003", "NORM", None) in rows
    assert len([r for r in rows if r[0] == "mdb005"]) == 2
