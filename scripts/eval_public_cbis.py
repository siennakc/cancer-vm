"""Evaluate on the official CBIS-DDSM test split — the public benchmark, as-is.

Protocol (matches the literature): image-level malignant-vs-benign over the
official mass+calc test CSVs (BENIGN_WITHOUT_CALLBACK -> negative), standard
AUROC with patient-clustered 95% CI, plus BenchX-style subgroup slices
(abnormality type, breast density, view). No custom staging or sealing — the
only integrity requirement is the splits_v2 quarantine: models scored here must
never have fit on official-test patients.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import (auroc, clustered_bootstrap_ci, ece_adaptive,
                                    sensitivity_at_specificity)
from oncoscope.models.head import LogisticHead

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)            # embedding cache tag
ap.add_argument("--head", required=True)           # head.json path
ap.add_argument("--name", required=True)           # report name
ap.add_argument("--encoder-checkpoint", default=None,
                help="best_model.pt behind the embedding tag; verifies the "
                     "encoder's manifest sha, warm-start lineage, and taint "
                     "flag against splits_v2 before any number is written")
args = ap.parse_args()

cases = read_case_table("data/processed/cases_v1.jsonl")
splits = load_manifest("data/processed/splits_v2.json")

# Provenance at the published-number boundary: embeddings arrive by tag with
# no history, so the check has to be asked for by handing over the encoder
# checkpoint. Warm starts make this matter: the encoder's OWN manifest and
# every init_lineage manifest can each independently contaminate the bench.
lineage_note = "unverified (no --encoder-checkpoint given)"
if args.encoder_checkpoint:
    import torch
    ck = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    if ck.get("tainted"):
        raise SystemExit("encoder checkpoint is TAINTED (smoke shards) — refusing")
    shas = [("self", ck.get("splits_sha256"))] + [
        (e.get("path", "?"), e.get("splits_sha256"))
        for e in ck.get("init_lineage", [])
    ]
    bad = [(who, s) for who, s in shas if s != splits.sha256]
    if bad:
        raise SystemExit(
            f"encoder lineage includes manifests other than splits_v2 "
            f"({splits.sha256[:12]}…): {[(w, str(s)[:12]) for w, s in bad]} — "
            "a fitting stage of this model may have touched bench patients")
    lineage_note = f"verified: {len(shas)} manifest(s), all splits_v2"
    print(f"[eval] encoder lineage {lineage_note}", flush=True)
else:
    print("[eval] WARNING: encoder lineage unverified — pass --encoder-checkpoint "
          "so the quarantine is checked in code rather than by memory", flush=True)
bench = [c for c in cases if splits.split_of(f"{c.site}/{c.patient_id}") == "public_bench"]
assert len({c.patient_id for c in bench}) == 349, "unexpected bench membership"

head = LogisticHead.load(args.head)
X = np.stack([np.load(f"data/embeddings/{args.tag}/{c.case_id}.npy") for c in bench])
y = np.array([c.label for c in bench], float)
p = head.predict_proba(X)
pids = [c.patient_id for c in bench]

def block(mask, label):
    if mask.sum() < 20 or len(set(y[mask])) < 2:
        return None
    sub_pids = [pid for pid, m in zip(pids, mask) if m]
    a = clustered_bootstrap_ci(y[mask], p[mask], sub_pids, auroc, iterations=2000)
    return {"slice": label, "n": int(mask.sum()), "prevalence": float(y[mask].mean()),
            "auroc": round(a[0], 4), "auroc_ci95": [round(a[1], 4), round(a[2], 4)]}

full = block(np.ones(len(y), bool), "official test (all)")
s = clustered_bootstrap_ci(y, p, pids, sensitivity_at_specificity, iterations=2000)
full.update({"sens_at_spec96": round(s[0], 4),
             "sens_at_spec96_ci95": [round(s[1], 4), round(s[2], 4)],
             "ece_adaptive": round(float(ece_adaptive(y, p)), 4)})

slices = []
ab = np.array([c.abnormality for c in bench])
dens = np.array([c.density_band or "?" for c in bench])
view = np.array([c.view or "?" for c in bench])
for v in ("mass", "calcification"):
    slices.append(block(ab == v, f"abnormality={v}"))
for v in "abcd":
    slices.append(block(dens == v, f"density={v}"))
for v in ("CC", "MLO"):
    slices.append(block(view == v, f"view={v}"))
slices = [x for x in slices if x]

report = {"benchmark": "CBIS-DDSM official test split",
          "protocol": "image-level malignant vs benign(+BWC), official mass+calc test CSVs",
          "model": args.name, "quarantine": "splits_v2 public_bench (349 patients, 709 images)",
          "encoder_lineage": lineage_note,
          "overall": full, "subgroups": slices}
out = Path(f"results/public_cbis/{args.name}.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=1))
print(json.dumps(report["overall"], indent=1))
for row in slices:
    print(f"  {row['slice']:22} n={row['n']:4} auroc={row['auroc']:.3f} {row['auroc_ci95']}")
