"""Local demo server: the v4 model, served honestly.

Serves the MODEL ALONE — the measured champion. The agent-harness path was
A/B-tested at −0.071 AUROC versus this configuration (results/ab_harness/),
so a demo routing through it would showcase the worse system; the harness
returns to serving if and when the patch-detector rematch clears the gate.

Same preprocessing module as training (axiom of T-1.1: one decoder, one
render path — train/serve skew is the silent killer). Every response carries
the model's honest limits alongside its score.

Run:  .venv/bin/python -m uvicorn oncoscope.serving.app:app --port 8321
      (weights: gh release download weights-v4 --pattern best_model.pt
       --dir runs/posttrain_v4)

**Not a medical device.** Research and education only.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("install serving extras: pip install -e '.[serving]'") from exc

from ..data.dicom_canonical import load_canonical
from ..models.encoder import FrozenEncoder
from ..models.head import LogisticHead

WEIGHTS = Path("runs/posttrain_v4/best_model.pt")
HEAD = Path("results/posttrain_v4/head.json")
OPERATING_POINT = Path("results/posttrain_v4/operating_point.json")

DISCLAIMER = ("NOT A MEDICAL DEVICE. Research/education demo only. No output "
              "may inform any diagnosis, screening, or treatment decision.")
LIMITS = [
    "Calibrated on biopsy-enriched cohorts (~58% malignant); probabilities do "
    "not transfer to screening prevalence (~0.5%) or to new sites without "
    "recalibration — externally (MIAS) calibration error reached ECE 0.49.",
    "Weak slice: masses in heterogeneously dense (BI-RADS c) breasts — CI "
    "includes chance there.",
    "Trained on CBIS-DDSM (US, digitized film) and CMMD (China, FFDM); "
    "anything else is out of distribution.",
]

app = FastAPI(title="oncoscope-demo", version="0.2.0")
_state: dict = {}


def _model():
    if not _state:
        if not WEIGHTS.exists():
            raise RuntimeError(
                f"{WEIGHTS} missing — run: gh release download weights-v4 "
                f"--pattern best_model.pt --dir {WEIGHTS.parent}")
        _state["encoder"] = FrozenEncoder(
            tag="serve_v4", weights_path=str(WEIGHTS), input_size=(1152, 896),
            normalize=False, mean=(0.449,) * 3, std=(0.226,) * 3)
        _state["head"] = LogisticHead.load(HEAD)
        _state["op"] = json.loads(OPERATING_POINT.read_text())
    return _state


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "weights_present": WEIGHTS.exists()}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    state = _model()
    with tempfile.NamedTemporaryFile(suffix=".dcm") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        canonical = load_canonical(tmp.name)
    prob = float(state["head"].predict_proba(
        state["encoder"].embed_batch([np.asarray(canonical.pixels)]))[0])
    threshold = state["op"]["threshold"]
    return {
        "disclaimer": DISCLAIMER,
        "model": "oncoscope v4 (high-res post-train, official CBIS test AUROC "
                 "0.771 [0.726-0.815])",
        "calibrated_probability": round(prob, 4),
        "operating_point": {
            "threshold": round(threshold, 4),
            "target_specificity": state["op"]["target_specificity"],
            "flag": "suspicious at the 96%-specificity operating point"
                    if prob > threshold else
                    "below the 96%-specificity operating point",
        },
        "known_limits": LIMITS,
        "input": {
            "modality": canonical.modality,
            "laterality": canonical.laterality,
            "view": canonical.view,
            "shape": list(np.asarray(canonical.pixels).shape),
        },
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html><meta charset="utf-8">
<title>Oncoscope demo</title>
<style>
 body{font:16px/1.5 system-ui;margin:0;background:#f6f7fa;color:#1b2130}
 .wrap{max-width:640px;margin:8vh auto;padding:0 20px}
 h1{font-size:26px;margin:0 0 4px} .sub{color:#4c5368;margin:0 0 18px}
 .warn{background:#f6e7ed;border-left:4px solid #a84768;padding:10px 14px;
       border-radius:0 8px 8px 0;font-size:14px;margin-bottom:22px}
 #drop{border:2px dashed #b9bed2;border-radius:12px;padding:44px;text-align:center;
       background:#fff;cursor:pointer;transition:.15s}
 #drop.hot{border-color:#4a4fa0;background:#e7e9f7}
 #out{margin-top:20px;white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace;
      background:#fff;border:1px solid #d9dce8;border-radius:10px;padding:16px;display:none}
 .score{font-size:34px;font-weight:700}
</style>
<div class="wrap">
 <h1>Oncoscope</h1>
 <p class="sub">v4 mammography malignancy model &mdash; local research demo</p>
 <div class="warn"><b>Not a medical device.</b> Research and education only.
  Scores here must never inform a medical decision.</div>
 <div id="drop">Drop a mammogram DICOM here<br>or click to choose a file
  <input id="f" type="file" accept=".dcm,application/dicom" hidden></div>
 <div id="out"></div>
</div>
<script>
const drop=document.getElementById('drop'),f=document.getElementById('f'),
      out=document.getElementById('out');
drop.onclick=()=>f.click();
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>send(ev.dataTransfer.files[0]));
f.onchange=()=>send(f.files[0]);
async function send(file){
 if(!file)return;
 out.style.display='block';out.textContent='analyzing '+file.name+' …';
 const fd=new FormData();fd.append('file',file);
 try{
  const r=await fetch('/analyze',{method:'POST',body:fd});
  const d=await r.json();
  if(!r.ok)throw new Error(JSON.stringify(d));
  out.innerHTML='<div class="score">'+(100*d.calibrated_probability).toFixed(1)+
   '%</div><div>'+d.operating_point.flag+'</div><hr>'+
   '<b>'+d.model+'</b>\\n\\nInput: '+JSON.stringify(d.input)+
   '\\n\\nKnown limits:\\n \\u2022 '+d.known_limits.join('\\n \\u2022 ')+
   '\\n\\n'+d.disclaimer;
 }catch(e){out.textContent='error: '+e.message}
}
</script>"""
