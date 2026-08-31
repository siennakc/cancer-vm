"""Score ISIC 2018 official test with the v2 model + 8-view TTA + Shades of Gray."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/bench")
import torch, timm
from PIL import Image
from train_isic_v2 import CLASSES, MEAN, STD, balanced_acc, shades_of_gray

ROOT = Path("data/raw/ISIC2018")
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load("runs/isic2018_v2/best_model.pt", map_location="cpu", weights_only=False)
net = timm.create_model(ck["arch"], pretrained=False, num_classes=7)
net.load_state_dict(ck["model"]); net = net.eval().to(dev)
S = ck["size"]

gt = {r["image"]: int(np.argmax([float(r[c]) for c in CLASSES]))
      for r in csv.DictReader(open(next((ROOT / "test_gt").rglob("*.csv"))))}
img_dir = ROOT / "test_input/ISIC2018_Task3_Test_Input"
names = sorted(gt)

probs = []
with torch.no_grad():
    for i in range(0, len(names), 16):
        batch = []
        for n in names[i:i + 16]:
            img = Image.open(img_dir / f"{n}.jpg").convert("RGB").resize((S, S), Image.BILINEAR)
            x = torch.from_numpy(shades_of_gray(np.asarray(img, np.float32)) / 255.0).permute(2, 0, 1)
            batch.append((x - MEAN) / STD)
        xb = torch.stack(batch).to(dev)
        acc = torch.zeros(len(batch), 7, device=dev)
        for k in range(4):
            xr = torch.rot90(xb, k, (2, 3))
            acc += torch.softmax(net(xr), 1)
            acc += torch.softmax(net(torch.flip(xr, (3,))), 1)
        probs.append((acc / 8).cpu().numpy())
probs = np.concatenate(probs)
# prior-corrected argmax: divide by train class priors (Bayes-optimal for
# balanced/mean-recall metrics; spec item C1)
train_gt = {r["image"]: int(np.argmax([float(r[c]) for c in CLASSES]))
            for r in csv.DictReader(open(next((ROOT / "train_gt").rglob("*.csv"))))}
priors = np.bincount(list(train_gt.values()), minlength=7).astype(float)
priors /= priors.sum()
pred = (probs / priors[None, :]).argmax(1)
pred_plain = probs.argmax(1)
y = np.array([gt[n] for n in names])
bma = balanced_acc(y, pred)
out = {"benchmark": "ISIC 2018 Task 3 (official test, public GT)", "model": "v2",
       "arch": ck["arch"], "input": S, "color_constancy": "shades_of_gray_p6",
       "tta": "8-view (rot90 x hflip)", "prior_corrected_argmax": True, "n": len(y),
       "bma_plain_argmax": round(balanced_acc(y, pred_plain), 4),
       "balanced_multiclass_accuracy": round(bma, 4), "val_bma": ck["val_bma"],
       "per_class": {CLASSES[c]: {"n": int((y == c).sum()),
                                  "recall": float(np.mean(pred[y == c] == c))}
                     for c in range(7)}}
Path("results/bench_isic2018").mkdir(parents=True, exist_ok=True)
Path("results/bench_isic2018/report_v2.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
