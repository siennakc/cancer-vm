"""Score the official ISIC 2018 Task 3 test set (1,512 images) once, with TTA."""
import csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
import torch, timm
from PIL import Image

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ROOT = Path("data/raw/ISIC2018")
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load("runs/isic2018/best_model.pt", map_location="cpu", weights_only=False)
net = timm.create_model(ck["arch"], pretrained=False, num_classes=7)
net.load_state_dict(ck["model"]); net = net.eval().to(dev)

gt_csv = next((ROOT / "test_gt").rglob("*.csv"))
gt = {}
for r in csv.DictReader(open(gt_csv)):
    gt[r["image"]] = int(np.argmax([float(r[c]) for c in CLASSES]))
img_dir = ROOT / "test_input"
if not list(img_dir.glob("*.jpg")):
    img_dir = next(d for d in img_dir.rglob("*") if d.is_dir() and list(d.glob("*.jpg")))
names = sorted(gt)
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

probs = []
with torch.no_grad():
    for i in range(0, len(names), 32):
        batch = []
        for n in names[i:i + 32]:
            img = Image.open(img_dir / f"{n}.jpg").convert("RGB").resize((380, 380), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
            batch.append((x - mean) / std)
        xb = torch.stack(batch).to(dev)
        p = torch.softmax(net(xb), 1)
        p = p + torch.softmax(net(torch.flip(xb, (3,))), 1)   # hflip TTA
        probs.append((p / 2).cpu().numpy())
probs = np.concatenate(probs)
pred = probs.argmax(1)
y = np.array([gt[n] for n in names])
bma = float(np.mean([np.mean(pred[y == c] == c) for c in range(7) if (y == c).any()]))
per = {CLASSES[c]: {"n": int((y == c).sum()), "recall": float(np.mean(pred[y == c] == c))}
       for c in range(7)}
out = {"benchmark": "ISIC 2018 Task 3 (official test, public GT)", "n": len(y),
       "balanced_multiclass_accuracy": round(bma, 4), "per_class": per,
       "arch": ck["arch"], "val_bma": ck["val_bma"], "tta": "hflip"}
Path("results/bench_isic2018").mkdir(parents=True, exist_ok=True)
Path("results/bench_isic2018/report.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
