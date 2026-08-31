"""ISIC v2: the overnight push from 0.753 toward 0.80+.

Levers (standard in the winning recipes): Shades-of-Gray color constancy,
EfficientNet-B4 at 448, longer cosine schedule, heavier aug, and 8-view TTA
at eval time (rot90 x flips). Same stratified val protocol; official test
scored once by eval_isic_v2.
"""
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import torch, timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ROOT = Path("data/raw/ISIC2018")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def shades_of_gray(img: np.ndarray, p: int = 6) -> np.ndarray:
    """Color-constancy normalization — the known big lever on dermoscopy."""
    img = img.astype(np.float32)
    norm = np.power(np.power(img, p).mean(axis=(0, 1)), 1.0 / p)
    norm = np.maximum(norm, 1e-6)
    img = img * (norm.mean() / norm)[None, None, :]
    return np.clip(img, 0, 255)


def read_gt(csv_path):
    return {r["image"]: int(np.argmax([float(r[c]) for c in CLASSES]))
            for r in csv.DictReader(open(csv_path))}


class Derm(Dataset):
    def __init__(self, items, img_dir, train, size=448):
        self.items, self.dir, self.train, self.size = items, img_dir, train, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, y = self.items[i]
        img = Image.open(self.dir / f"{name}.jpg").convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = shades_of_gray(np.asarray(img, np.float32))
        x = torch.from_numpy(arr / 255.0).permute(2, 0, 1)
        if self.train:
            if torch.rand(()) < 0.5: x = torch.flip(x, (2,))
            if torch.rand(()) < 0.5: x = torch.flip(x, (1,))
            k = int(torch.randint(0, 4, ()))
            if k: x = torch.rot90(x, k, (1, 2))
            # random resized crop-ish zoom
            z = float(torch.empty(()).uniform_(0.8, 1.0))
            s = int(self.size * z)
            if s < self.size:
                ox = int(torch.randint(0, self.size - s + 1, ()))
                oy = int(torch.randint(0, self.size - s + 1, ()))
                x = torch.nn.functional.interpolate(
                    x[None, :, oy:oy + s, ox:ox + s], size=(self.size, self.size),
                    mode="bilinear", align_corners=False)[0]
            x = (x * float(torch.empty(()).uniform_(0.85, 1.15))
                 + float(torch.empty(()).uniform_(-0.05, 0.05))).clamp(0, 1)
        return (x - MEAN) / STD, y


def balanced_acc(y, pred, k=7):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(k) if (y == c).any()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--arch", default="tf_efficientnet_b4.ns_jft_in1k")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    gt = read_gt(next((ROOT / "train_gt").rglob("*.csv")))
    items = sorted(gt.items())
    rng = np.random.default_rng(0)
    by_c = {}
    for i, (n, y) in enumerate(items): by_c.setdefault(y, []).append(i)
    val_idx = set()
    for c, idxs in by_c.items():
        val_idx.update(rng.choice(idxs, max(1, len(idxs) // 10), replace=False).tolist())
    tr_items = [it for i, it in enumerate(items) if i not in val_idx]
    va_items = [it for i, it in enumerate(items) if i in val_idx]
    img_dir = ROOT / "train_input/ISIC2018_Task3_Training_Input"
    print(f"[isic2] train={len(tr_items)} val={len(va_items)} arch={args.arch}", flush=True)

    ycnt = np.bincount([y for _, y in tr_items], minlength=7)
    w = torch.as_tensor([1.0 / ycnt[y] for _, y in tr_items])
    tl = DataLoader(Derm(tr_items, img_dir, True), batch_size=args.batch, num_workers=7,
                    sampler=WeightedRandomSampler(w, len(tr_items)), drop_last=True,
                    persistent_workers=True)
    vl = DataLoader(Derm(va_items, img_dir, False), batch_size=args.batch, num_workers=4,
                    persistent_workers=True)

    net = timm.create_model(args.arch, pretrained=True, num_classes=7).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    run = Path("runs/isic2018_v2"); run.mkdir(parents=True, exist_ok=True)
    best = -1
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = n = 0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            loss = lf(net(x.to(dev)), y.to(dev))
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(y); n += len(y)
        sched.step()
        net.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in vl:
                ps.append(net(x.to(dev)).argmax(1).cpu().numpy()); ys.append(y.numpy())
        bma = balanced_acc(np.concatenate(ys), np.concatenate(ps))
        print(f"[isic2] ep{ep:02d} loss={tot / n:.4f} val_bma={bma:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if bma > best:
            best = bma
            torch.save({"model": net.state_dict(), "arch": args.arch, "epoch": ep,
                        "val_bma": bma, "size": 448, "sog": True}, run / "best_model.pt")
    print(f"[isic2] done best val BMA={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
