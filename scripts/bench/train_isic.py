"""ISIC 2018 Task 3: 7-class dermoscopy classifier (official metric: balanced accuracy).

Train on HAM10000 (10,015 images), select on a stratified 10% val split,
evaluate ONCE on the official 1,512-image test set with the public ground
truth. Comparator: the live leaderboard (rank 1: MetaOptima, 0.885 BMA).
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import torch, timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
ROOT = Path("data/raw/ISIC2018")


def read_gt(csv_path):
    rows = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            rows[r["image"]] = int(np.argmax([float(r[c]) for c in CLASSES]))
    return rows


class Derm(Dataset):
    def __init__(self, items, img_dir, train, size=380):
        self.items, self.dir, self.train, self.size = items, img_dir, train, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, y = self.items[i]
        img = Image.open(self.dir / f"{name}.jpg").convert("RGB")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            if torch.rand(()) < 0.5: x = torch.flip(x, (2,))
            if torch.rand(()) < 0.5: x = torch.flip(x, (1,))
            k = int(torch.randint(0, 4, ()))
            if k: x = torch.rot90(x, k, (1, 2))
            x = (x * float(torch.empty(()).uniform_(0.85, 1.15))
                 + float(torch.empty(()).uniform_(-0.06, 0.06))).clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std, y


def balanced_acc(y, pred, k=7):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(k) if (y == c).any()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--arch", default="tf_efficientnet_b3.ns_jft_in1k")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    gt = read_gt(next((ROOT / "train_gt").rglob("*.csv")))
    items = sorted(gt.items())
    rng = np.random.default_rng(0)
    val_idx = set()
    by_c = {}
    for i, (n, y) in enumerate(items): by_c.setdefault(y, []).append(i)
    for c, idxs in by_c.items():
        val_idx.update(rng.choice(idxs, max(1, len(idxs) // 10), replace=False).tolist())
    tr_items = [it for i, it in enumerate(items) if i not in val_idx]
    va_items = [it for i, it in enumerate(items) if i in val_idx]
    img_dir = next(d for d in (ROOT / "train_input").rglob("*") if d.is_dir() and list(d.glob("*.jpg"))) \
        if not list((ROOT / "train_input").glob("*.jpg")) else ROOT / "train_input"
    print(f"[isic] train={len(tr_items)} val={len(va_items)} dir={img_dir} dev={dev}", flush=True)

    ycnt = np.bincount([y for _, y in tr_items], minlength=7)
    w = torch.as_tensor([1.0 / ycnt[y] for _, y in tr_items])
    tl = DataLoader(Derm(tr_items, img_dir, True), batch_size=args.batch, num_workers=6,
                    sampler=WeightedRandomSampler(w, len(tr_items)), drop_last=True,
                    persistent_workers=True)
    vl = DataLoader(Derm(va_items, img_dir, False), batch_size=args.batch, num_workers=4,
                    persistent_workers=True)

    net = timm.create_model(args.arch, pretrained=True, num_classes=7).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    run = Path("runs/isic2018"); run.mkdir(parents=True, exist_ok=True)
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
        print(f"[isic] ep{ep:02d} loss={tot / n:.4f} val_bma={bma:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if bma > best:
            best = bma
            torch.save({"model": net.state_dict(), "arch": args.arch, "epoch": ep,
                        "val_bma": bma}, run / "best_model.pt")
    print(f"[isic] done best val BMA={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
