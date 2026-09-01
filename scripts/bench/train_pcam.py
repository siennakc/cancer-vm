"""PCam patch classifier: stage 1 of the CAMELYON16 pipeline.

PatchCamelyon (Veeling et al., CC0): 96x96 H&E patches derived from CAMELYON16
TRAINING slides only for train/valid — its test split derives from C16 TEST
slides and is therefore NEVER touched here (that would contaminate the
benchmark we are about to take). ResNet-50 for architecture continuity with
the mammography line.
"""
from __future__ import annotations
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import pyarrow.parquet as pq
import torch, torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path("data/raw/PCam")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Shards(Dataset):
    """Loads all patches of the given parquet shards into memory as uint8."""

    def __init__(self, files, train):
        self.train = train
        xs, ys = [], []
        for f in files:
            t = pq.read_table(f)
            for img, lab in zip(t.column("image"), t.column("label")):
                d = img.as_py()
                raw = d["bytes"] if isinstance(d, dict) else d
                xs.append(np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), np.uint8))
                ys.append(int(lab.as_py() if hasattr(lab, "as_py") else lab))
        self.x = np.stack(xs); self.y = np.array(ys, np.int64)
        print(f"[pcam] loaded {len(self.y)} patches from {len(files)} shards "
              f"(pos={self.y.mean():.3f})", flush=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i].astype(np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            if torch.rand(()) < 0.5: x = torch.flip(x, (2,))
            if torch.rand(()) < 0.5: x = torch.flip(x, (1,))
            k = int(torch.randint(0, 4, ()))
            if k: x = torch.rot90(x, k, (1, 2))
            x = (x * float(torch.empty(()).uniform_(0.9, 1.1))
                 + float(torch.empty(()).uniform_(-0.05, 0.05))).clamp(0, 1)
        return (x - MEAN) / STD, float(self.y[i])


def auroc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-shards", type=int, default=13)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_files = sorted(ROOT.glob("train-*.parquet"))[: args.train_shards]
    valid_files = sorted(list(ROOT.glob("valid*.parquet")) + list(ROOT.glob("validation*.parquet")))
    assert train_files, "no PCam train shards on disk"
    tr = Shards(train_files, True)
    va = Shards(valid_files, False) if valid_files else None

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=6,
                    drop_last=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=args.batch, num_workers=4) if va else None

    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Linear(2048, 1)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lf = torch.nn.BCEWithLogitsLoss()
    run = Path("runs/pcam"); run.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, None, []
    for ep in range(args.epochs):
        net.train(); t0 = time.time(); tot = n = 0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            loss = lf(net(x.to(dev)).squeeze(1), y.float().to(dev))
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(y); n += len(y)
        sched.step()
        row = {"epoch": ep, "loss": round(tot / n, 4), "secs": round(time.time() - t0, 1)}
        msg = f"[pcam] ep{ep} loss={tot / n:.4f}"
        if vl:
            net.eval(); ys, ss = [], []
            with torch.no_grad():
                for x, y in vl:
                    ss.append(net(x.to(dev)).squeeze(1).cpu().numpy()); ys.append(y.numpy())
            a = auroc(np.concatenate(ys), np.concatenate(ss))
            row["val_auroc"] = round(a, 4)
            msg += f" val_auroc={a:.4f}"
            if a > best:
                best, best_epoch = a, ep
                torch.save({"model": net.state_dict(), "epoch": ep, "val_auroc": a},
                           run / "best_model.pt")
        else:
            torch.save({"model": net.state_dict(), "epoch": ep}, run / "best_model.pt")
        history.append(row)
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)

    # The committed artifact behind any quoted number — a claim that exists
    # only in a commit message is not a result (audit finding, 2026-08-31).
    report = {
        "benchmark": "PCam (PatchCamelyon) — VALIDATION split only",
        "protocol_note": (
            "PCam's own test split derives from CAMELYON16 TEST slides, which "
            "this pipeline holds out for the slide-level benchmark — so the "
            "official PCam test is deliberately never scored here. This number "
            "is patch-level validation AUROC, a stage-1 health check, not a "
            "leaderboard claim."),
        "config": vars(args),
        "n_train": len(tr), "n_valid": (len(va) if va else 0),
        "train_prevalence": round(float(tr.y.mean()), 4),
        "best_val_auroc": (round(best, 4) if best >= 0 else None),
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint": str(run / "best_model.pt"),
    }
    out = Path("results/bench_pcam/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[pcam] done best val AUROC={best:.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
