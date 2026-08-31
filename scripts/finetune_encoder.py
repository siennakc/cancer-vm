"""Fine-tune the encoder end-to-end on the train split (the real training run).

Upgrades the A11 control arm's frozen ImageNet features to a domain-adapted
encoder. The network's classifier head here is scaffolding: what ships is the
*backbone*, re-exported as FrozenEncoder embeddings, with the calibrated
LogisticHead stack refit on top — serving, gates, and the sealed scorer see
the exact same contract as baseline v1.

Discipline:
- trains on ``train`` only; epoch selection on ``calibration`` AUROC
  (threshold/slice_discovery/test remain untouched here)
- class-balanced sampling; augs are label-invariant for malignancy
  (hflip is safe for *this* task — the laterality pitfall concerns flip-TTA
  on laterality-sensitive outputs, not train-time augmentation)
- full checkpoint (model+optimizer+scheduler+epoch+splits sha) saved every
  epoch so training can resume: ``--resume runs/finetune_v2/checkpoint.pt``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import auroc

CACHE = Path("data/cache/render448")
MEAN, STD = 0.449, 0.226  # ImageNet grayscale-equivalent


class RenderDataset(Dataset):
    def __init__(self, cases, train: bool):
        self.cases, self.train = cases, train

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        x = torch.from_numpy(np.load(CACHE / f"{c.case_id}.npy").astype(np.float32))[None]
        if self.train:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(2,))
            ang = float(torch.empty(()).uniform_(-10, 10))
            scale = float(torch.empty(()).uniform_(0.9, 1.1))
            x = torchvision.transforms.functional.affine(
                x, angle=ang, translate=(0, 0), scale=scale, shear=(0.0, 0.0),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            x = (x * float(torch.empty(()).uniform_(0.9, 1.1))
                 + float(torch.empty(()).uniform_(-0.05, 0.05))).clamp(0, 1)
        x = (x - MEAN) / STD
        return x.repeat(3, 1, 1), float(c.label)


def build_model():
    net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Linear(2048, 1)
    return net


@torch.no_grad()
def eval_auroc(net, loader, device):
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(net(x.to(device)).squeeze(1).float().cpu().numpy())
        ys.append(y.numpy())
    return auroc(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--splits", type=str, default="data/processed/splits_v1.json")
    ap.add_argument("--run", type=str, default="runs/finetune_v2")
    args = ap.parse_args()

    global RUN
    RUN = Path(args.run)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    RUN.mkdir(parents=True, exist_ok=True)

    cases = read_case_table("data/processed/cases_v1.jsonl")
    splits = load_manifest(args.splits)
    of = lambda c: splits.split_of(f"{c.site}/{c.patient_id}")
    tr = [c for c in cases if of(c) == "train"]
    cal = [c for c in cases if of(c) == "calibration"]
    print(f"[ft] train={len(tr)} cal={len(cal)} device={device}", flush=True)

    y = np.array([c.label for c in tr], dtype=np.float64)
    w = np.where(y == 1, 0.5 / max(y.mean(), 1e-9), 0.5 / max(1 - y.mean(), 1e-9))
    train_loader = DataLoader(
        RenderDataset(tr, train=True), batch_size=args.batch, num_workers=6,
        sampler=WeightedRandomSampler(torch.as_tensor(w), num_samples=len(tr)),
        drop_last=True, persistent_workers=True)
    cal_loader = DataLoader(RenderDataset(cal, train=False), batch_size=args.batch,
                            num_workers=4, persistent_workers=True)

    net = build_model().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start, best = 0, -1.0
    history = []

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best, history = ck["epoch"] + 1, ck["best_cal_auroc"], ck["history"]
        print(f"[ft] resumed at epoch {start} (best {best:.4f})", flush=True)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(start, args.epochs):
        net.train()
        t0, seen, running = time.time(), 0, 0.0
        for x, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x.to(device)).squeeze(1), yb.float().to(device))
            loss.backward()
            opt.step()
            running += float(loss) * len(yb)
            seen += len(yb)
        sched.step()
        cal_auc = eval_auroc(net, cal_loader, device)
        history.append({"epoch": epoch, "loss": running / seen, "cal_auroc": cal_auc,
                        "secs": round(time.time() - t0, 1)})
        print(f"[ft] epoch {epoch:02d} loss={running / seen:.4f} "
              f"cal_auroc={cal_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if cal_auc > best:
            best = cal_auc
            torch.save({"model": net.state_dict(), "epoch": epoch,
                        "cal_auroc": cal_auc,
                        "splits_sha256": splits.sha256}, RUN / "best_model.pt")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best_cal_auroc": best, "history": history,
                    "splits_sha256": splits.sha256}, RUN / "checkpoint.pt")
        (RUN / "history.json").write_text(json.dumps(history, indent=1))

    print(f"[ft] done. best cal AUROC={best:.4f} -> {RUN}/best_model.pt", flush=True)


if __name__ == "__main__":
    main()
