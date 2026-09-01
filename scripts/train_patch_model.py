"""Train the 5-class patch classifier (the Shen et al. patch stage).

ResNet-50 on 224px patches: background / benign-calc / malignant-calc /
benign-mass / malignant-mass. The trained backbone then serves two consumers:

1. ``scripts/finetune_encoder.py --init-weights runs/patch_v1/best_model.pt``
   — whole-image training warm-started from lesion-aware features (the core
   of the published 0.88 recipe v4 is missing);
2. ``oncoscope.models.patch_detector.PatchDetector`` — the localizing
   detector the harness A/B diagnosed as absent.

Discipline mirrors finetune_encoder.py: train on the ``train`` shard, epoch
selection on the ``calibration`` shard (macro one-vs-rest AUROC), class-
balanced sampling, label-invariant augs, full checkpoint every epoch for
``--resume``. The build report's splits sha is stamped into every checkpoint.
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

from oncoscope.data.patches import verify_shard_provenance
from oncoscope.data.roi import PATCH_CLASSES
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import auroc

MEAN, STD = 0.449, 0.226  # grayscale stats, same as the whole-image recipe


class PatchDataset(Dataset):
    """Reads the memmapped uint8 shard; normalizes and augments on the fly.

    The memmap is opened lazily PER PROCESS: DataLoader workers on macOS are
    spawned and receive a pickle of this dataset, and pickling an np.memmap
    materializes the whole array — 6 workers x a multi-GB shard. Keeping only
    the path in __init__ makes the pickle a few bytes; each worker opens its
    own read-only view on first use.
    """

    def __init__(self, shard_dir: Path, split: str, train: bool):
        self.patches_path = shard_dir / f"patches_{split}.npy"
        meta = json.loads((shard_dir / f"meta_{split}.json").read_text())
        self.classes = np.array([m["cls"] for m in meta], dtype=np.int64)
        self.train = train
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.load(self.patches_path, mmap_mode="r")
        return self._arr

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_arr"] = None  # never pickle the memmap itself
        return state

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.arr[i], dtype=np.float32) / 255.0)[None]
        if self.train:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(2,))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(1,))
            k = int(torch.randint(0, 4, ()))
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
            x = (x * float(torch.empty(()).uniform_(0.9, 1.1))
                 + float(torch.empty(()).uniform_(-0.05, 0.05))).clamp(0, 1)
        x = (x - MEAN) / STD
        return x.repeat(3, 1, 1), int(self.classes[i])


def build_model(n_classes: int = len(PATCH_CLASSES)):
    net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Linear(2048, n_classes)
    return net


@torch.no_grad()
def eval_macro_auroc(net, loader, device, n_classes: int) -> tuple[float, float]:
    """(macro one-vs-rest AUROC over present classes, plain accuracy)."""
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(net(x.to(device)), dim=1).float().cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    per_class = [auroc((y == k).astype(int), p[:, k])
                 for k in range(n_classes) if 0 < (y == k).sum() < len(y)]
    macro = float(np.mean(per_class)) if per_class else float("nan")
    return macro, float((p.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="data/cache/patches224_v1")
    ap.add_argument("--splits", default="data/processed/splits_v2.json")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run", default="runs/patch_v1")
    ap.add_argument("--allow-tainted-shards", action="store_true",
                    help="train on shards built with --allow-unquarantined "
                         "(synthetic smoke tests only — never real data)")
    args = ap.parse_args()

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shards)
    # The last gate before gradients: re-derives taint, shard-hash, and
    # per-patch split membership from the manifest instead of trusting the
    # directory. Refuses stale, tainted, edited, or cross-manifest shards.
    report = verify_shard_provenance(
        shard_dir, load_manifest(args.splits),
        allow_tainted=args.allow_tainted_shards)
    tainted = bool(args.allow_tainted_shards)
    if tainted:
        print("[patch] WARNING: training on TAINTED shards — smoke test only, "
              "this checkpoint must never initialize a real run", flush=True)
    for needed in ("train", "calibration"):
        if needed not in report.get("counts", {}):
            raise SystemExit(
                f"shard directory has no {needed!r} shard (build produced zero "
                f"{needed} patches) — rebuild before training")
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu")

    tr = PatchDataset(shard_dir, "train", train=True)
    cal = PatchDataset(shard_dir, "calibration", train=False)
    counts = np.bincount(tr.classes, minlength=len(PATCH_CLASSES)).astype(np.float64)
    weights = np.where(counts[tr.classes] > 0, 1.0 / np.maximum(counts[tr.classes], 1), 0.0)
    print(f"[patch] train={len(tr)} cal={len(cal)} device={device} "
          f"class counts={counts.astype(int).tolist()}", flush=True)

    train_loader = DataLoader(
        tr, batch_size=args.batch, num_workers=6, drop_last=True,
        sampler=WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(tr)),
        persistent_workers=True)
    cal_loader = DataLoader(cal, batch_size=args.batch, num_workers=4,
                            persistent_workers=True)

    net = build_model().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    start, best, history = 0, -1.0, []

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        # A resume is a continuation of ONE fitting run: the checkpoint's
        # manifest must be the same one the current shards verified against,
        # or N earlier epochs of another membership launder into this run.
        if ck.get("splits_sha256") != report["splits_sha256"]:
            raise SystemExit(
                f"--resume checkpoint was trained under splits sha "
                f"{str(ck.get('splits_sha256'))[:12]}…, current shards are "
                f"{report['splits_sha256'][:12]}… — refusing the cross-manifest resume")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best, history = ck["epoch"] + 1, ck["best_cal_macro_auroc"], ck["history"]
        # Taint is sticky: epochs trained on tainted shards stay tainted no
        # matter what the current invocation's flags say.
        tainted = tainted or bool(ck.get("tainted"))
        print(f"[patch] resumed at epoch {start} (best {best:.4f})", flush=True)

    for epoch in range(start, args.epochs):
        net.train()
        t0, seen, running = time.time(), 0, 0.0
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            running += float(loss) * len(y)
            seen += len(y)
        sched.step()
        macro, acc = eval_macro_auroc(net, cal_loader, device, len(PATCH_CLASSES))
        history.append({"epoch": epoch, "loss": running / seen,
                        "cal_macro_auroc": macro, "cal_acc": acc,
                        "secs": round(time.time() - t0, 1)})
        print(f"[patch] epoch {epoch:02d} loss={running / seen:.4f} "
              f"cal_macro_auroc={macro:.4f} acc={acc:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if macro > best:
            best = macro
            torch.save({"model": net.state_dict(), "epoch": epoch,
                        "cal_macro_auroc": macro, "classes": list(PATCH_CLASSES),
                        "splits_sha256": report["splits_sha256"],
                        "init_lineage": [],  # ImageNet start: lineage begins here
                        "tainted": tainted},
                       run / "best_model.pt")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best_cal_macro_auroc": best, "history": history,
                    "splits_sha256": report["splits_sha256"],
                    "tainted": tainted},
                   run / "checkpoint.pt")
        (run / "history.json").write_text(json.dumps(history, indent=1))

    print(f"[patch] done. best cal macro AUROC={best:.4f} -> {run}/best_model.pt",
          flush=True)


if __name__ == "__main__":
    main()
