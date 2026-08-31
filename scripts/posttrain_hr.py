"""Post-train v4: progressive high-resolution fine-tune of v3 (literature recipe).

Applies the two changes the published recipes agree on and we lacked:

- **Resolution 448 -> 1152x896** (Shen et al. 2019, Sci Rep — the canonical
  CBIS-DDSM recipe trains whole images at 1152x896; mammographic lesions are
  destroyed by aggressive downsizing).
- **Their augmentation set**: horizontal AND vertical flips, rotation +/-25 deg,
  zoom 0.8-1.2, intensity shift (+/-0.08 on [0,1] ~ their +/-20 on 8-bit),
  plus positive-label smoothing (RSNA-2023 1st place).

Progressive: initialize from v3 best (already domain-adapted at 448), then a
single low-LR all-layers stage (Shen's stage 2: 1e-5), cosine, epoch selection
on calibration AUROC. splits_v2 quarantine respected — v4 stays eligible for
the official CBIS-DDSM test split.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import torch, torchvision
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from oncoscope.data.mammography import read_case_table
from oncoscope.data.splits import load_manifest
from oncoscope.eval.metrics import auroc

CACHE = Path("data/cache/render1152x896")
MEAN, STD = 0.449, 0.226


class HRDataset(Dataset):
    def __init__(self, cases, train):
        self.cases, self.train = cases, train

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        x = torch.from_numpy(np.load(CACHE / f"{c.case_id}.npy").astype(np.float32))[None]
        if self.train:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(2,))
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=(1,))
            ang = float(torch.empty(()).uniform_(-25, 25))
            scale = float(torch.empty(()).uniform_(0.8, 1.2))
            x = torchvision.transforms.functional.affine(
                x, angle=ang, translate=(0, 0), scale=scale, shear=(0.0, 0.0),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            x = (x + float(torch.empty(()).uniform_(-0.08, 0.08))).clamp(0, 1)
        x = (x - MEAN) / STD
        return x.repeat(3, 1, 1), float(c.label)


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
    ap.add_argument("--init", default="runs/finetune_v3/best_model.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pos-smooth", type=float, default=0.9)
    ap.add_argument("--run", default="runs/posttrain_v4")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    run = Path(args.run); run.mkdir(parents=True, exist_ok=True)

    cases = read_case_table("data/processed/cases_v1.jsonl")
    splits = load_manifest("data/processed/splits_v2.json")
    of = lambda c: splits.split_of(f"{c.site}/{c.patient_id}")
    tr = [c for c in cases if of(c) == "train"]
    cal = [c for c in cases if of(c) == "calibration"]
    print(f"[v4] train={len(tr)} cal={len(cal)} device={device} "
          f"res=1152x896 init={args.init}", flush=True)

    y = np.array([c.label for c in tr], float)
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    train_loader = DataLoader(HRDataset(tr, True), batch_size=args.batch,
        sampler=WeightedRandomSampler(torch.as_tensor(w), num_samples=len(tr)),
        num_workers=6, drop_last=True, persistent_workers=True)
    cal_loader = DataLoader(HRDataset(cal, False), batch_size=args.batch,
                            num_workers=4, persistent_workers=True)

    net = torchvision.models.resnet50(weights=None)
    net.fc = torch.nn.Linear(2048, 1)
    net.load_state_dict(torch.load(args.init, map_location="cpu",
                                   weights_only=False)["model"])
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best, history = -1.0, []
    for epoch in range(args.epochs):
        net.train()
        t0, seen, running = time.time(), 0, 0.0
        for x, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            target = (yb.float() * args.pos_smooth).to(device)  # smooth positives only
            loss = loss_fn(net(x.to(device)).squeeze(1), target)
            loss.backward(); opt.step()
            running += float(loss.detach()) * len(yb); seen += len(yb)
        sched.step()
        cal_auc = eval_auroc(net, cal_loader, device)
        history.append({"epoch": epoch, "loss": running / seen,
                        "cal_auroc": cal_auc, "secs": round(time.time() - t0, 1)})
        print(f"[v4] epoch {epoch:02d} loss={running / seen:.4f} "
              f"cal_auroc={cal_auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if cal_auc > best:
            best = cal_auc
            torch.save({"model": net.state_dict(), "epoch": epoch,
                        "cal_auroc": cal_auc, "input_hw": [1152, 896],
                        "init": args.init,
                        "splits_sha256": splits.sha256}, run / "best_model.pt")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch,
                    "best_cal_auroc": best, "history": history,
                    "splits_sha256": splits.sha256}, run / "checkpoint.pt")
        (run / "history.json").write_text(json.dumps(history, indent=1))
    print(f"[v4] done. best cal AUROC={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
