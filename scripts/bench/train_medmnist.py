"""MedMNIST v2 cancer subset, official protocol: 224px, official splits, test AUC+ACC.

Standardized suite with a public leaderboard (the user-cited 'nearest analogue in
spirit'). We take the cancer-relevant members with the same ResNet-50 backbone
as the rest of the fleet. Official test evaluated exactly once per dataset.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
import medmnist, torch, torchvision
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset

SETS = ["pathmnist", "dermamnist", "breastmnist", "bloodmnist"]
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Wrap(Dataset):
    def __init__(self, ds, train):
        self.ds, self.train = ds, train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, y = self.ds[i]
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)
        x = x.permute(2, 0, 1) if x.ndim == 3 else x[None].repeat(3, 1, 1)
        if self.train:
            if torch.rand(()) < 0.5: x = torch.flip(x, (2,))
            k = int(torch.randint(0, 4, ()))
            if k: x = torch.rot90(x, k, (1, 2))
        return (x - MEAN) / STD, int(np.asarray(y).ravel()[0])


def multiclass_auc(y, probs):
    aucs = []
    for c in range(probs.shape[1]):
        yb = (y == c).astype(float)
        if yb.sum() in (0, len(yb)): continue
        o = np.argsort(probs[:, c]); r = np.empty(len(yb)); r[o] = np.arange(1, len(yb) + 1)
        pos = yb == 1
        aucs.append((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))
    return float(np.mean(aucs))


def run(name, epochs, dev):
    info = INFO[name]
    cls = getattr(medmnist, info["python_class"])
    tr = Wrap(cls(split="train", download=True, size=224), True)
    va = Wrap(cls(split="val", download=True, size=224), False)
    te = Wrap(cls(split="test", download=True, size=224), False)
    n_cls = len(info["label"])
    print(f"[mm] {name}: train={len(tr)} test={len(te)} classes={n_cls}", flush=True)
    tl = DataLoader(tr, batch_size=96, shuffle=True, num_workers=6, drop_last=True,
                    persistent_workers=True)
    vl = DataLoader(va, batch_size=128, num_workers=3)
    el = DataLoader(te, batch_size=128, num_workers=3)
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Linear(2048, n_cls); net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = torch.nn.CrossEntropyLoss()
    best_acc, best_state = -1, None
    for ep in range(epochs):
        net.train(); t0 = time.time()
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            lf(net(x.to(dev)), y.to(dev)).backward(); opt.step()
        sched.step()
        net.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in vl:
                correct += int((net(x.to(dev)).argmax(1).cpu() == y).sum()); n += len(y)
        acc = correct / n
        print(f"[mm] {name} ep{ep} val_acc={acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in el:
            ps.append(torch.softmax(net(x.to(dev)), 1).cpu().numpy()); ys.append(y.numpy())
    y, probs = np.concatenate(ys), np.concatenate(ps)
    acc = float((probs.argmax(1) == y).mean())
    auc = multiclass_auc(y, probs)
    print(f"[mm] {name} TEST acc={acc:.4f} auc={auc:.4f}", flush=True)
    return {"dataset": name, "n_test": int(len(y)), "test_acc": round(acc, 4),
            "test_auc": round(auc, 4), "epochs": epochs, "input": 224,
            "arch": "resnet50_in1k_v2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = []
    for name in SETS:
        out.append(run(name, args.epochs, dev))
        Path("results/bench_medmnist").mkdir(parents=True, exist_ok=True)
        Path("results/bench_medmnist/report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
