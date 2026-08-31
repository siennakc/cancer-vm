"""Score CAMELYON16 slides with the PCam classifier: stream, tile, infer, delete.

Usage: infer_c16_slide.py --list test_001,test_002 [--keep]  (or --list-file)
Writes one JSON per slide to runs/c16/scores/ (max prob, top-k means, n tiles)
so aggregation can be chosen later on TRAIN slides without re-running inference.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/bench")
import torch, torchvision
from camelyon_lib import PATCH, fetch, open_slide, pick_level, read_patches, tissue_tiles

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="")
    ap.add_argument("--list-file", default="")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--weights", default="runs/pcam/best_model.pt")
    args = ap.parse_args()

    names = ([n for n in args.list.split(",") if n] if args.list
             else [l.strip() for l in open(args.list_file) if l.strip()])
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = torchvision.models.resnet50(weights=None)
    net.fc = torch.nn.Linear(2048, 1)
    net.load_state_dict(torch.load(args.weights, map_location="cpu",
                                   weights_only=False)["model"])
    net = net.eval().to(dev)
    out_dir = Path("runs/c16/scores"); out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path("runs/c16/slides"); scratch.mkdir(parents=True, exist_ok=True)

    import subprocess
    def prefetch(nm):
        if nm and not (out_dir / f"{nm}.json").exists() and not (scratch / f"{nm}.tif").exists():
            return subprocess.Popen(["curl", "-sL", "--retry", "5", "-C", "-", "-o",
                                     str(scratch / f"{nm}.tif.part"),
                                     f"https://camelyon-dataset.s3.us-west-2.amazonaws.com/CAMELYON16/images/{nm}.tif"])
        return None

    pf = None
    for idx, name in enumerate(names):
        out = out_dir / f"{name}.json"
        if out.exists():
            print(f"[c16] {name}: cached", flush=True); continue
        t0 = time.time()
        slide_path = scratch / f"{name}.tif"
        part = scratch / f"{name}.tif.part"
        if pf is not None:
            pf.wait()
            pf = None
        if part.exists() and not slide_path.exists():
            part.rename(slide_path)
        fetch(f"images/{name}.tif", slide_path)
        nxt = next((n for n in names[idx + 1:] if not (out_dir / f"{n}.json").exists()), None)
        pf = prefetch(nxt)
        t_dl = time.time() - t0
        slide = open_slide(slide_path)
        level, scale = pick_level(slide)
        coords = tissue_tiles(slide, level)
        probs = []
        batch = []
        with torch.no_grad():
            for patch in read_patches(slide, level, coords, scale):
                batch.append(torch.from_numpy(patch).permute(2, 0, 1))
                if len(batch) == args.batch:
                    x = (torch.stack(batch) - MEAN) / STD
                    probs.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
                    batch = []
            if batch:
                x = (torch.stack(batch) - MEAN) / STD
                probs.append(torch.sigmoid(net(x.to(dev)).squeeze(1)).cpu().numpy())
        slide.close()
        if not args.keep:
            slide_path.unlink(missing_ok=True)
        p = np.concatenate(probs) if probs else np.zeros(1)
        # per-tile evidence for post-hoc calibration / noisy-OR / TTA re-verify:
        # full prob vector + coords of the top-200 tiles (216 KB/slide)
        top_idx = np.argsort(p)[::-1][:200]
        np.savez_compressed(out_dir / f"{name}.npz", probs=p.astype(np.float16),
                            top_idx=top_idx.astype(np.int32),
                            top_coords=np.array([coords[j] for j in top_idx], np.int32),
                            level=level, scale=scale)
        p_sorted = np.sort(p)[::-1]
        rec = {"slide": name, "n_tiles": int(len(p)), "level": level,
               "max": float(p_sorted[0]),
               **{f"top{k}_mean": float(p_sorted[:k].mean()) for k in (5, 20, 50, 100)},
               "frac_over_0.5": float((p > 0.5).mean()),
               "frac_over_0.9": float((p > 0.9).mean()),
               "dl_s": round(t_dl, 1), "total_s": round(time.time() - t0, 1)}
        out.write_text(json.dumps(rec))
        print(f"[c16] {name}: {len(p)} tiles max={rec['max']:.3f} "
              f"top20={rec['top20_mean']:.3f} ({rec['total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
