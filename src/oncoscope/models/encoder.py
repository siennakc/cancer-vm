"""Frozen foundation-model encoder for cached embeddings (T-2.1, axiom A11).

The v0 encoder is a deliberately boring baseline: ImageNet-pretrained
ResNet-50, frozen, global-average-pooled. It is not a mammography FM — it is
the control arm every fancier encoder must beat through the gate. Swapping in
a real pathology/radiology FM later means implementing ``FrozenEncoder`` with
a new ``tag`` and re-running the cache; nothing downstream changes.

Preprocessing here is the *screener's* view (whole breast, letterboxed).
Axiom A1 (resolution before reasoning) is the harness's job via crop tools —
the cheap screener is allowed to see small, the zoom loop is not.

Determinism: eval mode, no grad, fixed resize; embeddings are L2-normalized so
the logistic head sees unit-scale features regardless of encoder family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def breast_crop(pixels: np.ndarray, threshold: float = 0.02, pad: int = 16) -> np.ndarray:
    """Crop to the breast's bounding box, dropping the empty background field.

    Mammograms are mostly air; resizing the full field wastes most of the
    encoder's resolution on black. Threshold + bounding box is crude but
    deterministic, and failures degrade to the uncropped image.
    """
    mask = pixels > threshold
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return pixels
    r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(pixels.shape[0], r1 + pad), min(pixels.shape[1], c1 + pad)
    crop = pixels[r0:r1, c0:c1]
    return crop if crop.size else pixels


def letterbox(pixels: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Aspect-preserving resize onto an HxW canvas (area interpolation).

    ``size`` may be one int (square) or (H, W) — mammograms are portrait, and
    the literature recipe (Shen et al. 2019) uses 1152x896.
    """
    import torch
    import torch.nn.functional as F

    th, tw = (size, size) if isinstance(size, int) else size
    h, w = pixels.shape
    scale = min(th / h, tw / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    t = torch.from_numpy(np.ascontiguousarray(pixels))[None, None]
    resized = F.interpolate(t, size=(nh, nw), mode="area")[0, 0].numpy()
    canvas = np.zeros((th, tw), dtype=np.float32)
    canvas[(th - nh) // 2 : (th - nh) // 2 + nh,
           (tw - nw) // 2 : (tw - nw) // 2 + nw] = resized
    return canvas


@dataclass
class FrozenEncoder:
    """ResNet-50 feature extractor. ``tag`` versions the embedding cache.

    ``weights_path`` loads a fine-tuned checkpoint (``best_model.pt`` from
    ``scripts/finetune_encoder.py``); the classifier head in the checkpoint is
    training scaffolding and is dropped here — downstream always consumes
    embeddings + the calibrated LogisticHead stack.
    """

    tag: str = "resnet50_in1k_v2_448"
    input_size: int | tuple[int, int] = 448
    embed_dim: int = 2048
    weights_path: str | None = None
    normalize: bool = True  # L2-norm suits generic frozen features; a fine-tuned
                            # backbone's fc consumed raw GAP magnitudes — keep them
    # Normalization stats MUST match how the weights were trained (preprocessing
    # skew pitfall): ImageNet per-channel for stock weights; the grayscale pair
    # (0.449, 0.226) for checkpoints from scripts/finetune_encoder.py.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        import torch
        import torchvision

        self._torch = torch
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        # With a local checkpoint, never touch the torchvision hub — the
        # hermetic benchmark denies network, and offline loads must stay offline.
        net = torchvision.models.resnet50(
            weights=None if self.weights_path is not None
            else torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        if self.weights_path is not None:
            state = torch.load(self.weights_path, map_location="cpu",
                               weights_only=False)["model"]
            state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            missing, unexpected = net.load_state_dict(state, strict=False)
            assert not unexpected, f"unexpected keys: {unexpected[:3]}"
            assert all(k.startswith("fc.") for k in missing), f"missing: {missing[:3]}"
        net.fc = torch.nn.Identity()
        self.net = net.eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def preprocess(self, pixels: np.ndarray) -> np.ndarray:
        """Canonical [0,1] grayscale -> (3, S, S) ImageNet-normalized float32."""
        img = letterbox(breast_crop(pixels.astype(np.float32)), self.input_size)
        rgb = np.repeat(img[None], 3, axis=0)
        mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.std, dtype=np.float32)[:, None, None]
        return (rgb - mean) / std

    def embed_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        """List of canonical images -> (n, embed_dim) L2-normalized embeddings."""
        torch = self._torch
        x = torch.from_numpy(np.stack([self.preprocess(p) for p in batch]))
        with torch.no_grad():
            feats = self.net(x.to(self.device)).float().cpu().numpy()
        if not self.normalize:
            return feats
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.clip(norms, 1e-9, None)
