"""Sliding-window lesion detector on the trained patch model.

The localizing detector the harness architecture presumes (axiom A2:
detector proposes, adjudication filters) and whose absence the harness A/B
diagnosed: model-alone 0.771 vs harness 0.700 with 41% uninformative
deferrals, because the "detector" was a whole-image model reading windows it
was never trained on.

The detector runs the 5-class patch classifier over a strided grid of the
SAME render the whole-image model sees, scores each window
1 - P(background), and proposes peak boxes (greedy NMS). Boxes are reported
in SOURCE pixel coordinates via the shared RenderGeometry, so every
downstream measurement (mm sizes, crop tools, the evidence ledger) speaks
the original image's frame — the LLM never receives render-space
coordinates it could confuse with source-space ones.

Torch is imported lazily; importing this module costs nothing in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data.patches import RenderGeometry, render_geometry

DEFAULT_RENDER = (1152, 896)


def content_windows(
    corners: list[tuple[int, int]],
    geometry: RenderGeometry,
    patch: int,
) -> list[tuple[int, int]]:
    """Keep only grid windows that intersect the letterboxed CONTENT region.

    Pure-padding windows are out-of-distribution for the patch model (training
    never produced a near-constant black window) and their boxes map entirely
    outside the source crop, where naive clamping yields inverted coordinates.
    Filtering them is both a correctness and a compute win.
    """
    r0, r1, c0, c1 = geometry.box
    oy, ox = geometry.offset
    nh = max(1, round((r1 - r0) * geometry.scale))
    nw = max(1, round((c1 - c0) * geometry.scale))
    return [
        (y, x) for y, x in corners
        if y < oy + nh and y + patch > oy and x < ox + nw and x + patch > ox
    ]


def symmetric_nms(
    corners: list[tuple[int, int]],
    scores: np.ndarray,
    patch: int,
    score_threshold: float,
    max_candidates: int,
) -> list[int]:
    """Greedy NMS suppressing any equal-size window that OVERLAPS a pick.

    Two patch-size windows overlap iff both center distances are < patch, so
    the criterion is symmetric by construction. (The previous center-inside
    test used a half-open interval: at stride = patch/2 the up/left neighbour
    was suppressed while the down/right one survived, emitting duplicate boxes
    trailing down-right from every peak.)
    """
    order = np.argsort(scores)[::-1]
    picked: list[int] = []
    for i in order:
        if scores[i] < score_threshold or len(picked) >= max_candidates:
            break
        cy, cx = corners[i][0] + patch / 2, corners[i][1] + patch / 2
        if any(
            abs(cy - (corners[j][0] + patch / 2)) < patch
            and abs(cx - (corners[j][1] + patch / 2)) < patch
            for j in picked
        ):
            continue
        picked.append(int(i))
    return picked


def window_to_source_box(
    geometry: RenderGeometry,
    y: int,
    x: int,
    patch: int,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Render window -> source-space box with INCLUSIVE endpoints, or None.

    Corners are ordered before clamping (a window straddling the content edge
    maps partially outside the crop), rounded OUTWARD (floor the near corner,
    ceil the far one — a detector box may grow a pixel, never shrink), then
    clamped; a window with no valid source extent returns None instead of an
    inverted box.
    """
    r0a, c0a = geometry.render_to_source(y, x)
    r1a, c1a = geometry.render_to_source(y + patch, x + patch)
    rlo, rhi = sorted((r0a, r1a))
    clo, chi = sorted((c0a, c1a))
    h, w = image_shape
    y0 = max(0, int(np.floor(rlo)))
    x0 = max(0, int(np.floor(clo)))
    y1 = min(h - 1, int(np.ceil(rhi)))
    x1 = min(w - 1, int(np.ceil(chi)))
    if y1 <= y0 or x1 <= x0:
        return None
    return (x0, y0, x1, y1)


@dataclass(frozen=True)
class PatchCandidate:
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 in SOURCE pixels,
                                     # endpoints INCLUSIVE (crop with [y0:y1+1, x0:x1+1])
    score: float                     # 1 - P(background) at the peak window
    cls: int                         # argmax lesion class (1..4)
    cls_probs: tuple[float, ...]     # full softmax, background included


@dataclass
class PatchDetector:
    weights_path: str | Path
    render_size: tuple[int, int] = DEFAULT_RENDER
    patch: int = 224
    stride: int = 112
    score_threshold: float = 0.25
    max_candidates: int = 16
    min_breast_frac: float = 0.5   # mirrors the sampler's background gate
    batch: int = 64
    mean: float = 0.449
    std: float = 0.226
    _net: object = field(default=None, repr=False)
    _device: object = field(default=None, repr=False)

    def _ensure_net(self):
        if self._net is not None:
            return
        import torch
        import torchvision

        from ..data.roi import PATCH_CLASSES

        ck = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        if ck.get("tainted"):
            raise RuntimeError(
                f"{self.weights_path} is TAINTED (trained on smoke-test shards) "
                "— refusing to serve detections from it"
            )
        n_classes = len(ck.get("classes", PATCH_CLASSES))
        net = torchvision.models.resnet50(weights=None)
        net.fc = torch.nn.Linear(2048, n_classes)
        net.load_state_dict(ck["model"])
        self._device = (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cpu"))
        self._net = net.eval().to(self._device)
        for p in self._net.parameters():
            p.requires_grad_(False)

    # -- scoring ---------------------------------------------------------
    def _grid(self, canvas: tuple[int, int]) -> list[tuple[int, int]]:
        ys = list(range(0, canvas[0] - self.patch + 1, self.stride))
        xs = list(range(0, canvas[1] - self.patch + 1, self.stride))
        if ys and ys[-1] != canvas[0] - self.patch:
            ys.append(canvas[0] - self.patch)
        if xs and xs[-1] != canvas[1] - self.patch:
            xs.append(canvas[1] - self.patch)
        return [(y, x) for y in ys for x in xs]

    def score_render(
        self,
        render: np.ndarray,
        corners: list[tuple[int, int]] | None = None,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Softmax per grid window over the rendered canvas."""
        import torch

        self._ensure_net()
        if corners is None:
            corners = self._grid(render.shape)
        if not corners:
            return np.zeros((0, 5)), []
        probs = []
        for i in range(0, len(corners), self.batch):
            chunk = corners[i:i + self.batch]
            windows = np.stack([
                render[y:y + self.patch, x:x + self.patch] for y, x in chunk
            ]).astype(np.float32)
            t = torch.from_numpy((windows - self.mean) / self.std)[:, None]
            t = t.repeat(1, 3, 1, 1).to(self._device)
            with torch.no_grad():
                probs.append(torch.softmax(self._net(t), dim=1).float().cpu().numpy())
        return np.concatenate(probs), corners

    def propose(self, pixels: np.ndarray) -> list[PatchCandidate]:
        """Canonical [0,1] full-resolution image -> source-space candidates."""
        from .encoder import breast_crop, letterbox

        geometry = render_geometry(pixels.astype(np.float32), self.render_size)
        render = letterbox(breast_crop(pixels.astype(np.float32)), self.render_size)

        # Score only windows that intersect the content region AND contain a
        # meaningful fraction of breast: training produced no padding/air
        # windows (the sampler's min_breast_frac), so scoring them here would
        # be pure out-of-distribution noise.
        corners = content_windows(self._grid(render.shape), geometry, self.patch)
        breast = render > 0.02
        corners = [
            (y, x) for y, x in corners
            if breast[y:y + self.patch, x:x + self.patch].mean() >= self.min_breast_frac
        ]
        probs, corners = self.score_render(render, corners)
        if not corners:
            return []
        lesion_scores = 1.0 - probs[:, 0]

        picked = symmetric_nms(corners, lesion_scores, self.patch,
                               self.score_threshold, self.max_candidates)
        out = []
        for i in picked:
            y, x = corners[i]
            box = window_to_source_box(geometry, y, x, self.patch, pixels.shape)
            if box is None:
                continue
            out.append(PatchCandidate(
                box=box, score=round(float(lesion_scores[i]), 4),
                cls=int(np.argmax(probs[i, 1:]) + 1),
                cls_probs=tuple(round(float(p), 4) for p in probs[i]),
            ))
        return out
