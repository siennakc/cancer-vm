"""MIAS v1.21 -> benchmark case list (external site ``mias``).

The Mammographic Image Analysis Society database: UK film-screen mammograms,
digitized — a third country, third era, third acquisition chain, and a source
neither training site ever touched. Small (322 images) but authentic; CIs are
reported wide, not hidden.

Label policy (Info.txt fields: reference class severity ...):
- severity M                      -> label 1
- severity B, or class NORM       -> label 0
MIAS truth mixes biopsy and expert reading; that caveat ships in the bench card.

MIAS ids are ``mdb001``..``mdb322``; consecutive pairs are the same woman's
left/right films (mdb001+mdb002 = patient 1). Grouping uses that pairing —
same A9 discipline as everywhere else, even in an eval-only set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    patient_id: str
    image_path: str    # relative to the raw root
    label: int
    mias_class: str
    severity: str | None


def read_pgm_canonical(path: Path) -> np.ndarray:
    """PGM (P2/P5) -> float32 [0,1], same contract as load_canonical.pixels."""
    data = Path(path).read_bytes()
    m = re.match(rb"(P[25])\s+(?:#.*\s+)*(\d+)\s+(\d+)\s+(\d+)\s", data)
    if not m:
        raise ValueError(f"not a PGM: {path}")
    magic, w, h, maxval = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    offset = m.end()
    if magic == b"P5":
        dtype = np.uint16 if maxval > 255 else np.uint8
        arr = np.frombuffer(data, dtype=dtype, count=w * h, offset=offset)
    else:
        arr = np.array(data[offset:].split()[: w * h], dtype=np.float64)
    arr = arr.reshape(h, w).astype(np.float64)
    return (arr / maxval).astype(np.float32)


def parse_info(info_text: str) -> list[tuple[str, str, str | None]]:
    """Info listing -> [(mias_id, class, severity|None)]. One row per abnormality;
    an image with any M abnormality is malignant."""
    rows: list[tuple[str, str, str | None]] = []
    for line in info_text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"mdb\d{3}", parts[0]):
            severity = parts[3] if len(parts) > 3 and parts[3] in ("B", "M") else None
            rows.append((parts[0], parts[2], severity))
    return rows


def build_mias_cases(raw_root: Path | str) -> list[BenchCase]:
    raw_root = Path(raw_root)
    pgms = {p.stem: p for p in raw_root.rglob("mdb*.pgm")}
    info_files = sorted(raw_root.rglob("*.txt"))
    info_rows: list[tuple[str, str, str | None]] = []
    for f in info_files:
        try:
            rows = parse_info(f.read_text(errors="ignore"))
        except OSError:
            continue
        if len(rows) > len(info_rows):
            info_rows = rows

    by_image: dict[str, dict] = {}
    for mid, klass, severity in info_rows:
        rec = by_image.setdefault(mid, {"classes": [], "severities": []})
        rec["classes"].append(klass)
        if severity:
            rec["severities"].append(severity)

    cases: list[BenchCase] = []
    for mid, path in sorted(pgms.items()):
        rec = by_image.get(mid)
        if rec is None:
            continue  # no truth row -> excluded, never guessed
        if "M" in rec["severities"]:
            label = 1
        elif "B" in rec["severities"] or "NORM" in rec["classes"]:
            label = 0
        else:
            continue
        num = int(mid[3:])
        cases.append(BenchCase(
            case_id=f"mias-{mid}",
            patient_id=f"mias-p{(num + 1) // 2:03d}",   # films come in L/R pairs
            image_path=str(path.relative_to(raw_root)),
            label=label,
            mias_class="+".join(sorted(set(rec["classes"]))),
            severity="M" if "M" in rec["severities"] else
                     ("B" if "B" in rec["severities"] else None),
        ))
    return cases
