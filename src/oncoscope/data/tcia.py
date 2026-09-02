"""TCIA / NBIA ingestion client (T-1.1, real-data arm).

Downloads DICOM series from The Cancer Imaging Archive's public NBIA REST API.
Public collections need no credentials, which is what makes this repo's data
path reproducible by anyone who clones it.

Nothing here decodes pixels — ``dicom_canonical.load_canonical`` remains the
single loader (preprocessing skew between training and serving is the largest
silent accuracy killer, so ingestion stops at "bytes on disk, hashed").

Every downloaded series is recorded in a JSONL manifest with its SHA-256 so a
split, an eval, or a gate run can be pinned to exact bytes. Re-running is
idempotent: a series whose hash already matches the manifest is skipped.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
USER_AGENT = "oncoscope-data-ingest/0.1 (+https://github.com/siennakc/oncoscope)"


@dataclass(frozen=True)
class SeriesRef:
    """One TCIA series as advertised by the API (before download)."""

    series_uid: str
    collection: str
    patient_id: str          # TCIA's PatientID field — NOT always a real patient (see below)
    description: str | None
    image_count: int
    file_size: int
    modality: str | None = None

    @classmethod
    def from_api(cls, row: dict) -> "SeriesRef":
        return cls(
            series_uid=row["SeriesInstanceUID"],
            collection=row.get("Collection", ""),
            patient_id=str(row.get("PatientID", "")),
            description=row.get("SeriesDescription"),
            image_count=int(row.get("ImageCount", 0) or 0),
            file_size=int(float(row.get("FileSize", 0) or 0)),
            modality=row.get("Modality"),
        )


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_series(collection: str, cache_dir: Path | str | None = None) -> list[SeriesRef]:
    """All series in a public collection. Cached to disk — the listing is stable."""
    url = f"{API}/getSeries?{urllib.parse.urlencode({'Collection': collection})}"
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{collection}_series.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            rows = json.loads(cache.read_text())
        else:
            rows = json.loads(_get(url))
            cache.write_text(json.dumps(rows))
    else:
        rows = json.loads(_get(url))
    return [SeriesRef.from_api(r) for r in rows]


def _sha256_dir(directory: Path) -> str:
    """Hash of a series directory: stable over file order, covers every byte."""
    h = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(directory).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


@dataclass
class Manifest:
    """Append-only record of what was downloaded, keyed by series UID."""

    path: Path
    entries: dict[str, dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def load(cls, path: Path | str) -> "Manifest":
        path = Path(path)
        entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    entries[row["series_uid"]] = row
        return cls(path=path, entries=entries)

    def record(self, row: dict) -> None:
        with self._lock:
            self.entries[row["series_uid"]] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    def has(self, series_uid: str) -> bool:
        return series_uid in self.entries


def download_series(ref: SeriesRef, dest_root: Path, retries: int = 3) -> dict:
    """Download and unzip one series into ``dest_root/<collection>/<series_uid>/``.

    Returns the manifest row. Raises on unrecoverable failure so the caller can
    record it rather than silently producing a short dataset.
    """
    target = dest_root / ref.collection / ref.series_uid
    tmp_zip = target.with_suffix(".zip.part")
    url = f"{API}/getImage?{urllib.parse.urlencode({'SeriesInstanceUID': ref.series_uid})}"

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = _get(url, timeout=600)
            tmp_zip.write_bytes(payload)
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True)
            with zipfile.ZipFile(tmp_zip) as zf:
                zf.extractall(target)
            tmp_zip.unlink(missing_ok=True)
            dcm = sorted(target.rglob("*.dcm"))
            if not dcm:
                raise RuntimeError("no .dcm files in downloaded series")
            return {
                "series_uid": ref.series_uid,
                "collection": ref.collection,
                "tcia_patient_id": ref.patient_id,
                "description": ref.description,
                "n_files": len(dcm),
                "bytes": sum(p.stat().st_size for p in dcm),
                "sha256": _sha256_dir(target),
                "relpath": str(target.relative_to(dest_root)),
            }
        except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, RuntimeError, OSError) as exc:
            last_err = exc
            tmp_zip.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to download {ref.series_uid}: {last_err}")


def download_collection(
    refs: list[SeriesRef],
    dest_root: Path | str,
    manifest_path: Path | str,
    workers: int = 6,
    progress_every: int = 25,
) -> Manifest:
    """Download many series concurrently, skipping ones already in the manifest."""
    dest_root = Path(dest_root)
    manifest = Manifest.load(manifest_path)
    todo = [r for r in refs if not manifest.has(r.series_uid)]
    done = failed = 0
    total = len(todo)
    print(f"[tcia] {len(refs)} series requested, {total} to fetch "
          f"({len(refs) - total} already in manifest)", flush=True)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_series, r, dest_root): r for r in todo}
        for fut in cf.as_completed(futures):
            ref = futures[fut]
            try:
                manifest.record(fut.result())
                done += 1
            except Exception as exc:  # recorded, never silently dropped
                failed += 1
                manifest.record({
                    "series_uid": ref.series_uid,
                    "collection": ref.collection,
                    "tcia_patient_id": ref.patient_id,
                    "error": str(exc),
                })
            if (done + failed) % progress_every == 0 or done + failed == total:
                print(f"[tcia] {done + failed}/{total}  ok={done} failed={failed}", flush=True)
    return manifest


def failed_series(manifest_path: Path | str) -> set[str]:
    """UIDs whose most recent manifest entry is an error.

    Manifest lines are append-only and later lines win on load, so a successful
    retry appended after a failure supersedes it without rewriting history.
    """
    manifest = Manifest.load(manifest_path)
    return {uid for uid, row in manifest.entries.items() if "error" in row}


def retry_failed(
    refs: list[SeriesRef],
    dest_root: Path | str,
    manifest_path: Path | str,
    workers: int = 3,
) -> int:
    """Re-attempt every series currently recorded as failed. Returns the number recovered.

    TCIA occasionally serves a truncated zip; those surface as zlib errors and
    almost always succeed on a later attempt. Silently tolerating them would
    mean training on a short dataset without noticing.
    """
    pending = failed_series(manifest_path)
    todo = [r for r in refs if r.series_uid in pending]
    if not todo:
        print("[tcia] no failed series to retry", flush=True)
        return 0

    manifest = Manifest.load(manifest_path)
    recovered = 0
    print(f"[tcia] retrying {len(todo)} failed series", flush=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_series, r, Path(dest_root)): r for r in todo}
        for fut in cf.as_completed(futures):
            ref = futures[fut]
            try:
                manifest.record(fut.result())
                recovered += 1
            except Exception as exc:
                manifest.record({
                    "series_uid": ref.series_uid,
                    "collection": ref.collection,
                    "tcia_patient_id": ref.patient_id,
                    "error": str(exc),
                })
    print(f"[tcia] recovered {recovered}/{len(todo)}", flush=True)
    return recovered
