"""Fetch the CBIS-DDSM ROI-mask + cropped-image series (~50-60 GB, public).

The patch-pretraining stage needs the lesion annotations that fetch_cbis.py
deliberately skipped under decision D4. Same machinery, separate manifest:
full-image bytes stay pinned by manifest_cbis.jsonl, annotations by
manifest_cbis_roi.jsonl.

Series are selected as everything EXCEPT "full mammogram images" and
classified later by pixel content (see oncoscope.data.roi) — CBIS's own
series descriptions and file ordering are not reliable enough to trust here.

Bounded retry sweeps are built in: TCIA serves truncated zips routinely
(156 across the first two collections), and a drain loop is cheaper than
rediscovering that.
"""
import sys; sys.path.insert(0, "src")
from oncoscope.data.tcia import download_collection, failed_series, list_series, retry_failed

MANIFEST = "data/metadata/manifest_cbis_roi.jsonl"

refs = [r for r in list_series("CBIS-DDSM", cache_dir="data/metadata")
        if r.description != "full mammogram images"]
by_desc: dict[str, int] = {}
for r in refs:
    by_desc[r.description or "?"] = by_desc.get(r.description or "?", 0) + 1
print(f"CBIS-DDSM ROI/crop: {len(refs)} series, "
      f"{sum(r.file_size for r in refs)/1e9:.1f} GB, by description: {by_desc}")

download_collection(refs, "data/raw", MANIFEST, workers=4)
for sweep in range(4):
    if not failed_series(MANIFEST):
        break
    print(f"[fetch-roi] retry sweep {sweep + 1}")
    retry_failed(refs, "data/raw", MANIFEST)

remaining = failed_series(MANIFEST)
if remaining:
    print(f"[fetch-roi] WARNING: {len(remaining)} series still failed — "
          "re-run this script before building the patch dataset")
    sys.exit(1)
print("[fetch-roi] complete, zero failed series")
