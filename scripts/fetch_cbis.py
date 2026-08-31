"""Fetch CBIS-DDSM full mammogram images from TCIA (public, no credentials).

Only the 'full mammogram images' series: the ROI-mask and cropped-image series
are derived products (60 GB) that a detection-only v1 (decision D4) does not use.
"""
import sys; sys.path.insert(0, "src")
from oncoscope.data.tcia import list_series, download_collection

refs = [r for r in list_series("CBIS-DDSM", cache_dir="data/metadata")
        if r.description == "full mammogram images"]
print(f"CBIS-DDSM: {len(refs)} full-mammogram series, {sum(r.file_size for r in refs)/1e9:.1f} GB")
download_collection(refs, "data/raw", "data/metadata/manifest_cbis.jsonl", workers=4)
