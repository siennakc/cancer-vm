"""Fetch the CMMD mammography collection from TCIA (public, no credentials)."""
import sys; sys.path.insert(0, "src")
from oncoscope.data.tcia import list_series, download_collection

refs = list_series("CMMD", cache_dir="data/metadata")
print(f"CMMD: {len(refs)} series, {sum(r.file_size for r in refs)/1e9:.1f} GB")
download_collection(refs, "data/raw", "data/metadata/manifest_cmmd.jsonl", workers=6)
