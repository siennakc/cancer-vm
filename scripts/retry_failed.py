"""Re-attempt series that TCIA served truncated. Idempotent; safe to re-run."""
import sys; sys.path.insert(0, "src")
from oncoscope.data.tcia import list_series, retry_failed

collection, manifest, desc = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
refs = list_series(collection, cache_dir="data/metadata")
if desc:
    refs = [r for r in refs if r.description == desc]
retry_failed(refs, "data/raw", manifest, workers=3)
