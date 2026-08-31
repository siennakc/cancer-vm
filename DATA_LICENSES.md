# Data sources, licences, and attribution

No image data is redistributed by this repository. `data/raw/` is git-ignored;
`scripts/fetch_*.py` re-download from the original archives, and
`data/metadata/manifest_*.jsonl` records a SHA-256 per series so a training run
or a gate result can be pinned to exact bytes.

Both collections are public on The Cancer Imaging Archive and need **no
credentials**, which is what keeps the data path reproducible by anyone who
clones this repo.

## CBIS-DDSM — site `ddsm`

- **Collection:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
- **Licence:** CC BY 3.0
- **DOI:** <https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY>
- **Page:** <https://www.cancerimagingarchive.net/collection/cbis-ddsm/>
- **Publication:** Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M.,
  & Rubin, D. L. (2017). *A curated mammography data set for use in computer-aided
  detection and diagnosis research.* Scientific Data, 4, 170177.
- **Used here:** the 3,103 `full mammogram images` series. The `ROI mask images`
  and `cropped images` series (~60 GB) are derived products a detection-only v1
  (decision D4) does not consume.
- **Labels:** biopsy-proven `pathology` from the four case-description CSVs.
  `BENIGN_WITHOUT_CALLBACK` is mapped to the **negative** class — it is tissue
  that was not recalled, and folding it into the positives inflates prevalence.

> **Patient-id caveat.** TCIA's `PatientID` for this collection is per-*view*
> (`Mass-Training_P_01239_RIGHT_CC`). The real patient key is the `patient_id`
> column of the CSVs (`P_01239`); see `data/mammography.py` and
> `tests/test_mammography_cases.py`.

## CMMD — site `cmmd`

- **Collection:** CMMD (The Chinese Mammography Database)
- **Licence:** CC BY 4.0
- **DOI:** <https://doi.org/10.7937/tcia.eqde-4b16>
- **Page:** <https://www.cancerimagingarchive.net/collection/cmmd/> — cite as
  directed there in any publication.
- **Used here:** all 1,775 series (~23 GB).
- **Labels:** biopsy-proven `classification` (benign/malignant) from
  `CMMD_clinicaldata_revision.xlsx`, joined per **breast** via each file's
  `ImageLaterality` tag — CMMD labels a breast, not a study.

## Prevalence warning

Neither collection is at screening prevalence. Both are biopsy-enriched: CMMD is
~70% malignant, CBIS-DDSM roughly balanced, against a real screening rate near
0.5%. Any probability from a model trained on these is on the wrong prior, so
the analytic prior-correction logit shift in `models/head.py` (axiom A10) is
mandatory before a number is reported, and PPV must never be quoted from these
test sets (pitfall register: "PPV from an enriched test set").

## Not used, and why

- **RSNA Screening Mammography** (decision D1's default) — requires a Kaggle
  account and competition-rule acceptance, so it cannot be fetched by an
  unattended, credential-free script. It remains the natural third site once
  someone with an account accepts the terms.
- **VinDr-Mammo**, **EMBED** — credentialed / application-gated.

## MIAS MiniMammographic Database — external benchmark only, site `mias`

- **Source:** Mammographic Image Analysis Society, via the Internet Archive's
  copy of the official distribution (`all-mias.tar.gz`, peipa.essex.ac.uk,
  Dec 2012; Apollo/Cambridge mirror was 500ing at fetch time).
- **Licence:** research use only; no redistribution (images are git-ignored;
  only the seal, gold labels, and results are committed). Cite: Suckling et al.,
  "The Mammographic Image Analysis Society digital mammogram database" (1994).
- **Used here:** all 322 images as a sealed external benchmark. Never used for
  training, calibration, thresholds, or model selection — enforced by byte-hash
  and near-duplicate leakage gates at build and at every run.
- **Label caveat:** MIAS severity mixes biopsy-proven and expert consensus.
