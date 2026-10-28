# Datasets, preparation and attribution

Download the raw datasets from their original records. They are not bundled
with this documentation or model code. Work from the repository root and
preserve each release's original source files.

## BUS-BRA

Official release: [Zenodo](https://zenodo.org/records/8231412),
[DOI 10.5281/zenodo.8231412](https://doi.org/10.5281/zenodo.8231412).
Original tools: [wgomezf/BUS-BRA](https://github.com/wgomezf/BUS-BRA).

The baseline expects the following paths relative to the repository root:

| Path | Contents |
| --- | --- |
| `data/raw/BUSBRA/Images/` | Original image PNGs |
| `data/raw/BUSBRA/Masks/` | Corresponding binary mask PNGs |
| `data/raw/BUSBRA/bus_data.csv` | Image/Case/pathology/device metadata |
| `data/raw/BUSBRA/5-fold-cv.csv` | Supplied five-fold metadata, audited |
| `data/raw/BUSBRA/10-fold-cv.csv` | Supplied outer assignments used for the fixed split |

```bash
python scripts/prepare_splits.py
```

This writes `data/processed/busbra_manifest.csv` and
`data/processed/metadata_audit.json`. It requires related images to remain
within their supplied `Case` group and checks real image/mask pairs. The
completed full audit found 1,875 image pairs and 1,064 Cases, with no Case
overlap between train, validation and test.

The four source CSV/PNG discrepancies are retained as audit records:

| Image ID | CSV width × height | Matching image/mask PNG width × height |
| --- | --- | --- |
| `bus_0856-l` | 387 × 405 | 387 × 353 |
| `bus_0857-r` | 384 × 409 | 384 × 393 |
| `bus_0912-l` | 360 × 354 | 360 × 352 |
| `bus_0990-l` | 540 × 354 | 540 × 352 |

Only verified matching PNG dimensions populate the manifest; the CSV is
unchanged. Unexpected image/mask mismatch, invalid labels or corrupted images
remain errors.

The source release requires citation of Gómez-Flores, Gregorio-Calas and
de Albuquerque Pereira (2024), *BUS-BRA: A Breast Ultrasound Dataset for
Assessing Computer-aided Diagnosis Systems*, Medical Physics, 51, 3110–3123,
[DOI 10.1002/mp.16812](https://doi.org/10.1002/mp.16812).
Consult the dataset record for applicable data terms. The GPL-3.0 licence of
the authors' source-code repository is not, by itself, a dataset licence.

## BUS-UCLM V3

Official release: [Mendeley Data, V3](https://data.mendeley.com/datasets/7fvgj4jsp7/3),
published 27 January 2025,
[DOI 10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3).

If the supplied dataset ZIP is in the repository root, extract it with:

```bash
python -m zipfile -e \
  "BUS-UCLM Breast ultrasound lesion segmentation dataset.zip" \
  data/raw/BUSUCLM
```

Run extraction once into a clean destination, preserving the release layout.
The auditor finds the single `INFO.csv` recursively under `data/raw/BUSUCLM`.
For the reviewed release, its containing dataset directory ends in
`BUS-UCLM Breast ultrasound lesion segmentation dataset/BUS-UCLM`.
Do not flatten the directories.

| Relative to the discovered dataset directory | Contents |
| --- | --- |
| `INFO.csv` | Semicolon-delimited, UTF-8 BOM metadata |
| `images/` | Original image PNGs |
| `masks/` | Original RGB reference PNGs |

CSV fields are `Image`, `Resolution`, `Label`, `Doppler`, `Marks`, `Combined`.
The four-letter filename prefix identifies the supplied patient group.
The original release has 683 images from 38 patient prefixes: 174 benign,
90 malignant and 419 normal. References use green for benign lesions, red
for malignant lesions and black for background/normal reference images.
The evaluator uses the green/red union as foreground and preserves raw flags;
`POFR_013` has unresolved `Doppler=Y`, which is not imputed.

```bash
python scripts/audit_busuclm_external.py
```

The protocol-v2 audit requires all 683 original pairs, including the four
quarantined ones. Its outputs go to `data/processed/busuclm_external_v2/`.
The full audit passed with 679 eligible pairs and no unexpected errors.

Four positive references, `HESN_002`–`HESN_005`, were excluded before external
inference because image/mask alignment could not be validated. Nine reviewed
all-black HESN normal references are represented as empty native-grid arrays
in memory. No positive reference is aligned by guessing, and no raw file is
rewritten. Read the [geometry amendment](BUSUCLM_GEOMETRY_AMENDMENT.md), which
records all 13 reviewed dimensions and the exact byte-identity rule.

## Attribution for the derived prediction gallery

The gallery at `figures/busuclm_external_predictions_review.png` contains
dataset-derived views and must retain source credit and its change notice:

> Source: Vallez, Noelia; Bueno, Gloria; Deniz, Oscar; Rienda, Miguel Angel;
> Pastor, Carlos (2025), BUS-UCLM: Breast ultrasound lesion segmentation
> dataset, Mendeley Data, V3, DOI 10.17632/7fvgj4jsp7.3. CC BY 4.0.
> Grayscale views, reference contours and model prediction overlays added.
> No endorsement by the dataset authors is implied.

Dataset and licence links:

- [BUS-UCLM V3 source](https://data.mendeley.com/datasets/7fvgj4jsp7/3)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- [Dataset paper: Scientific Data 12, 242 (2025)](https://doi.org/10.1038/s41597-025-04562-3)

This notice attributes the derived views; it does not assign a new licence
to project source code or replace third-party terms.
