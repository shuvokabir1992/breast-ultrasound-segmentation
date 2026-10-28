# BUS-UCLM reference geometry amendment

Protocol version 2, 28 October 2026. This decision precedes external inference
with the six frozen BUS-BRA research checkpoints. It is based on source image
and reference-mask validity, without using external model predictions or scores.

## Observed source files

The full audit on aries passed 670 matching pairs and reported thirteen
dimension mismatches, HESN_000 through HESN_012. All thirteen diagnostic pairs
were then supplied and decoded locally. Their images are opaque RGBA PNGs
with varying dimensions; every reference mask is RGB, 856 by 606 pixels.
Nine references are entirely black and four contain benign green foreground.

| ID | Image W x H | Original mask W x H | Original label | Version 2 handling |
| --- | --- | --- | --- | --- |
| HESN_000 | 620 x 489 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_001 | 789 x 489 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_002 | 790 x 468 | 856 x 606 | Benign | Excluded: unresolved positive-reference geometry |
| HESN_003 | 792 x 485 | 856 x 606 | Benign | Excluded: unresolved positive-reference geometry |
| HESN_004 | 792 x 483 | 856 x 606 | Benign | Excluded: unresolved positive-reference geometry |
| HESN_005 | 787 x 486 | 856 x 606 | Benign | Excluded: unresolved positive-reference geometry |
| HESN_006 | 612 x 487 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_007 | 620 x 489 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_008 | 616 x 489 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_009 | 621 x 487 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_010 | 608 x 486 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_011 | 637 x 486 | 856 x 606 | Normal | Empty native-grid reference |
| HESN_012 | 616 x 488 | 856 x 606 | Normal | Empty native-grid reference |

The image PNG chunks contain IHDR, sRGB, gAMA, pHYs, tIME, IDAT and IEND.
No offset, EXIF or text/XMP transform metadata is present. The decoded image
and mask grids and visual inspection do not establish a validated crop,
translation or scaling map for the four nonempty references. Their masks
must not be stretched or shifted to obtain a plausible-looking fit. Recovering
those annotations would require verified original images or coordinate mappings.
The export/cropping history itself is not established by these files.

## Prespecified handling

For each of the nine verified normal cases, the reference foreground set is
empty. Its representation on the actual image grid is therefore an all-false
array. This uses no lesion location, scale, interpolation or model prediction.
The model still receives the complete original native image using the frozen
grayscale/letterbox input path. Source masks and images are never rewritten.

The four nonempty references cannot be mapped reliably from the supplied
evidence. Exclude HESN_002, HESN_003, HESN_004 and HESN_005 from external
inference and overlap/distance aggregation. They are not assigned Dice=0,
Dice=1 or synthetic annotations. Keep their raw files in place: the audit still
decodes and validates their source colours, labels, sizes and byte hashes and
records them as quarantined rather than pretending they are absent.

Both rules apply only to the exact thirteen reviewed source image/mask hashes
in `breastseg/busuclm_geometry_policy.json`. Changed sources, other dimension
mismatches and nonempty normal-labelled masks continue to fail the audit.
Every raw source file is hashed again when planning evaluation, including
quarantined pairs. Eligible pairs are checked against their hashes when loaded
for inference. Exclusions and native-empty representations are retained in
the audit, manifest, runtime protocol and final combined summary.

## Effect on the reported cohort

| Analysis | Version 1 intended | Version 2 eligible |
| --- | --- | --- |
| Source dataset inventory | 683 pairs, 38 patients | Same 683 source pairs, 38 patients |
| Inference images | 683 | 679 |
| Primary positive segmentation | 264 images, 36 patients | 260 images, 35 patients |
| Benign images | 174 | 170 |
| Malignant images | 90 | 90 |
| Normal-image burden | 419 images, 35 patients | Same 419 images, 35 patients |
| No-Doppler/No-Combined eligible images | 640 | 637 |
| Unmarked No-Doppler/No-Combined eligible images | 521 | 519 |

The omission of all four positive views from one patient restricts the target
cohort and must be disclosed when comparing against published results using
different eligibility rules. The normal-image analysis still includes HESN.
Having 35 patients in each analysis does not mean the two patient sets are the
same. Patients can contribute to both normal and positive analyses, and to
both benign and malignant image-label subgroups where applicable.

Resolution 384 remains the previously validation-selected primary setting;
256 remains the reference. Seeds, checkpoint identities, threshold 0.5,
patient weighting, float32 inference, no postprocessing and uncertainty
definitions remain unchanged. No new model or parameter choice is made from
external scores. Use the new `_v2` output directories to preserve the original
failed audit and any earlier artifacts.

## Provenance

Local source evidence is the user-supplied `busuclm_geometry_diagnostics.zip`:
13 original image/mask pairs, INFO.csv, audit.json and geometry_report.json.
The machine-readable local review is in `busuclm_geometry_review.json`.
The review also covers the eight earlier supplied format examples. No full
local dataset or actual research checkpoint inference was performed.

Dataset attribution: Vallez, Bueno, Deniz, Rienda and Pastor (2025),
*BUS-UCLM: Breast ultrasound lesion segmentation dataset*, Mendeley Data V3,
[doi:10.17632/7fvgj4jsp7.3](https://doi.org/10.17632/7fvgj4jsp7.3), CC BY 4.0.
This amendment is this project's handling of the provided source discrepancies,
not an author-endorsed replacement annotation or an official dataset update.
