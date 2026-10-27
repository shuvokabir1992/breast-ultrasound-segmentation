# BUS-BRA validation failure review

This add-on uses the existing baseline checkpoint and evaluation scripts. It adds
a targeted visual review and an optional, fixed largest-component comparison on
the full validation split. It does not retrain the model or change saved results.
It reads training reference masks and validation images/masks, but no test images
or test masks. Run commands from the project root with the existing venv active.

## 1. Install the add-on

Save `busbra-validation-review.zip` in `~/breast-ultrasound-segmentation` and run:

```bash
cd ~/breast-ultrasound-segmentation
python -m zipfile -e busbra-validation-review.zip .
python -m unittest discover -s tests -p test_validation_review.py -v
```

No new packages are needed beyond the baseline requirements. The add-on contains
one new script, its tests, and this guide. Existing baseline files are not replaced.

## 2. Export all original validation predictions

Use a fresh output directory. This is validation inference, not training:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
python -u scripts/evaluate_baseline.py \
  --checkpoint runs/unet_baseline_seed42/best.pt \
  --split val \
  --num-workers 0 \
  --save-masks \
  --output-dir reports/unet_baseline_seed42_val_masks
```

If that directory already contains this exact completed validation evaluation and
all 193 predicted masks, reuse it. Otherwise choose a new directory and use the
same path for `--eval-dir` below. The scripts refuse nonempty output directories.

## 3. Make targeted overlays and compare one candidate

```bash
python scripts/review_validation_failures.py \
  --run-dir runs/unet_baseline_seed42 \
  --eval-dir reports/unet_baseline_seed42_val_masks \
  --output-dir reports/unet_baseline_seed42_validation_review \
  --compare-largest-component
```

This second command runs on CPU and uses the already saved native-size masks.
Omit `--compare-largest-component` for a diagnostic-only run. Use a different
output directory when rerunning it.

Outputs:

- `failure_overview.png`: six named validation failures, with original image,
  reference mask, original prediction and pixel errors. Green means correct
  foreground, orange means extra foreground, and blue means missed foreground.
- `review_summary.json`: verified original metrics, training/validation reference
  component counts, candidate metrics by pathology/scanner, and a paired Case
  bootstrap interval for the candidate-minus-baseline Dice difference.
- `review_per_image.csv`: original metrics, component sizes and candidate changes.
- `reference_components.csv`: component audit for all training and validation
  reference masks. Connected regions are not automatically separate lesions.

The default overview includes `bus_0452-r`, `bus_0452-l`, `bus_1047-r`,
`bus_1047-l`, `bus_0026-s` and `bus_0321-s`. To inspect other validation images,
pass `--ids ID1 ID2 ...` and choose a fresh output directory. All IDs must belong
to the validation split.

## 4. Interpret the images before selecting a change

Check whether each failure is predominantly extra foreground, missed foreground,
a boundary disagreement, or an image/reference alignment concern. For
`bus_0026-s`, check both reference and prediction for distant regions: its large
HD95 is not explained by Dice alone. Record image-specific observations without
changing a prediction or annotation by hand to improve the score.

The candidate rule keeps the largest eight-connected predicted foreground region
in every image. It uses no ground truth and keeps the threshold at 0.5. Diagonally
touching pixels belong to the same region. HD95 retains the original baseline
definition. The candidate is an experiment, not an automatically selected fix.
This add-on measures the candidate; the original training, evaluation and
prediction scripts retain their existing behavior. If the candidate is selected,
implement the same fixed rule consistently for subsequent evaluation and use.

Review the reference audit to understand the risk of removing real annotated
foreground, including disconnected parts of one lesion. Inspect the largest Dice
losses and gains listed in the JSON, not just the six original examples. The
baseline and candidate use the same complete validation cohort and the same
per-Case weighting. The bootstrap samples Cases as pairs, preserving both views.
It does not measure variation from retraining the model with different seeds.

Use validation Case Dice as the primary selection metric, consistent with the
baseline checkpoint rule. Assess HD95, recall, pathology groups and qualitative
losses as secondary evidence. A lower HD95 alone is insufficient to justify
losing annotated foreground. If the evidence is unclear, retain the original
baseline. A single predeclared comparison is more interpretable than searching
many component rules on this small validation cohort.

If missed extent or boundary detail dominates, largest-component filtering cannot
recover missing pixels. A later experiment could change one training setting,
such as image resolution, while holding the manifest, seed, training budget and
selection rule fixed. Choose that experiment after inspecting the review. Do not
change the split, remove difficult cases, use a ground-truth crop as input, or
choose a separate postprocessing rule for each image.

## 5. Record the decision

Before another test evaluation, write down the candidate, what changed, why,
the full-validation results, important regressions, and whether it was retained.
Keep the original run and test report. The original test results have already
been inspected; do not use subsequent test scores to choose hyperparameters.
Any future score on this same test set is a comparison on a reused holdout.

Suggested wording for the completed original experiment:

> We trained a 2D MONAI residual U-Net with seed 42 using one fixed, Case-disjoint
> holdout derived from the supplied BUS-BRA outer ten-fold assignments. Folds
> 3-10 were used for training (1,489 images, 851 Cases), fold 2 for validation
> (193 images, 107 Cases), and fold 1 for testing (193 images, 106 Cases). No
> Case appeared in more than one split. The checkpoint was selected by validation
> Case-macro Dice at epoch 28. With a threshold of 0.5 and native-resolution
> evaluation, test Case-macro Dice was 0.8815 (95% Case-bootstrap confidence
> interval 0.8537-0.9037; 2,000 resamples). This is a single-seed fixed-holdout
> experiment; ten-fold cross-validation was not performed.

The Case-bootstrap interval describes variability from sampling the evaluated
Cases under this model and protocol. It does not establish training-seed
stability or performance on an external dataset.

## Verification supplied with this add-on

Six CPU checks cover detached false positives, loss of genuine disconnected
foreground, empty masks and diagonal connectivity, paired Case weighting,
test-split rejection, and a synthetic end-to-end review. The integration check
uses native-size masks, verifies unchanged inputs, rejects tampered metrics, and
succeeds without any test images or masks being present. Real BUS-BRA PNGs and
CUDA inference were not available in the delivery environment.
