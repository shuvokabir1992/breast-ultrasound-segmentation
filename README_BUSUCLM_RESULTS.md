# External results and saved-prediction review

The aggregate evaluation and the actual selected prediction review are
complete. The report now includes the observed failure patterns. The ZIP
includes the uploaded review figure and JSON under `figures/` and `docs/`.
No additional GPU run is needed to finish reporting this baseline.

Extract `busuclm-external-results.zip` into the existing project root. It adds
the results report, machine-readable derived metrics, a results figure, a
report-reproduction script and a saved-prediction inspection script. It does
not replace the frozen inference scripts, dataset, checkpoints or evaluation
artifacts. It requires the previously installed protocol-v2 external package
and the baseline packages. The image-based review does not import Torch or
run models; it uses NumPy, Pillow, SciPy and Matplotlib already in the baseline
environment.

To install this report update:

```bash
cd ~/breast-ultrasound-segmentation
python -m zipfile -e busuclm-external-results.zip .
```

Read `BUSUCLM_EXTERNAL_RESULTS.md` together with
`figures/busuclm_external_predictions_review.png`. The report concerns one fixed
Case-disjoint source holdout, three training seeds per resolution, and a frozen
external evaluation; it does not represent completed ten-fold CV.

For a GitHub project, the report, figures, small review JSON, verification
records and reporting/review code can accompany the source code. Include
dataset attribution and a CC BY 4.0 licence link with the derived image figure.
Keep raw datasets, virtual environments, model weights, `runs/` and ZIP archives
out of ordinary source commits. Review the exact staged file list before any
commit or push.

## Optional: regenerate the saved-mask review on aries

The uploaded review has already been inspected. Only rerun this step if you
want to reproduce it; the fresh output path below preserves the existing review.

```bash
cd ~/breast-ultrasound-segmentation
python -m zipfile -e busuclm-external-results.zip .

python -m unittest discover -s tests -p test_busuclm_prediction_review.py -v

python scripts/review_busuclm_predictions.py \
  --output-dir reports/busuclm_external_review_reproduced
```

The review uses the frozen **384-pixel primary setting** and displays all three
seeds for the same selected images. It checks the complete audited source
inventory and the saved artifact hashes and summaries for those three primary
runs. It also verifies that all six original source checkpoint files and
training manifests still match their frozen identities; weights are not loaded.

Selection is deterministic from existing scores: two low-Dice patients,
patients near the middle and top of the patient-Dice ordering, normal images
spanning foreground burden from distinct patients, and one normal HESN case.
For the two low-scoring positive patients, the lowest mean-Dice view is chosen;
the middle/high controls use a view closest to that patient's mean. Normal
ordering uses mean per-image area across the three fixed models. These are
diagnostic examples and should not be presented as a representative random sample.

The script reloads the actual source image and native reference, verifies
prediction dimensions and binary labels, recomputes selected overlap/distance
and foreground scores, and compares them to the saved per-image JSON. Any
score or artifact mismatch stops the review. The original raw data, saved
predictions and research-model outputs remain unchanged.

Optional reproduced outputs:

- `reports/busuclm_external_review_reproduced/predictions_review.png`
- `reports/busuclm_external_review_reproduced/review_summary.json`

The PNG shows grayscale source images, cyan reference contours and orange
saved model predictions. Normal references are empty. Each seed's title shows
Dice for positive references or predicted foreground area for normal images.
Dataset attribution and overlay changes are included in the figure.

If the output folder already contains a review, preserve it and choose a new
one with `--output-dir reports/busuclm_external_review_2`. Other optional flags
are `--project-root`, `--data-root`, `--audit-dir` and `--evaluation-dir`.
The default input paths are the existing protocol-v2 source audit and frozen
evaluation folders.

## Reproduce the aggregate figure and numerical report

The included report and figure are already generated from the received exports;
rerunning this reporting step is optional. It checks patient-level arithmetic
and source identities without reading model prediction pixels or running models.
The following command uses the nested dataset root discovered by the existing
auditor:

```bash
python - <<'PY'
from pathlib import Path
from breastseg.busuclm_external import find_root
import subprocess, sys

root = find_root(Path("data/raw/BUSUCLM"))
subprocess.run([
    sys.executable, "report_busuclm_external_results.py",
    "--audit", "data/processed/busuclm_external_v2/audit.json",
    "--external-summary", "reports/busuclm_external_frozen_v2/summary.json",
    "--baseline-test-summary", "reports/frozen_resolution_test/summary.json",
    "--metadata", str(root / "INFO.csv"),
    "--output-dir", "reports/busuclm_results_reproduced",
], check=True)
PY
```

Use a fresh output directory. The reporting script deliberately checks this
completed experiment's identities, sizes, source implementation and fully
defined HD95 exports; it is not a general evaluator for arbitrary new datasets.
Future methods need their own frozen protocol rather than changing this report
to select settings using the external results.

## Verification performed locally

The real aggregate figure uses the uploaded six-model exports. Patient-level
arithmetic, metadata membership, counts, subgroup partitions, seed SD and
bootstrap values were checked. Raw saved masks for the 679 evaluation images
were not provided here. The actual eight-image composite gallery and review
JSON have now been received: protocol identity, metadata, dimensions and
integer-count overlap arithmetic for 24 saved predictions were checked, and
the overlays were visually inspected. These checks do not independently
recompute native pixel intersections or HD95 from the composite image. The
aries script reports successful native-mask rescoring against saved metrics.
Four CPU checks exercise deterministic gallery selection,
native binary-mask validation and score mismatch rejection. Gallery rendering
is checked with clearly marked synthetic fixtures; those fixtures are not
included as research results.

Original dataset files and model weights are not included in this ZIP.
The results document contains the dataset references, licence attribution,
geometry exclusions and limits on interpreting the scores.

The 384-pixel setting stays primary. Preserve this completed evaluation and
select future changes using designated training and validation data. BUS-UCLM
has now been observed; a method selected using its feedback needs a separate
independent assessment.
