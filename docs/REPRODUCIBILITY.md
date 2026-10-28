# Reproduction and frozen experiment records

The completed study uses one fixed Case-disjoint BUS-BRA holdout and seeds
42/43/44 at both 256 and 384 pixels. Resolution is ranked by mean validation
Case Dice, with an exact tie favouring 256. The selected primary setting is
384. All six best checkpoints are evaluated separately; there is no ensemble.

## What can be reproduced

Source data preparation, training and BUS-BRA per-checkpoint evaluation are
available through the project scripts. Raw datasets and trained checkpoints
must be supplied locally; this documentation package does not contain them.

The historical frozen evaluators verify exact source files, manifests and
checkpoint SHA256 values. A newly trained model is a new reproduction, even
with the same configuration; it is not guaranteed to have the historical
checkpoint bytes or pass those identity gates. A new external run therefore
needs its own frozen checkpoint identities and protocol, rather than disabling
the checks or presenting changed predictions as the original experiment.

Published derived JSON files and figures let readers inspect the reported
results. They do not replace native predictions, source exports or model
weights for independent rescoring.

## 1. Environment and data

Work from the repository root. Activate the existing `.venv`, or create one
on a new machine with `python3 -m venv .venv`. Install a compatible PyTorch
build using its [official installation guide](https://pytorch.org/get-started/locally/)
before the remaining baseline requirements:

```bash
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python -m pip check
```

The completed research exports recorded:

| Item | Recorded value |
| --- | --- |
| Python | 3.10.12 |
| PyTorch / bundled CUDA runtime | 2.14.1+cu126 / 12.6 |
| MONAI | 1.6.1 |
| NumPy | 2.2.6 |
| Pillow | 12.3.0 |
| SciPy | 1.15.3 |
| GPU | NVIDIA RTX A6000 |
| Physical device used for external evaluation | 3 |
| External inference batch size / workers | 16 / 0 |

These are the recorded environment values from the completed exports, not
claims that those versions are the latest or that all platforms support them.
Keep the working research environment intact when reviewing historical outputs.

Follow [dataset preparation](DATASETS_AND_ATTRIBUTION.md). The source manifest
has SHA256:

```text
e37a37ea057ff2e93728150bb36a67bcc7f47dcbb3228a78fb3bfe8f3d68820c
```

For a directly allocated physical GPU 3, set visibility before starting Python:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
```

Use scheduler-provided visibility instead when the scheduler assigns devices.
Inside this process the selected device is logical `cuda:0`.

## 2. Preparation and software checks

```bash
python scripts/prepare_splits.py
python -m unittest discover -s tests -p test_baseline.py -v
python -m unittest discover -s tests -p test_busuclm_external.py -v
python -m unittest discover -s tests -p test_busuclm_prediction_review.py -v
```

The baseline checks cover grouping, dimensions, geometry, native metrics,
empty predictions, padding loss/gradients, variable native sizes and CPU
integration with a small real MONAI model. Baseline metadata tests need the
source CSVs under `data/raw/BUSBRA`, or an explicit `BUSBRA_METADATA_DIR`.
External software tests use fixtures; passing them is not dataset-wide or
research-model validation.

Run a short GPU operational check with a fresh output directory:

```bash
python scripts/train_baseline.py --smoke --num-workers 0 \
  --run-dir runs/unet_smoke_reproduction
```

The smoke run uses partial data and one epoch. Its scores are not experiment
results and its checkpoints cannot be used by the research evaluators.

## 3. Train a new reproduction

For a new 256-pixel, seed-42 run with the baseline configuration:

```bash
python -u scripts/train_baseline.py \
  --config configs/unet_baseline.json \
  --num-workers 0 \
  --run-dir runs/unet_reproduction_256_seed42
```

This starts training. It does not resume the historical run. Existing nonempty
run directories are rejected, and automatic training resume is not implemented.
The terminal prints after each completed epoch and validation pass.

For the full study design, create independent configuration files for all
resolution/seed pairs. This snippet creates configs only; it does not launch
training and refuses to overwrite existing files:

```bash
python - <<'PY'
from pathlib import Path
import json

base = json.loads(Path("configs/unet_baseline.json").read_text())
folder = Path("configs/reproduction")
jobs = [(size, seed) for size in (256, 384) for seed in (42, 43, 44)]
paths = [folder / f"unet_{size}_seed{seed}.json" for size, seed in jobs]
if any(path.exists() for path in paths):
    raise FileExistsError("Preserve existing reproduction configurations first")
folder.mkdir(parents=True, exist_ok=True)
for (size, seed), path in zip(jobs, paths):
    config = dict(base)
    config.update(image_size=size, seed=seed, num_workers=0,
                  run_dir=f"runs/unet_reproduction_{size}_seed{seed}")
    path.write_text(json.dumps(config, indent=2) + "\n")
    print(path)
PY
```

Train each desired configuration sequentially, for example:

```bash
python -u scripts/train_baseline.py \
  --config configs/reproduction/unet_384_seed42.json --num-workers 0
```

The six configurations preserve architecture, batch size, optimizer, loss,
augmentation, training budget and threshold. Only image size and training
seed vary. Select each run's `best.pt` by validation Case Dice.

If the 256/seed42 example was already trained above, its reproduction config
points to that same completed run directory. Evaluate that run rather than
launching its training command again. To repeat training, choose a new path.

## 4. Validation and source test evaluation

Evaluate a completed checkpoint on validation, with a fresh output directory:

```bash
python scripts/evaluate_baseline.py \
  --checkpoint runs/unet_reproduction_384_seed42/best.pt \
  --split val --num-workers 0 --save-masks \
  --output-dir reports/unet_reproduction_384_seed42_val
```

Inspect validation failures and compare settings using designated training
and validation data. In the historical study, largest-component filtering
was rejected on validation. Ranking mean validation Case Dice across all
three seeds selected 384 by a small margin; exact ties would favour 256.

For a newly frozen reproduction, evaluate the predeclared checkpoint on test:

```bash
python scripts/evaluate_baseline.py \
  --checkpoint runs/unet_reproduction_384_seed42/best.pt \
  --split test --num-workers 0 --save-masks \
  --output-dir reports/unet_reproduction_384_seed42_test
```

The original source test set has already been inspected, so a reproduction
on it is still a reused holdout. Do not describe it as a new blind assessment.

## 5. Historical identities and orchestration

These exact six historical `best.pt` files were used for the completed source
test and external evaluations. Paths are relative to the repository root.

| Resolution | Seed | Best epoch | Checkpoint |
| --- | ---: | ---: | --- |
| 256 | 42 | 28 | `runs/unet_baseline_seed42/best.pt` |
| 256 | 43 | 23 | `runs/unet_baseline_256_seed43/best.pt` |
| 256 | 44 | 25 | `runs/unet_baseline_256_seed44/best.pt` |
| 384 | 42 | 20 | `runs/unet_baseline_384_seed42/best.pt` |
| 384 | 43 | 29 | `runs/unet_baseline_384_seed43/best.pt` |
| 384 | 44 | 24 | `runs/unet_baseline_384_seed44/best.pt` |

| Resolution / seed | Checkpoint SHA256 |
| --- | --- |
| 256 / 42 | `4754f8bbc1ae4ed57fffbab4256612b1c6642461fc05fea5fb2e54e15fe360f8` |
| 256 / 43 | `793b98c517e273f9c796dbab50f341672c94cf615e816989ce5bd2b8eb02fb0f` |
| 256 / 44 | `9b97730263c19634412b2d2b3cb1c1b3814ed93a27c3fb5e81c3254a41d2841e` |
| 384 / 42 | `a0d078f02c53ac201058bfbc9d0549b092dc8305d2914e9cf4d52d33410111e6` |
| 384 / 43 | `ff09abace0c861f53a2b1598ac69073be0a319093c13dd820b222fde7bf9d287` |
| 384 / 44 | `bd0f4d95382bb55976a82de3586be30b933f2b1844d703c6eedf78a3149359c4` |

With the original completed runs and required historical exports available
locally, these plan commands check identities without starting GPU work:

```bash
python scripts/run_resolution_repeats.py --gpu 3 --plan
python scripts/evaluate_frozen_resolution_test.py --gpu 3 --plan
```

The first orchestrator reused the two seed-42 runs and produced seeds 43/44.
It is a historical continuation tool, not a clean-clone six-run launcher.
The frozen test tool requires the combined historical validation summary.

## 6. Frozen BUS-UCLM V3 evaluation

With the complete original dataset and the original six checkpoints available:

```bash
python scripts/audit_busuclm_external.py
python scripts/evaluate_busuclm_external.py --gpu 3 --plan
python -u scripts/evaluate_busuclm_external.py --gpu 3
```

Continue only when the preceding step succeeds. The audit accounts for all
683 original pairs, including four quarantined references. Evaluation uses
679 eligible images, float32 inference with AMP/TF32 disabled, native-grid
threshold 0.5, no filtering, and no external parameter fitting.

Default outputs:

| Artifact | Path |
| --- | --- |
| Source audit and eligible manifest | `data/processed/busuclm_external_v2/` |
| Frozen runtime protocol | `reports/busuclm_external_frozen_v2/protocol.json` |
| Combined six-run summary | `reports/busuclm_external_frozen_v2/summary.json` |
| Each model's native masks and per-image scores | `reports/busuclm_external_frozen_v2/unet_<size>_seed<seed>/` |

Complete outputs are reused only after identity and artifact checks. Partial
evaluation folders are not resumed. Confirm the process has stopped and
preserve an incomplete run outside the evaluation folder before rerunning.
The [external protocol](BUSUCLM_EXTERNAL_PROTOCOL.md) and
[geometry amendment](BUSUCLM_GEOMETRY_AMENDMENT.md) specify eligibility and scoring.

## 7. Reproduce the selected saved-mask review

The original review has already been completed. To reproduce it from the
saved native masks without training or new inference, choose a fresh path:

```bash
python scripts/review_busuclm_predictions.py \
  --output-dir reports/busuclm_external_review_reproduced
```

This verifies the source data, original checkpoint identities and primary-run
artifacts, selects eight examples deterministically and recomputes their
native scores. It writes `predictions_review.png` and `review_summary.json`.
The displayed examples are illustrative, not a random sample.

## 8. Reproduce aggregate reporting

With the historical combined source exports still available locally:

```bash
python report_baseline_results.py \
  --validation-summary reports/resolution_repeats/summary.json \
  --test-summary reports/frozen_resolution_test/summary.json \
  --output-dir reports/baseline_final_reproduced
```

For the historical external summary, the reporting program discovers the
nested metadata path through the existing dataset helper:

```bash
python - <<'PY'
from pathlib import Path
from breastseg.busuclm_external import find_root
import subprocess
import sys

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

Both output directories must be fresh. Reporting checks exported arithmetic
and identities; it does not run inference or independently rescore every
native prediction. The generated figure/JSON files are included in the
documentation package, while the original full machine exports remain local.

## Uncertainty and interpretation

Across-seed sample SD describes these three training seeds on the same
cohort. Per-checkpoint Case/patient bootstrap intervals describe cohort
sampling conditional on fixed models. Exploratory paired bootstrap intervals
in the detailed reports average each group's scores across the three fixed
seed pairs before resampling groups; they exclude training-seed and model
selection uncertainty. Do not interpret these as interchangeable quantities.

Keep the completed experiment frozen. Any method selected using the observed
BUS-UCLM failures needs a separate independent assessment. Future improvements
must use explicitly designated training and validation data and their own
recorded evaluation plan.
