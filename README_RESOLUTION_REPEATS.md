# BUS-BRA resolution repeats: seeds 43 and 44

The next step is four full training runs on physical GPU 3, sequentially:

| New run | Input size | Seed |
| --- | ---: | ---: |
| 1 | 256 | 43 |
| 2 | 384 | 43 |
| 3 | 256 | 44 |
| 4 | 384 | 44 |

The two completed seed-42 runs are reused. The manifest, batch size, optimizer
settings, augmentation, threshold, training budget and early stopping remain
fixed. Each run selects its own checkpoint by validation Case Dice. No test
evaluation is performed. Largest-component filtering is not applied.

## Install and check

Save `busbra-resolution-repeats.zip` in `~/breast-ultrasound-segmentation`, then:

```bash
cd ~/breast-ultrasound-segmentation
source .venv/bin/activate
python -m zipfile -e busbra-resolution-repeats.zip .
python -m unittest discover -s tests -p test_resolution_repeats.py -v
python scripts/run_resolution_repeats.py --gpu 3 --plan
```

The plan command validates the original configurations, frozen manifests,
completed training histories and seed-42 validation evaluations. It prints the
four planned train/validation command pairs without writing files or starting
GPU work. No additional pip installation is needed.

Default source folders:

```text
runs/unet_baseline_seed42
runs/unet_baseline_384_seed42
reports/unet_baseline_seed42_val_masks
reports/unet_baseline_384_seed42_val
```

If you used different folder names, pass the appropriate `--baseline-run-dir`,
`--candidate-run-dir`, `--baseline-val-dir`, or `--candidate-val-dir` options.
Use the same options when launching the actual runner.

## Start the four repeats

To keep the runner running after closing the terminal:

```bash
nohup python -u scripts/run_resolution_repeats.py --gpu 3 \
  >> runs/resolution_repeats.log 2>&1 &
```

The append redirection preserves earlier log output if the command is run again.
Only one copy of this runner can execute at a time for this project. It starts
one training process, waits for successful completion, evaluates validation,
checks those outputs, and moves to the next run. It never starts the four runs
concurrently. Do not launch a separate manual training command for these same
run folders.

Follow progress:

```bash
tail -f runs/resolution_repeats.log
```

Training prints one line after each complete epoch and validation pass. A quiet
interval within an epoch is expected. Press Ctrl+C to stop `tail`; this does not
stop the background runner. In another terminal you can inspect physical GPU 3:

```bash
nvidia-smi -i 3
```

The four runs will take several hours. Exact duration depends on early stopping
and current machine load. The combined summary is written only after all four
new training/validation runs have completed successfully.

## Outputs

Generated configurations go under `configs/resolution_repeats/`. New run folders:

```text
runs/unet_baseline_256_seed43
runs/unet_baseline_384_seed43
runs/unet_baseline_256_seed44
runs/unet_baseline_384_seed44
```

Each validation evaluation is saved under
`reports/unet_baseline_<size>_seed<seed>_val`, including native predicted masks.
The combined outputs are:

```text
reports/resolution_repeats/protocol.json
reports/resolution_repeats/summary.json
```

`summary.json` includes all six per-run evaluations, saved configurations and
environment records, per-Case metrics, mean and sample standard deviation across
three training seeds, subgroup summaries and paired differences by training
seed. Upload this one summary file for review. There is no need to upload model
checkpoints or all six sets of predicted masks.

The comparison ranks resolutions by mean validation Case Dice over seeds 42,
43 and 44. An exact tie favours 256. The ranking is descriptive; with three
seeds it does not prove superiority. Sample standard deviation across seeds is
different from a Case-bootstrap confidence interval and from the standard error.
After this step, report three training seeds per resolution on one fixed holdout,
not ten-fold cross-validation. Keep any subsequent test evaluation separate from
configuration selection; the original test set has already been inspected.

## If a run is interrupted

Completed training and evaluation folders are reused only after checking
configuration, history, manifest, checkpoint identity and validation metrics.
An interrupted training folder is rejected because the baseline trainer does
not implement checkpoint resume. The runner does not silently overwrite it or
claim to resume it. Preserve that incomplete folder under another name and run
the same command again to restart that one training run from its fixed seed;
verified completed runs will be reused.

For an interrupted validation evaluation, preserve its incomplete report folder
under another name and rerun. The completed training checkpoint is then reused
for a fresh validation evaluation. If the original configuration comparison
fails, inspect the listed differences before launching any new jobs; do not
silently change settings to make the check pass.

## Delivery verification

Eight CPU checks cover frozen controls and manifest use, rejection of changed
hyperparameters, rejection of test evaluations, incomplete histories, mixed
checkpoints/tampered metrics, undefined HD95 handling, sequential orchestration
and completed-run reuse, six-run summary statistics, and a read-only plan.
The orchestration test uses synthetic files and a test child callback; it does
not claim to train a real model or validate CUDA performance. Actual GPU
training and real-image evaluation will run on your server using the existing
baseline scripts.
