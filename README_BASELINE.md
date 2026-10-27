# BUS-BRA lesion segmentation baseline

A full-image, binary segmentation baseline built around the BUS-BRA metadata supplied on 27 October 2026. The first model is a 2D MONAI residual U-Net. This package extends the existing breast-ultrasound-segmentation project.

## What the metadata check found

All 1,875 image identifiers match across bus_data.csv, 5-fold-cv.csv and 10-fold-cv.csv. The Case column contains 1,064 groups: 811 with two related images and 253 with one. All images within a Case have the same pathology and device.

The outer kFold assignments in both files keep every Case together. The valid_f flags do not consistently do this. For example, 244 Cases have both 0 and 1 among their images in valid_1 of the 10-fold file. Using those flags for a training/validation partition would mix related images between the two subsets.

This baseline uses whole outer folds from 10-fold-cv.csv:

| Split | Supplied folds | Images | Cases | Benign images | Malignant images |
| --- | --- | ---: | ---: | ---: | ---: |
| Training | 3–10 | 1,489 | 851 | 1,009 | 480 |
| Validation | 2 | 193 | 107 | 130 | 63 |
| Test | 1 | 193 | 106 | 129 | 64 |

There are zero shared Case IDs between these subsets. This is one fixed holdout experiment, not a completed ten-fold cross-validation experiment. The supplied valid_f columns are audited but do not determine this baseline's splits.

## Install into the existing project

Extract the package into the existing project folder on aries. The files scripts/train_baseline.py and breastseg/torch_pipeline.py should be directly beneath that project folder.

Run the following in the current virtual environment, from the project root:

    cd ~/breast-ultrasound-segmentation
    source .venv/bin/activate
    python -m pip install -r requirements-baseline.txt
    python -m pip check
    python -m pip freeze > requirements-lock.txt

Keep the CUDA-enabled PyTorch installation already verified on aries. That environment reported Python 3.10.12, PyTorch 2.14.1+cu126 and MONAI 1.6.1.

For a directly allocated GPU 0, use:

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0

If a scheduler assigns the GPU, retain its settings.

## Rebuild and verify the split

    python scripts/prepare_splits.py

This checks the CSV identifiers and labels, both sets of folds, image decoding, mask values, image/mask dimensions, and correspondence with the dimensions in bus_data.csv. It uses the dimensions of the decoded, matching PNG pair for the manifest. Differences from the CSV are recorded in the audit and printed as warnings; the original CSV is preserved. Image/mask size disagreement, corrupt images, and invalid masks still stop preparation. It then writes:

- data/processed/busbra_manifest.csv
- data/processed/metadata_audit.json

The package already includes these outputs generated from the supplied CSVs. Its initial report explicitly records that image files were unavailable in this environment. Running the command on aries verifies the actual images and replaces that report.

Dataset filenames and metadata stay in their original layout under data/raw/BUSBRA. The manifest uses paths relative to that directory.

## Run the checks

    python -m unittest discover -s tests -p test_baseline.py -v

The tests cover Case separation, ambiguous identifiers, row-order independence, mask geometry, tiny-lesion loss during resizing, analytic overlap metrics, HD95 units, empty predictions and Case-weighted aggregation. When PyTorch and MONAI are present, they also exercise a small real model forward/backward pass, checkpoint loading, loss masking and variable-size batching on CPU.

The metadata tests use data/raw/BUSBRA by default. To test a separate metadata folder, set BUSBRA_METADATA_DIR to that folder.

The verification record in docs/verification.json lists the checks run for this packaged version. The CPU model checks use synthetic images. The original BUS-BRA PNGs and CUDA execution remain to be checked on aries. Python source is also checked against Python 3.10 grammar.

## Run a short GPU smoke test

    python scripts/train_baseline.py --smoke --num-workers 0

This runs two training batches and two validation batches for one epoch. It checks data loading, CUDA forward/backward operations, mixed precision, metrics and checkpoint saving. Its outputs go into runs/unet_smoke. These partial-data metrics are only operational checks. Evaluation scripts reject smoke checkpoints.

If a run directory already contains files, choose a fresh path:

    python scripts/train_baseline.py --smoke --num-workers 0 --run-dir runs/unet_smoke_02

## Train the baseline

After the smoke test passes:

    python scripts/train_baseline.py --config configs/unet_baseline.json

The configuration starts with 256 × 256 inputs, batch size 16, AdamW, learning rate 0.001, weight decay 0.0001, up to 100 epochs and early stopping after 20 epochs without validation improvement. These are initial settings, not tuned results.

Training saves the best checkpoint by validation Case-macro Dice, the last checkpoint, epoch history, resolved configuration, environment details and a copy of the manifest. The test images are not used during training or checkpoint selection.

Choose a new run directory for each experiment. This initial trainer saves optimizer/scaler state for inspection but does not implement automatic resume.

## Inspect validation performance

    python scripts/evaluate_baseline.py --checkpoint runs/unet_baseline/best.pt --split val --output-dir reports/unet_validation

The output includes per-image metrics, image- and Case-macro summaries, benign/malignant and scanner breakdowns, a Case-level bootstrap interval for Dice, and eight examples selected with a fixed seed.

Freeze the protocol and model choices before a final test evaluation:

    python scripts/evaluate_baseline.py --checkpoint runs/unet_baseline/best.pt --split test --output-dir reports/unet_test

Use a new output directory each time. Optional --save-masks writes native-size binary prediction PNGs.

## Predict one image

    python scripts/predict_baseline.py --checkpoint runs/unet_baseline/best.pt --image data/raw/BUSBRA/Images/bus_0001-l.png --output reports/example_prediction.png

This needs only the image and checkpoint; it does not use a ground-truth mask or bounding box. Use evaluate_baseline.py for performance claims on a defined subset.

## Method and interpretation

See docs/BASELINE_PROTOCOL.md for preprocessing, loss, aggregation, empty-mask conventions, native-resolution HD95, reproducibility and limitations. See data/processed/metadata_audit.json for the machine-readable fold audit.

This is a research baseline for a portfolio and later experiments. It does not establish clinical validity or research novelty. There are no trained weights or measured segmentation results in this package.

## Sources

- Supplied files: bus_data.csv, 5-fold-cv.csv, 10-fold-cv.csv; their SHA-256 hashes are retained in metadata_audit.json.
- Gómez-Flores, Gregorio-Calas and de Albuquerque Pereira. BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems. Medical Physics 51, 3110–3123 (2024). https://doi.org/10.1002/mp.16812
- Original dataset: https://zenodo.org/records/8231412
- Author repository: https://github.com/wgomezf/BUS-BRA
- MONAI network API: https://monai.readthedocs.io/en/stable/networks.html
- PyTorch mixed precision API: https://docs.pytorch.org/docs/2.14/amp.html
