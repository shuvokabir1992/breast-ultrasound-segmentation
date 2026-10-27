# Breast Ultrasound Lesion Segmentation — Research Starter

**Purpose:** A reproducible breast-ultrasound *lesion segmentation* research portfolio project. Research use only; not a diagnostic medical device.

## First milestone: dataset verification (no GPU required)

1. Download the original **BUSBRA.zip** from https://zenodo.org/records/8231412.
2. Extract it under `data/raw/` while keeping the dataset's original file layout and CSV metadata:

   ```bash
   mkdir -p data/raw
   unzip /path/to/BUSBRA.zip -d data/raw/
   ```

3. On Linux, create a clean environment and install the audit requirements:

   ```bash
   conda create -n breast_seg python=3.11 -y
   conda activate breast_seg
   python -m pip install -r requirements-audit.txt
   ```

4. Audit image-mask pairs, image dimensions and metadata:

   ```bash
   python scripts/audit_data.py --root data/raw --output reports/audit.json
   python scripts/visualize_samples.py --root data/raw --output reports/preview.png --num 8
   ```

5. Inspect `reports/audit.json`, `reports/preview.png`, and the original dataset's CSV files. **Do not split by individual images** when multiple images correspond to the same patient. BUS-BRA provides official 5/10-fold partitions; inspect and validate those before training.

6. Inspect GPU support:

   ```bash
   nvidia-smi
   ```

   Install the matching CUDA-enabled PyTorch wheel using the official selector at https://pytorch.org/get-started/locally/ and then install MONAI (`python -m pip install monai`). Verify:

   ```bash
   python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
   ```

## Project direction

**Core task:** binary segmentation of breast lesions (not malignant/benign diagnosis). Start with a 2D U-Net baseline, then benchmark nnU-Net v2 and one foundation-model approach as resources allow. Train/validate on properly partitioned BUS-BRA; use BUS-UCLM as a strictly held-out external dataset after checking annotation conventions and terms.

**Suggested metrics:** patient-level mean Dice, IoU, sensitivity/recall, precision and 95th percentile Hausdorff distance (HD95). Handle empty masks explicitly and report Dice confidence intervals when possible. Report performance by lesion size if metadata/annotation supports it.

**Research goal:** compare accuracy, stability, and generalization across datasets. The publication hypothesis must be established only after checking prior work and running reliable baselines.

## File structure

```text
breast-ultrasound-segmentation-starter/
├── README.md
├── requirements-audit.txt
├── .gitignore
├── data/                   # user-downloaded datasets, not committed
├── docs/research_plan.md
├── scripts/audit_data.py
├── scripts/visualize_samples.py
├── tests/test_audit.py
├── configs/                # future training configurations
├── models/                 # future model source code
├── checkpoints/            # never commit private/work-owned weights
└── reports/                # generated QA results
```

## Data and publication ethics

- Source: https://zenodo.org/records/8231412 (Gómez-Flores et al., *Medical Physics*, 2024, DOI: 10.1002/mp.16812). Any research using BUS-BRA must cite its paper. Check the dataset's usage terms before re-sharing images or derived assets.
- Do not commit datasets, model weights without distribution rights, patient-identifying metadata, or data-derived images unless redistribution is permitted.
- Use compute resources and transfer model checkpoints only with the infrastructure owner's approval.
- An example segmentation result does not establish clinical validity; use the demo for research and education only.
