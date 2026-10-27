# Initial research protocol (draft, not a novelty claim)

## Stage A: Data quality and honest evaluation

- Verify matching ultrasound images and lesion masks, dimensions, annotation values, and unexpected missing/empty masks.
- Inspect official BUS-BRA CSV partition definitions and metadata. Confirm whether all scans belonging to the same patient occur in only one partition. Do not assume filename ID represents patient ID.
- Reserve test patients/folds from training, model selection, threshold tuning, or preprocessing fit. Use validation only for model choices.
- Preserve original aspect ratio where reasonable; resize image and mask with appropriate interpolation (bilinear for image; nearest for label mask).

## Stage B: Model baseline

- Start with 2D U-Net, binary Dice+BCE loss, deterministic seeds, logged software versions, augmentation limited to training, and checkpoint selection using patient-level validation metrics.
- Record learning curves, GPU memory, inference latency, model parameters and per-case predictions.

## Stage C: Benchmarks and external validity

- Compare a well-defined nnU-Net v2 baseline and a suitable publicly available pretrained model. Fix splits and evaluation rules.
- Train on BUS-BRA, externally evaluate on BUS-UCLM only after harmonizing target label semantics. Do not tune on the held-out external set.
- Inspect subgroup performance where labels support it (small versus large lesions, scanners, clinical strata); report uncertainty.

## Stage D: Candidate research question

Can a method that improves robustness to scanner/contrast variation reduce the performance drop between in-domain and external-dataset breast-lesion segmentation?

This is an **investigation topic**, not a novel technique or expected result. Assess recent literature and conduct ablations before deciding whether the work supports a paper.

## Deliverables

1. Audited dataset manifest and documented partitions.
2. Reproducible model code/configs and baseline checkpoints.
3. Qualitative error analysis and metrics with confidence intervals.
4. A research/education demo and clear GitHub README.
5. Paper draft only if comparisons demonstrate a meaningful contribution.
