# Fixed holdout protocol for the first BUS-BRA baseline

## Unit of separation

Use the supplied Case identifier to group related images. This establishes separation of those recorded groups. Relationships absent from the released metadata cannot be tested by this audit.

Use the original ten-fold outer assignment: test fold 1, validation fold 2, training folds 3–10. The choice is fixed before training. It is an engineering baseline, not a ten-fold aggregate or a replication of the original paper's complete protocol.

The five-fold and ten-fold valid_f flags are retained as source data and audited. They are excluded from the baseline's split construction because they divide some Cases across their 0/1 values.

## Preprocessing

Read the full PNG as grayscale. Convert pixel values to float32 in [0, 1] by dividing by 255. Do not crop with the ground-truth BBOX or use pathology as a model input.

During full dataset preparation, decode each image and mask and require identical dimensions. Use that verified PNG size for the manifest. Record differences from the original CSV dimensions in metadata_audit.json under dimension_check, while leaving the source CSV unchanged. Such a CSV discrepancy does not stop preparation. Corrupt files, mismatched image/mask shapes, and invalid masks still fail. Metadata-only preparation explicitly marks its dimensions as unverified.

Fit the image inside a 256 × 256 canvas, preserving aspect ratio. Use bilinear interpolation for image intensities and nearest-neighbour interpolation for the binary mask. Centre-pad with zeros. Track the unpadded region and both the resized and original dimensions. Stop if a nonempty lesion disappears after mask resizing.

Map mask values 0 to background and 255 to foreground. Masks with unexpected values fail explicitly. Apply a horizontal flip with probability 0.5 to image and mask together during training only. The validation and test datasets use no random augmentation.

Keep source images intact. They contain scanner text and caliper markings; marker sensitivity and scanner transfer are later experimental questions.

## Network and optimization

The model is MONAI UNet with spatial_dims=2, one input channel and one output logit channel. Feature channels are 32, 64, 128, 256 and 512; four downsampling strides are 2; num_res_units=2; normalization is InstanceNorm. This is a residual U-Net implementation and should be named that way in comparisons.

Train with AdamW and equal-weight soft Dice loss plus binary cross-entropy with logits. Compute both losses over the unpadded region. Each training image contributes one averaged loss; padding does not contribute loss or gradients.

Use mixed precision for CUDA training. Compute the loss reductions in float32 and use gradient scaling. Clip the gradient norm at 1.0. Reduce the learning rate by 0.5 after five validation epochs without improvement. Select the best checkpoint by Case-macro validation Dice; early-stop after 20 epochs without improvement.

Initial learning rate is 0.001, weight decay 0.0001, batch size 16 and maximum epochs 100. No segmentation accuracy is assumed or promised.

## Evaluation geometry and metrics

Remove the letterbox padding from sigmoid probabilities. Resize probabilities to the original image dimensions with bilinear interpolation. Apply a fixed threshold of 0.5 and compare with the original binary mask.

Report per-image Dice, IoU, pixel sensitivity/recall and precision. Report both image-macro means and Case-macro means. For Case-macro metrics, first average the metric across the images of each Case, then give each Case equal weight. This avoids giving two-image Cases double the evaluation weight of single-image Cases.

During model selection, validation uses the same threshold and geometry. The final export evaluates in float32; training-time validation may use mixed-precision forward passes.

HD95 uses native image pixels. Physical spacing is not established by these CSVs, so results must not be labelled millimetres. Extract four-neighbour mask boundaries, calculate Euclidean distances in both directions, and take the larger of the two directional 95th percentiles.

## Empty masks

| Target | Prediction | Dice/IoU | Precision/recall | HD95 |
| --- | --- | --- | --- | --- |
| Empty | Empty | 1 | 1 | 0 px |
| Nonempty | Empty | 0 | 0 | Undefined |
| Empty | Nonempty | 0 | 0 | Undefined |

BUS-BRA's supplied audit reports no empty targets. Empty predictions still need explicit handling. Undefined HD95 values are written as null, not as zero or a fabricated finite distance. The report provides the mean over defined image distances together with the defined and undefined counts. A Case distance is reported only if every image in that Case has a defined distance.

Dice and IoU include failures from empty predictions. Never report the finite-only HD95 mean without its undefined counts.

The Dice confidence interval uses 2,000 bootstrap resamples of Case-level mean Dice, with seed 42. It describes sampling variation within this holdout under the supplied grouping; it does not capture scanner shift, missing patient links or training-seed variation.

## Reproducibility

Record the resolved configuration, Python/library versions, GPU name, seeds, source metadata hashes, and manifest hash. Copy the manifest into the run directory. Reject evaluation with a different manifest. Request deterministic algorithms with warnings and disable cuDNN benchmarking. These controls do not promise bitwise equality across hardware and library releases.

The initial trainer saves best.pt and last.pt atomically, plus optimizer/scaler/scheduler state. Automatic resume is not implemented in this first version. Use fresh run directories to keep experiment outputs distinct.

## Staged experiment plan

1. Rebuild the manifest against the real image files.
2. Run all CPU checks and the two-batch CUDA smoke test.
3. Train the fixed baseline and inspect validation metrics, learning curves and failed examples.
4. Freeze choices before evaluating the held-out test fold.
5. Repeat seeds and/or expand to cross-validation before a publication-level comparison.
6. Establish BUS-UCLM annotation conventions and a held-out external evaluation protocol.

The supplied metadata check is a reproducibility finding for this project. It is not a claim of a novel medical method or a conclusion about the original authors' reported results.
