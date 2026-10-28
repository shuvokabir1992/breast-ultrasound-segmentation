# BUS-UCLM V3 external evaluation protocol, version 2

Amended on 28 October 2026 after inspecting dataset metadata, eight initial
source pairs and all thirteen HESN geometry diagnostic pairs, before running
the frozen research models on BUS-UCLM. No external model predictions or
performance scores were used to choose eligibility or geometry handling.
The original protocol is preserved in
[version 1](BUSUCLM_EXTERNAL_PROTOCOL_V1.md); the reason and exact source
identities for the amendment are documented in
[the geometry amendment](BUSUCLM_GEOMETRY_AMENDMENT.md).

## Frozen models and analysis scope

The source study is one fixed Case-disjoint BUS-BRA holdout with three training
seeds per resolution, not completed ten-fold cross-validation. The 384-pixel
resolution was selected by mean BUS-BRA validation Case Dice before the final
test evaluations. It remains the external primary model setting. The 256-pixel
resolution is a prespecified reference, including its higher observed BUS-BRA
test Dice. All six validation-selected best checkpoints must match their frozen
SHA256 identities, copied BUS-BRA manifest and recorded epochs.

BUS-UCLM is used for external inference only. No training, fine-tuning,
normalization fitting, threshold search, seed/checkpoint selection, ensemble,
largest-component filtering or other fitted postprocessing is permitted in
this experiment. There are three seed results per setting. They reuse the same
external patients and are not three independent external cohorts.

## Dataset, identity and formats

Source: BUS-UCLM V3, DOI 10.17632/7fvgj4jsp7.3, published 27 January 2025.
The official record lists CC BY 4.0. The exact supplied INFO.csv SHA256 is
`a9d5356f0639caa1bafa5d46de58c70e81894c39ddb4feff10a7858d87d9e91c`.
The metadata enumerates 683 images: 174 benign, 90 malignant and 419 normal,
with 38 four-letter patient prefixes. The image filename prefix before `_`
is used for grouping, consistent with the authors' tools.

Every listed image and mask must exist and decode as a single PNG, including
quarantined pairs. Source bytes and decoded dimensions are recorded. Native
image sizes take precedence over CSV sizes; source mask dimensions remain
recorded separately. Except for thirteen byte-identified HESN pairs described
below, image/mask dimensions must agree. Unexpected mask colours, missing or
corrupt files, nonopaque transparency, changed exception source bytes, and
label/mask inconsistencies fail the audit.

Nine HESN normal masks are verified to be entirely black despite different
image dimensions. They are represented as all-false arrays on their native
image grids in memory; this introduces no foreground and uses no spatial
interpolation or offset guess. The four nonempty HESN references, HESN_002
through HESN_005, are quarantined because no validated image/mask coordinate
mapping could be established from the supplied files. No positive mask is
resized, shifted, reannotated or assigned a failure score to bypass this issue.
All original files and the original CSV remain unchanged.

RGB black is background, green `(0,255,0)` is benign lesion and red `(255,0,0)`
is malignant lesion. Their foreground union forms the binary lesion reference.
Normal-labelled references must be entirely black; positive-labelled source
masks, including the four quarantined pairs, must contain the corresponding
lesion colour. Unexpected inconsistencies stop inference.

All eight initial examples decode at 856 by 606 pixels. All thirteen HESN
images have different native dimensions from their 856-by-606 source masks.
These 21 local inspections do not substitute for the full source audit on
aries. The original RGB masks are
never converted to grayscale and thresholded as a substitute for colour labels.

## Cohorts declared before predictions

The primary segmentation cohort includes **all 260 eligible lesion-positive
images from 35 patient prefixes**: 170 benign and 90 malignant images. Four
benign HESN images are excluded solely for unresolved nonempty-reference
geometry. HESN no longer contributes to the primary positive cohort but its
nine normal images remain in the normal-image analysis. The remaining 679
eligible images still cover all 38 source patient prefixes. This is an
eligibility-restricted external evaluation, not segmentation evaluation on
every image in the original 683-image dataset.

No image is removed for visible marks, Doppler colour, composite views, lesion
size, model performance or expected difficulty. Combined images are kept whole;
they are not split, manually cropped or rescaled by view.

Normal-image analysis includes **all 419 normal-reference images from 35
patients**. Normal references do not enter lesion-positive Dice/IoU averages.
Most patients have both normal and lesion-positive images, so these normal-image
outputs do not establish healthy-patient diagnostic specificity.

Prespecified secondary descriptions are:

| Subset | Rule in original CSV | Images | Lesion-positive | Normal |
| --- | --- | ---: | ---: | ---: |
| Plain B-mode | Doppler=No and Combined=No | 637 | 224 | 413 |
| Unmarked plain B-mode | Above plus Marks=No | 519 | 115 | 404 |

Additional descriptive subsets use Doppler=Yes, unresolved Doppler,
Combined=Yes, Marks=Yes and Marks=No. These can overlap and must not be added
together as independent groups. Benign and malignant segmentation groups use
the image's original Label value. CHVI, FLBA, FUHI, HUBL and MENE each have both
benign and malignant images; patients must not be forced into one pathology.

POFR_013 has Doppler=`Y`. Retain this as an unresolved flag rather than
silently imputing a value. Include it in the full cohort, give it an unresolved
subgroup, and exclude it from the explicit Doppler=No B-mode subsets. The raw
CSV remains unchanged. No primary/secondary cohort is promoted or discarded
after inspecting external scores.

## Preprocessing and inference

Reuse the baseline image-only path: convert the whole image with Pillow `L`,
divide by 255, resize bilinearly with aspect ratio preserved, and center it in
a zero-padded square at the checkpoint's original 256 or 384 input size. No
reference mask affects image cropping, normalization, resizing or model input.
There are no flips, rotations, test-time augmentation or external fitting.

Use model.eval() and inference_mode(), float32 inference with AMP/TF32 disabled,
batch size 16 and zero workers by default. Remove letterbox padding from
sigmoid probabilities and bilinearly restore them to original H/W with
align_corners=False, then apply the frozen probability threshold >=0.5.
Save the resulting unfiltered binary mask. Seeds and package/device versions
are recorded; deterministic operations are requested with warnings. Bitwise
portability across hardware/software is not guaranteed.

## Aggregation, distances and normal-image burden

For positive references, compute Dice, IoU, pixel recall and pixel precision
per image. Average eligible images within each patient, then average patients
equally. Apply the same patient weighting separately within every subgroup.
Pixel recall is a segmentation statistic, not diagnostic sensitivity or lesion
detection probability.

HD95 is the maximum of the two directional 95th-percentile Euclidean boundary
distances using four-neighbour boundaries. Report native image pixels, because
physical spacing is not established by INFO.csv. Empty prediction against a
positive target gives overlap scores zero and HD95 null. A patient's HD95 is
defined only if every eligible image's HD95 is defined. Report defined and
undefined image/patient counts alongside conditional distance means.

On normal references, report predicted foreground pixels divided by the full
native image area, and the fraction of images having at least one predicted
foreground pixel. The image area includes the original borders/overlays, as
the frozen model uses the whole image. Also average these measures within
patient, then across patients equally. Record how many patients have any
foreground on at least one normal image. The any-pixel definition has no
minimum component size or filtering. Normal overlap scores remain null, even
when both prediction and reference are empty; they cannot inflate primary Dice.

## Uncertainty and reporting

Report each seed and mean plus sample SD across seeds 42/43/44. SD describes
training-seed variability on this fixed cohort, not a confidence interval.
For each fixed checkpoint's full positive cohort, additionally report a
percentile Dice interval from 2,000 resampled patients with bootstrap RNG seed
42. This interval is conditional on that checkpoint, excludes training-seed
and model-selection uncertainty, and does not count images as independent.

Report benign/malignant image-label subsets and acquisition flags descriptively.
Do not infer scanner causation, clinical diagnostic validity or general model
superiority from these subgroup results. Keep the selected 384-pixel model
identified as primary regardless of the external ranking. A future method
chosen using these external results would require a new independent assessment.

The runtime saves the protocol, source dataset fingerprint, source-code hashes,
geometry policy hash, all thirteen reviewed exception identities, the four
quarantined source records, all six checkpoint identities, full eligible
per-image scores and patient summaries.
Complete outputs can be reused only after identity, artifact and metric checks;
partial outputs are preserved for review and not resumed silently.
