"""Evaluate all six frozen BUS-BRA checkpoints on audited BUS-UCLM V3."""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import platform
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from breastseg.busuclm_external import (
    FROZEN, MODEL, TRAIN_MANIFEST_SHA256, aggregate_runs, decode_pair, digest,
    json_bytes, load_audit, read_json, score_prediction, sha256,
    summarize_external, validate_scores,
)


def make_jobs(root, frozen=FROZEN):
    jobs = []
    for size, seed, epoch, run_name, expected_hash in frozen:
        checkpoint = root / "runs" / run_name / "best.pt"
        if sha256(checkpoint) != expected_hash:
            raise ValueError(f"Frozen checkpoint hash mismatch: {checkpoint}")
        if sha256(checkpoint.parent / "manifest.csv") != TRAIN_MANIFEST_SHA256:
            raise ValueError(f"Frozen BUS-BRA training manifest mismatch: {checkpoint.parent}")
        jobs.append({"image_size": size, "seed": seed, "epoch": epoch,
                     "checkpoint": checkpoint, "checkpoint_sha256": expected_hash,
                     "name": f"unet_{size}_seed{seed}"})
    return jobs


def make_protocol(root, report, jobs, batch_size=16, num_workers=0):
    files = ("breastseg/preprocess.py", "breastseg/metrics.py", "breastseg/torch_pipeline.py",
             "breastseg/busuclm_external.py", "breastseg/busuclm_geometry_policy.json", "scripts/audit_busuclm_external.py",
             "scripts/evaluate_busuclm_external.py")
    return {"protocol_version": 2, "dataset": "BUS-UCLM", "version": 3,
            "dataset_doi": "10.17632/7fvgj4jsp7.3", "licence": "CC BY 4.0",
            "external_manifest_sha256": report["manifest_sha256"],
            "metadata_sha256": report["metadata_sha256"],
            "geometry_policy_sha256": report["geometry_policy_sha256"],
            "source_images": report["metadata_images"], "eligible_images": report["eligible_pairs"],
            "quarantined_records": report["quarantined_records"],
            "empty_reference_native_grid_ids": report["empty_reference_native_grid_ids"],
            "training_manifest_sha256": TRAIN_MANIFEST_SHA256,
            "primary_resolution": 384, "reference_resolution": 256, "seeds": [42, 43, 44],
            "selection": "384 selected previously using BUS-BRA validation; no selection using BUS-UCLM",
            "checkpoints": [{k: j[k] for k in ("image_size", "seed", "epoch", "checkpoint_sha256")} for j in jobs],
            "threshold": 0.5, "postprocessing": "none", "precision": "float32; AMP and TF32 disabled",
            "batch_size": batch_size, "num_workers": num_workers,
            "preprocessing": "Whole original image converted to Pillow L, divided by 255; aspect-preserving bilinear resize and centered zero letterbox. No crop, view splitting, augmentation or external fitting.",
            "mask_rule": "Exact green or red RGB pixels form a binary lesion union. Matching positive masks remain native and are never supplied to the model. Nine byte-identified all-black normal masks are represented on their native image grids without interpolation; four byte-identified nonempty mismatched references are quarantined without spatial guessing.",
            "primary_cohort": "All 260 eligible lesion-positive images, averaged within 35 patients and then equally across patients; HESN_002 through HESN_005 excluded for unresolved reference geometry before external predictions",
            "normal_cohort": "All 419 normal-reference images from 35 patients; report foreground pixel fractions and any-foreground image rates separately; no normal-image Dice enters the primary average",
            "secondary_cohorts": {
                "plain_bmode": "Doppler=No AND Combined=No (637 eligible images; 224 lesion-positive, 413 normal)",
                "unmarked_plain_bmode": "Doppler=No AND Combined=No AND Marks=No (519 eligible images; 115 lesion-positive, 404 normal)",
                "others": "Doppler=Yes; unresolved Doppler; Combined=Yes; Marks=Yes; Marks=No; benign and malignant image-label subgroups"},
            "subgroup_rule": "Image-level eligibility followed by equal patient weighting within that subgroup; one patient may enter both benign and malignant subgroups",
            "doppler_Y_rule": "Preserve Y as unresolved; include in all-images primary cohort, exclude from No/No B-mode cohort",
            "hd95": "Max of directional 95th-percentile Euclidean boundary distances with four-neighbour boundaries, in native pixels. Empty lesion prediction: overlap=0, HD95=null; retain undefined image and patient counts.",
            "bootstrap": "2000 patient resamples, RNG seed 42, conditional on each fixed checkpoint; seed SD reported separately",
            "external_use": "All six models evaluated without training, fine-tuning, threshold fitting, checkpoint/seed selection or postprocessing selection on BUS-UCLM",
            "implementation_sha256": {name: sha256(root / name) for name in files}}


class ExternalDataset:
    def __init__(self, records, root, size):
        self.records, self.root, self.size = records, root, size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        from breastseg.preprocess import letterbox
        row = self.records[index]
        image, truth, _ = decode_pair(self.root, row, verify_hashes=True)
        item = letterbox(image, mask=None, size=self.size)
        return {"image": torch.from_numpy(item["image"][None]), "layout": item["layout"],
                "native_mask": truth, "record": row}


def identity(job, protocol_hash):
    return {"protocol_sha256": protocol_hash, "image_size": job["image_size"],
            "seed": job["seed"], "checkpoint_epoch": job["epoch"],
            "checkpoint_sha256": job["checkpoint_sha256"]}


def read_result(output, job, records, protocol_hash):
    marker = output / "completion.json"
    if not marker.is_file():
        raise ValueError(f"Incomplete evaluation: {output}. Move this incomplete run folder outside the output directory before rerunning; keep complete runs.")
    complete = read_json(marker)
    expected = identity(job, protocol_hash)
    expected_files = {"metrics.json", "per_image.json", "per_image.csv"}
    expected_files |= {f"predicted_masks/{r['id']}.png" for r in records}
    if complete["identity"] != expected or set(complete["files"]) != expected_files:
        raise ValueError("Saved external result identity or file inventory differs")
    for name, expected_hash in complete["files"].items():
        if sha256(output / name) != expected_hash:
            raise ValueError(f"Saved result file changed: {output / name}")
    rows = read_json(output / "per_image.json")
    validate_scores(rows, records)
    document = read_json(output / "metrics.json")
    recomputed = summarize_external(rows)
    if document["identity"] != expected or document["metrics"] != recomputed:
        raise ValueError("Recomputed patient/subgroup metrics differ from the saved export")
    return {"image_size": job["image_size"], "seed": job["seed"],
            "role": "primary" if job["image_size"] == 384 else "reference",
            "checkpoint_epoch": job["epoch"], "checkpoint_sha256": job["checkpoint_sha256"],
            "environment": document["environment"], "metrics": recomputed}


def evaluate_one(job, records, root, output, protocol_hash, device_name="cuda", batch_size=16, num_workers=0):
    import torch
    from torch.utils.data import DataLoader
    from breastseg.torch_pipeline import build_model, collate_batch, restore_probability

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty per-run directory: {output}")
    if sha256(job["checkpoint"]) != job["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after planning")
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if (checkpoint.get("smoke", False) or int(checkpoint["epoch"]) != job["epoch"] or
            int(config["image_size"]) != job["image_size"] or int(config["seed"]) != job["seed"] or
            float(config["threshold"]) != 0.5 or checkpoint["manifest_sha256"] != TRAIN_MANIFEST_SHA256):
        raise ValueError("Checkpoint settings differ from planned identity")
    # Model architecture is also checked in main against the frozen source configuration.
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no inference started")
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    dataset = ExternalDataset(records, root, job["image_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_batch, pin_memory=device.type == "cuda", drop_last=False)
    output.mkdir(parents=True, exist_ok=True)
    masks = output / "predicted_masks"
    masks.mkdir()
    scored = []
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            logits = model(batch["image"].to(device, non_blocking=True))
            probability = torch.sigmoid(logits.float())
            for index, row in enumerate(batch["record"]):
                native = restore_probability(probability[index], batch["layout"][index]).cpu().numpy()
                prediction = native >= 0.5
                scored.append(score_prediction(row, prediction, batch["native_mask"][index]))
                Image.fromarray(prediction.astype(np.uint8) * 255).save(masks / f"{row['id']}.png")
            if step == 1 or step % 10 == 0 or step == len(loader):
                print(f"  Batch {step}/{len(loader)} | images {len(scored)}/{len(records)}", flush=True)
    validate_scores(scored, records)
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "monai": importlib.metadata.version("monai"), "numpy": np.__version__,
                   "pillow": importlib.metadata.version("Pillow"), "scipy": importlib.metadata.version("scipy"),
                   "device": device.type, "cuda_runtime": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "visible_gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                   "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                   "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "batch_size": batch_size, "num_workers": num_workers,
                   "reproducibility": "Deterministic operations requested with warnings; bitwise portability not guaranteed"}
    document = {"identity": identity(job, protocol_hash), "environment": environment,
                "metrics": summarize_external(scored)}
    (output / "metrics.json").write_bytes(json_bytes(document))
    (output / "per_image.json").write_bytes(json_bytes(scored))
    with (output / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    artifacts = ["metrics.json", "per_image.json", "per_image.csv"]
    artifacts.extend(f"predicted_masks/{r['id']}.png" for r in records)
    complete = {"identity": identity(job, protocol_hash),
                "files": {name: sha256(output / name) for name in artifacts}}
    (output / "completion.json").write_bytes(json_bytes(complete))
    del model, checkpoint, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return read_result(output, job, records, protocol_hash)


def preflight(output, jobs, records, protocol):
    allowed = {"protocol.json", "summary.json", "summary.json.tmp", ".runner.lock"} | {j["name"] for j in jobs}
    if output.exists():
        unexpected = {p.name for p in output.iterdir()} - allowed
        if unexpected:
            raise ValueError(f"Output directory contains unrelated files: {sorted(unexpected)}")
        if any(p.name != ".runner.lock" for p in output.iterdir()) and not (output / "protocol.json").exists():
            raise ValueError("Nonempty output directory has no frozen protocol")
    if (output / "protocol.json").exists() and read_json(output / "protocol.json") != protocol:
        raise ValueError("Saved external protocol differs; preserve it and choose a new output directory")
    protocol_hash = digest(json_bytes(protocol))
    for job in jobs:
        folder = output / job["name"]
        if folder.exists() and any(folder.iterdir()):
            read_result(folder, job, records, protocol_hash)
    return protocol_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/BUSUCLM"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/processed/busuclm_external_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/busuclm_external_frozen_v2"))
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdecimal() or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Use one GPU index, a positive batch size and nonnegative workers")
    root = args.project_root.resolve()
    def locate(p):
        return p.resolve() if p.is_absolute() else (root / p).resolve()
    data_root, records, report = load_audit(locate(args.data_root), locate(args.audit_dir))
    output = locate(args.output_dir)
    if output == data_root or data_root in output.parents:
        raise ValueError("Evaluation outputs must be outside the raw dataset")
    jobs = make_jobs(root)
    protocol = make_protocol(root, report, jobs, args.batch_size, args.num_workers)
    protocol_hash = preflight(output, jobs, records, protocol)
    print("BUS-UCLM V3: 683 audited source pairs; 679 eligible images / 38 patient prefixes; four positive pairs quarantined.", flush=True)
    print("Primary overlap: 260 lesion-positive images / 35 patients. Normal burden: 419 images / 35 patients.", flush=True)
    print("Frozen primary: 384 px. Reference: 256 px. Seeds: 42, 43, 44. Threshold: 0.5. No postprocessing.", flush=True)
    print(f"Device: {args.device}; physical GPU index for CUDA: {args.gpu}", flush=True)
    if args.plan:
        for job in jobs:
            status = "REUSE" if (output / job["name"] / "completion.json").exists() else "EVALUATE"
            print(f"{status} {job['image_size']} px / seed {job['seed']} -> {output / job['name']}")
        print("Plan verified. No inference started and no files written.")
        return
    # Set visibility before the first Torch/MONAI import in this process.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected exactly one visible CUDA GPU; inspect the requested physical index")
    # Verify all source checkpoint configurations before writing the protocol or predicting.
    for job in jobs:
        checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if (config["model"] != MODEL or float(config["threshold"]) != 0.5 or
                int(config["seed"]) != job["seed"] or int(config["image_size"]) != job["image_size"] or
                int(checkpoint["epoch"]) != job["epoch"] or checkpoint.get("smoke", False) or
                checkpoint["manifest_sha256"] != TRAIN_MANIFEST_SHA256):
            raise ValueError("A frozen checkpoint has unexpected source settings")
        del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A BUS-UCLM runner is already active") from error
        protocol_hash = preflight(output, jobs, records, protocol)
        if not (output / "protocol.json").exists():
            (output / "protocol.json").write_bytes(json_bytes(protocol))
        results = []
        for index, job in enumerate(jobs, 1):
            folder = output / job["name"]
            print(f"\n[{index}/6] {job['image_size']} px / seed {job['seed']}", flush=True)
            if (folder / "completion.json").is_file():
                print("Reusing verified complete inference.", flush=True)
                result = read_result(folder, job, records, protocol_hash)
            else:
                result = evaluate_one(job, records, data_root, folder, protocol_hash,
                                      args.device, args.batch_size, args.num_workers)
            results.append(result)
            primary = result["metrics"]["cohorts"]["all_images"]
            print(f"  Lesion-positive patient Dice: {primary['lesion_positive']['patient_macro']['dice']:.6f}", flush=True)
            print(f"  Normal patient-macro foreground fraction: {primary['normal']['patient_macro_foreground_fraction']:.6f}", flush=True)
        summary = {"dataset": "BUS-UCLM", "version": 3, "protocol": protocol,
                   "protocol_sha256": protocol_hash, **aggregate_runs(results)}
        temporary = output / "summary.json.tmp"
        temporary.write_bytes(json_bytes(summary))
        temporary.replace(output / "summary.json")
    print("\nCompleted:", output / "summary.json", flush=True)
    print("The primary setting remains 384 px. No external model or parameter selection was performed.", flush=True)


if __name__ == "__main__":
    main()
