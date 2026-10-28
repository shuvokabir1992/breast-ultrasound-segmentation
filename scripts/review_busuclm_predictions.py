"""Inspect saved external predictions without training or model inference."""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from breastseg.busuclm_external import (
    decode_pair, digest, json_bytes, load_audit, read_json, score_prediction,
)
from evaluate_busuclm_external import make_jobs, read_result


SEEDS = (42, 43, 44)


def choose_cases(exports):
    maps = {seed: {r["id"]: r for r in exports[seed]} for seed in SEEDS}
    ids = sorted(maps[42])
    if any(set(m) != set(ids) or len(m) != len(exports[seed]) for seed, m in maps.items()):
        raise ValueError("Seed export IDs differ or contain duplicates")
    cases = []
    positive = defaultdict(list)
    for image_id in ids:
        row = maps[42][image_id]
        if any(maps[s][image_id]["patient_id"] != row["patient_id"] or maps[s][image_id]["label"] != row["label"] for s in SEEDS):
            raise ValueError("Seed export metadata differ")
        if row["label"] != "normal":
            value = sum(maps[s][image_id]["dice"] for s in SEEDS) / 3
            positive[row["patient_id"]].append((value, image_id))
    ordered = sorted(positive, key=lambda p: (sum(v for v, _ in positive[p]) / len(positive[p]), p))
    requested = []
    if ordered:
        requested = [(ordered[0], "Low Dice patient"),
                     (ordered[min(1, len(ordered) - 1)], "Second low Dice patient"),
                     (ordered[len(ordered) // 2], "Middle Dice patient"),
                     (ordered[-1], "High Dice patient")]
    used_patients = set()
    for patient, reason in requested:
        if patient in used_patients:
            continue
        used_patients.add(patient)
        views = sorted(positive[patient])
        if reason in ("Low Dice patient", "Second low Dice patient"):
            _, image_id = views[0]
        else:
            centre = sum(v for v, _ in views) / len(views)
            _, image_id = min(views, key=lambda pair: (abs(pair[0] - centre), pair[1]))
        cases.append({"id": image_id, "reason": reason})
    normal = sorted((sum(maps[s][i]["foreground_fraction"] for s in SEEDS) / 3, i)
                    for i in ids if maps[42][i]["label"] == "normal" and maps[42][i]["patient_id"] != "HESN")
    selected_normal_patients = set()
    if normal:
        positions = [(len(normal) - 1, "Normal: large foreground area"),
                     (len(normal) // 2, "Normal: middle foreground area"),
                     (0, "Normal: small foreground area")]
        used_ids = {c["id"] for c in cases}
        for position, reason in positions:
            candidates = sorted(range(len(normal)), key=lambda i: (abs(i - position), normal[i][1]))
            for index in candidates:
                _, image_id = normal[index]
                patient = maps[42][image_id]["patient_id"]
                if image_id not in used_ids and patient not in selected_normal_patients:
                    cases.append({"id": image_id, "reason": reason})
                    used_ids.add(image_id)
                    selected_normal_patients.add(patient)
                    break
    hesn = [i for i in ids if maps[42][i]["label"] == "normal" and maps[42][i]["patient_id"] == "HESN"]
    if hesn:
        cases.append({"id": hesn[0], "reason": "Normal: reviewed HESN native grid"})
    return cases


def checked_prediction(path, shape):
    with Image.open(path) as source:
        source.load()
        if source.format != "PNG" or source.mode != "L":
            raise ValueError("Expected original saved grayscale binary prediction PNG")
        if source.size != (shape[1], shape[0]):
            raise ValueError("Saved prediction has incorrect native dimensions")
        array = np.asarray(source)
        if not set(np.unique(array).tolist()).issubset({0, 255}):
            raise ValueError("Saved prediction is not binary 0/255")
        return array > 0


def compare_scores(actual, saved):
    for key, value in actual.items():
        if type(value) is float:
            if saved[key] is None or not math.isclose(value, saved[key], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Recomputed pixel score differs: {key}")
        elif value != saved[key]:
            raise ValueError(f"Recomputed score or metadata differs: {key}")


def render(items, output, title="BUS-UCLM external prediction review: frozen 384-pixel models",
           attribution="Source: BUS-UCLM V3, Vallez et al. (2025), DOI 10.17632/7fvgj4jsp7.3, CC BY 4.0. Grayscale views and model overlays added."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(items), 4, figsize=(15.5, max(5, 2.7 * len(items))), squeeze=False)
    for index, item in enumerate(items):
        for column, ax in enumerate(axes[index]):
            ax.imshow(item["image"], cmap="gray", vmin=0, vmax=255)
            if item["truth"].any():
                ax.contour(item["truth"], levels=[0.5], colors=["#00d7e9"], linewidths=0.8)
            if column:
                seed = SEEDS[column - 1]
                prediction = item["predictions"][seed]
                if prediction.any():
                    overlay = np.zeros((*prediction.shape, 4), dtype=float)
                    overlay[prediction] = (1.0, 0.45, 0.05, 0.22)
                    ax.imshow(overlay)
                    if not prediction.all():
                        ax.contour(prediction, levels=[0.5], colors=["#ff730d"], linewidths=0.65)
                score = item["scores"][seed]
                value = f"FG area {100 * score['foreground_fraction']:.1f}%" if item["label"] == "normal" else f"Dice {score['dice']:.3f}"
                ax.set_title(f"Seed {seed} | {value}", fontsize=9)
            else:
                ax.set_title(f"{item['id']} | {item['label']}\n{item['reason']}", fontsize=9)
            ax.axis("off")
    fig.suptitle(title, fontsize=14, y=0.997)
    fig.text(0.5, 0.978, "Cyan: reference lesion | orange: saved prediction | threshold 0.5 | no filtering | normal references are empty", ha="center", fontsize=10)
    fig.text(0.5, 0.012, attribution, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.028, 1, 0.965), h_pad=1.1, w_pad=0.5)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/BUSUCLM"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/processed/busuclm_external_v2"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("reports/busuclm_external_frozen_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/busuclm_external_review"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    def locate(p):
        return p.resolve() if p.is_absolute() else (project / p).resolve()
    output = locate(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh review output directory")
    print("Verifying audited source files...", flush=True)
    root, records, audit = load_audit(locate(args.data_root), locate(args.audit_dir))
    evaluation = locate(args.evaluation_dir)
    summary = read_json(evaluation / "summary.json")
    protocol = read_json(evaluation / "protocol.json")
    protocol_hash = digest(json_bytes(protocol))
    if (summary["protocol"] != protocol or summary["protocol_sha256"] != protocol_hash or
            protocol["protocol_version"] != 2 or protocol["primary_resolution"] != 384 or
            protocol["external_manifest_sha256"] != audit["manifest_sha256"]):
        raise ValueError("Saved evaluation protocol or dataset identity differs")
    if output == root or root in output.parents or output == evaluation or evaluation in output.parents:
        raise ValueError("Review output must be separate from raw data and evaluation artifacts")
    for name, expected in protocol["implementation_sha256"].items():
        from breastseg.busuclm_external import sha256
        if sha256(project / name) != expected:
            raise ValueError(f"Source implementation changed: {name}")
    jobs = {(r["image_size"], r["seed"]): r for r in make_jobs(project)}
    exports = {}
    for seed in SEEDS:
        print(f"Checking saved 384-pixel evaluation, seed {seed}...", flush=True)
        job = jobs[384, seed]
        checked = read_result(evaluation / job["name"], job, records, protocol_hash)
        saved = next(r for r in summary["runs"] if r["image_size"] == 384 and r["seed"] == seed)
        if checked != saved:
            raise ValueError("Combined summary differs from saved individual evaluation")
        exports[seed] = read_json(evaluation / job["name"] / "per_image.json")
    cases = choose_cases(exports)
    maps = {seed: {r["id"]: r for r in rows} for seed, rows in exports.items()}
    records_by_id = {r["id"]: r for r in records}
    items, review_rows = [], []
    for case in cases:
        print(f"Rescoring saved masks: {case['id']}...", flush=True)
        row = records_by_id[case["id"]]
        image, truth, _ = decode_pair(root, row, verify_hashes=True)
        predictions, scores = {}, {}
        for seed in SEEDS:
            path = evaluation / jobs[384, seed]["name"] / "predicted_masks" / f"{row['id']}.png"
            prediction = checked_prediction(path, truth.shape)
            score = score_prediction(row, prediction, truth)
            compare_scores(score, maps[seed][row["id"]])
            predictions[seed], scores[seed] = prediction, score
        items.append({**case, "label": row["label"], "image": np.asarray(image.convert("L")),
                      "truth": truth, "predictions": predictions, "scores": scores})
        review_rows.append({**case, "patient_id": row["patient_id"], "label": row["label"],
                            "doppler_raw": row["doppler_raw"], "marks_raw": row["marks_raw"],
                            "combined_raw": row["combined_raw"], "reference_policy": row["reference_policy"],
                            "width": row["width"], "height": row["height"], "per_seed": scores})
    output.mkdir(parents=True, exist_ok=True)
    render(items, output / "predictions_review.png")
    report = {"protocol_sha256": protocol_hash, "resolution": 384, "seeds": list(SEEDS),
              "inference_performed": False, "models_changed": False,
              "pixel_scores_recomputed_for_selected_images": True, "selected_images": review_rows,
              "selection": "Four patients spanning low/middle/high mean Dice; normal images spanning mean foreground burden plus one HESN native-grid example. These are diagnostic examples, not a representative random sample.",
              "use": "Verify saved-mask alignment, scoring and failure locations. No threshold, postprocessing or model choice is selected using these external examples."}
    (output / "review_summary.json").write_bytes(json_bytes(report))
    print("Saved:", output / "predictions_review.png")
    print("Saved:", output / "review_summary.json")
    print("Selected saved masks were rescored; no model inference or training was run.")


if __name__ == "__main__":
    main()
