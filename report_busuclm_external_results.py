"""Audit exported BUS-UCLM patient summaries and plot the frozen experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np


SIZES = (256, 384)
SEEDS = (42, 43, 44)
OVERLAP = ("dice", "iou", "pixel_recall", "pixel_precision")
EXCLUDED = {f"HESN_{i:03d}" for i in range(2, 6)}
EMPTY_NATIVE = {f"HESN_{i:03d}" for i in (0, 1, 6, 7, 8, 9, 10, 11, 12)}
INFO_HASH = "a9d5356f0639caa1bafa5d46de58c70e81894c39ddb4feff10a7858d87d9e91c"
TRAIN_HASH = "e37a37ea057ff2e93728150bb36a67bcc7f47dcbb3228a78fb3bfe8f3d68820c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    else:
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (actual, expected)


def avg(values):
    return st.mean(values) if values else None


def stats(values):
    return {"mean": avg(values), "sample_std": st.stdev(values) if len(values) > 1 else None, "n": len(values)}


def groups(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["Image"][:4]].append(row)
    return result


def cohorts(rows):
    return {
        "all_images": rows,
        "plain_bmode": [r for r in rows if r["Doppler"] == "No" and r["Combined"] == "No"],
        "unmarked_plain_bmode": [r for r in rows if all(r[k] == "No" for k in ("Doppler", "Combined", "Marks"))],
        "doppler_yes": [r for r in rows if r["Doppler"] == "Yes"],
        "doppler_unresolved": [r for r in rows if r["Doppler"] not in ("Yes", "No")],
        "combined_yes": [r for r in rows if r["Combined"] == "Yes"],
        "marks_yes": [r for r in rows if r["Marks"] == "Yes"],
        "marks_no": [r for r in rows if r["Marks"] == "No"],
    }


def check_positive(summary, source):
    expected = groups(source)
    patients = summary["per_patient"]
    assert len(patients) == len({p["patient_id"] for p in patients})
    assert {p["patient_id"] for p in patients} == set(expected)
    assert summary["images"] == len(source) and summary["patients"] == len(expected)
    for patient in patients:
        images = expected[patient["patient_id"]]
        assert patient["images"] == len(images)
        assert patient["benign_images"] == sum(r["Label"] == "Benign" for r in images)
        assert patient["malignant_images"] == sum(r["Label"] == "Malignant" for r in images)
        for metric in OVERLAP:
            assert math.isfinite(patient[metric]) and 0 <= patient[metric] <= 1
        assert patient["iou"] <= patient["dice"] + 1e-12
    for metric in OVERLAP:
        close(summary["patient_macro"][metric], avg([p[metric] for p in patients]))
        close(summary["image_macro"][metric],
              sum(p[metric] * p["images"] for p in patients) / len(source) if source else None)
    defined = [p for p in patients if p["hd95_px"] is not None]
    hd = summary["hd95"]
    assert hd["patient_fully_defined_count"] == len(defined)
    assert hd["patient_undefined_count"] == len(patients) - len(defined)
    assert hd["image_defined_count"] + hd["image_undefined_count"] == len(source)
    for p in defined:
        assert math.isfinite(p["hd95_px"]) and p["hd95_px"] >= 0
    close(hd["patient_mean_fully_defined_px"], avg([p["hd95_px"] for p in defined]))
    # These exports contain no undefined positive distances, so the image mean
    # is recoverable from patient means and each patient's eligible image count.
    assert hd["image_undefined_count"] == 0 and len(defined) == len(patients)
    assert summary["empty_prediction_images"] == 0
    close(hd["image_mean_defined_px"],
          sum(p["hd95_px"] * p["images"] for p in patients) / len(source) if source else None)
    if source and "patient_dice_bootstrap_95ci" in summary:
        assert summary["bootstrap_resamples"] == 2000 and summary["bootstrap_seed"] == 42
        values = np.array([p["dice"] for p in patients])
        draws = np.random.default_rng(42).choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        interval = np.percentile(draws, [2.5, 97.5])
        for actual, expected in zip(summary["patient_dice_bootstrap_95ci"], interval):
            close(actual, float(expected))
    return {p["patient_id"]: p for p in patients}


def check_normal(summary, source):
    expected = groups(source)
    patients = summary["per_patient"]
    assert len(patients) == len({p["patient_id"] for p in patients})
    assert {p["patient_id"] for p in patients} == set(expected)
    assert summary["images"] == len(source) and summary["patients"] == len(expected)
    for patient in patients:
        assert patient["images"] == len(expected[patient["patient_id"]])
        assert 0 <= patient["foreground_fraction"] <= 1
        rate = patient["fraction_images_with_any_foreground"]
        assert 0 <= rate <= 1
        close(rate * patient["images"], round(rate * patient["images"]))
        assert patient["any_foreground_on_at_least_one_normal_image"] == (rate > 0)
    close(summary["patient_macro_foreground_fraction"], avg([p["foreground_fraction"] for p in patients]))
    close(summary["patient_macro_fraction_images_with_any_foreground"], avg([p["fraction_images_with_any_foreground"] for p in patients]))
    for key, column in (("image_mean_foreground_fraction", "foreground_fraction"),
                        ("image_fraction_with_any_foreground", "fraction_images_with_any_foreground")):
        close(summary[key], sum(p[column] * p["images"] for p in patients) / len(source) if source else None)
    assert summary["patients_with_any_foreground_on_normal_images"] == sum(p["any_foreground_on_at_least_one_normal_image"] for p in patients)
    return {p["patient_id"]: p for p in patients}


def audit(audit_doc, external, baseline, info_path, implementation_root):
    assert audit_doc["valid"] and not audit_doc["issues"] and audit_doc["full_frozen_v3"]
    assert audit_doc["schema_version"] == 2 and audit_doc["decoded_pairs"] == 683
    assert audit_doc["eligible_pairs"] == 679 and audit_doc["quarantined_pairs"] == 4
    assert {r["id"] for r in audit_doc["quarantined_records"]} == EXCLUDED
    assert set(audit_doc["empty_reference_native_grid_ids"]) == EMPTY_NATIVE
    assert sha(info_path) == audit_doc["metadata_sha256"] == INFO_HASH
    with Path(info_path).open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle, delimiter=";"))
    eligible = [r for r in source if Path(r["Image"]).stem not in EXCLUDED]
    assert len(source) == 683 and len(eligible) == 679
    assert len({r["Image"] for r in source}) == 683
    p = external["protocol"]
    assert p["protocol_version"] == 2 and external["dataset"] == "BUS-UCLM" and external["version"] == 3
    assert hashlib.sha256(encoded(p)).hexdigest() == external["protocol_sha256"]
    assert external["selected_resolution"] == p["primary_resolution"] == 384
    assert external["reference_resolution"] == p["reference_resolution"] == 256
    assert p["threshold"] == 0.5 and p["postprocessing"] == "none"
    assert p["eligible_images"] == 679 and p["source_images"] == 683
    assert p["metadata_sha256"] == INFO_HASH
    assert p["training_manifest_sha256"] == TRAIN_HASH
    assert p["external_manifest_sha256"] == audit_doc["manifest_sha256"]
    assert p["geometry_policy_sha256"] == audit_doc["geometry_policy_sha256"]
    assert p["quarantined_records"] == audit_doc["quarantined_records"]
    assert set(p["empty_reference_native_grid_ids"]) == EMPTY_NATIVE
    for name, expected in p["implementation_sha256"].items():
        assert sha(implementation_root / name) == expected, f"Source implementation changed: {name}"
    assert baseline["split"] == "test" and baseline["selected_resolution"] == 384
    assert baseline["protocol"]["manifest_sha256"] == TRAIN_HASH
    old = {(r["image_size"], r["seed"]): r for r in baseline["runs"]}
    runs = {(r["image_size"], r["seed"]): r for r in external["runs"]}
    frozen = {(r["image_size"], r["seed"]): r for r in p["checkpoints"]}
    assert len(external["runs"]) == len(baseline["runs"]) == len(p["checkpoints"]) == 6
    assert set(old) == set(runs) == set(frozen) == {(s, seed) for s in SIZES for seed in SEEDS}
    expected_cohorts = cohorts(eligible)
    for key, row in runs.items():
        assert row["checkpoint_sha256"] == frozen[key]["checkpoint_sha256"] == old[key]["checkpoint_sha256"]
        assert row["checkpoint_epoch"] == frozen[key]["epoch"] == old[key]["checkpoint_epoch"]
        assert row["role"] == ("primary" if key[0] == 384 else "reference")
        mm = row["metrics"]
        assert mm["images"] == 679 and mm["patients"] == 38
        assert set(mm["cohorts"]) == set(expected_cohorts)
        for name, selected in expected_cohorts.items():
            report = mm["cohorts"][name]
            assert report["images"] == len(selected) and report["patients"] == len(groups(selected))
            check_positive(report["lesion_positive"], [r for r in selected if r["Label"] != "Normal"])
            check_normal(report["normal"], [r for r in selected if r["Label"] == "Normal"])
        for label in ("benign", "malignant"):
            check_positive(mm["by_lesion_label"][label], [r for r in eligible if r["Label"].lower() == label])
        # Reconstruct each patient's all-positive mean from the two label groups.
        full = {r["patient_id"]: r for r in mm["cohorts"]["all_images"]["lesion_positive"]["per_patient"]}
        pieces = defaultdict(list)
        for label in ("benign", "malignant"):
            for patient in mm["by_lesion_label"][label]["per_patient"]:
                pieces[patient["patient_id"]].append(patient)
        for patient, parts in pieces.items():
            assert sum(r["images"] for r in parts) == full[patient]["images"]
            for metric in OVERLAP:
                close(full[patient][metric], sum(r[metric] * r["images"] for r in parts) / full[patient]["images"])
        # Marks=Yes/No partition the full cohort, while other subsets overlap.
        for kind, measures in (("lesion_positive", OVERLAP),
                               ("normal", ("foreground_fraction", "fraction_images_with_any_foreground"))):
            full = {r["patient_id"]: r for r in mm["cohorts"]["all_images"][kind]["per_patient"]}
            pieces = defaultdict(list)
            for name in ("marks_yes", "marks_no"):
                for patient in mm["cohorts"][name][kind]["per_patient"]:
                    pieces[patient["patient_id"]].append(patient)
            for patient, parts in pieces.items():
                assert sum(r["images"] for r in parts) == full[patient]["images"]
                for metric in measures:
                    close(full[patient][metric], sum(r[metric] * r["images"] for r in parts) / full[patient]["images"])
    for size in SIZES:
        values = [old[size, seed]["metrics"]["case_macro"]["dice"] for seed in SEEDS]
        for seed in SEEDS:
            per_case = old[size, seed]["per_case"]
            assert len(per_case) == 106
            close(old[size, seed]["metrics"]["case_macro"]["dice"], st.mean(r["dice"] for r in per_case))
        close(st.mean(values), baseline["by_resolution"][str(size)]["dice"]["mean"])
        close(st.stdev(values), baseline["by_resolution"][str(size)]["dice"]["sample_std"])
    return runs, old, eligible


def derive(runs, old, external):
    def all_group(row, name):
        return row["metrics"]["cohorts"]["all_images"][name]
    by_size, rows = {}, []
    for size in SIZES:
        selected = [runs[size, seed] for seed in SEEDS]
        metrics = {k: stats([all_group(r, "lesion_positive")["patient_macro"][k] for r in selected]) for k in OVERLAP}
        metrics["hd95_native_px"] = stats([all_group(r, "lesion_positive")["hd95"]["patient_mean_fully_defined_px"] for r in selected])
        metrics["normal_foreground_fraction"] = stats([all_group(r, "normal")["patient_macro_foreground_fraction"] for r in selected])
        metrics["normal_any_foreground_image_rate"] = stats([all_group(r, "normal")["patient_macro_fraction_images_with_any_foreground"] for r in selected])
        for label in ("benign", "malignant"):
            metrics[label + "_dice"] = stats([r["metrics"]["by_lesion_label"][label]["patient_macro"]["dice"] for r in selected])
        baseline_dice = stats([old[size, seed]["metrics"]["case_macro"]["dice"] for seed in SEEDS])
        by_size[str(size)] = {"role": "primary" if size == 384 else "reference", "external": metrics,
                             "baseline_test_dice": baseline_dice,
                             "external_minus_baseline_dice": metrics["dice"]["mean"] - baseline_dice["mean"]}
        source_names = {"dice": "lesion_positive_patient_dice", "iou": "lesion_positive_patient_iou",
                        "pixel_recall": "lesion_positive_patient_pixel_recall", "pixel_precision": "lesion_positive_patient_pixel_precision",
                        "hd95_native_px": "lesion_positive_patient_hd95_defined_px",
                        "normal_foreground_fraction": "normal_patient_macro_foreground_fraction",
                        "normal_any_foreground_image_rate": "normal_patient_macro_fraction_images_with_any_foreground"}
        for name, source in source_names.items():
            original = external["by_resolution"][str(size)]["metrics"][source]
            close(metrics[name]["mean"], original["mean"])
            close(metrics[name]["sample_std"], original["sample_std"])
            assert original["defined_seed_count"] == 3 and original["undefined_seed_count"] == 0
        for row in selected:
            rows.append({"image_size": size, "seed": row["seed"], "role": row["role"],
                         **{k: all_group(row, "lesion_positive")["patient_macro"][k] for k in OVERLAP},
                         "normal_foreground_fraction": all_group(row, "normal")["patient_macro_foreground_fraction"]})
    patient_values, normal_values = {}, {}
    for size in SIZES:
        pp = defaultdict(list)
        nn = defaultdict(list)
        for seed in SEEDS:
            for p in all_group(runs[size, seed], "lesion_positive")["per_patient"]:
                pp[p["patient_id"]].append(p["dice"])
            for p in all_group(runs[size, seed], "normal")["per_patient"]:
                nn[p["patient_id"]].append(p["foreground_fraction"])
        patient_values[size] = {k: st.mean(v) for k, v in sorted(pp.items())}
        normal_values[size] = {k: st.mean(v) for k, v in sorted(nn.items())}
    assert set(patient_values[256]) == set(patient_values[384]) and len(patient_values[384]) == 35
    delta = np.array([patient_values[384][k] - patient_values[256][k] for k in patient_values[384]])
    draws = np.random.default_rng(42).choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
    comparison = {"difference_384_minus_256": float(delta.mean()),
                  "paired_patient_bootstrap_95ci": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
                  "patients": len(delta), "resamples": 2000, "seed": 42,
                  "patients_higher_at_384": int((delta > 0).sum()), "patients_lower_at_384": int((delta < 0).sum()),
                  "scope": "Exploratory patient resampling after averaging Dice across three fixed seed pairs; excludes training-seed and model-selection uncertainty"}
    secondary = {}
    for name in ("plain_bmode", "unmarked_plain_bmode", "doppler_yes", "combined_yes", "marks_yes", "marks_no"):
        secondary[name] = {}
        for size in SIZES:
            reports = [runs[size, seed]["metrics"]["cohorts"][name] for seed in SEEDS]
            secondary[name][str(size)] = {"positive_images": reports[0]["lesion_positive"]["images"],
                                          "positive_patients": reports[0]["lesion_positive"]["patients"],
                                          "dice": stats([r["lesion_positive"]["patient_macro"]["dice"] for r in reports])}
    return {"primary_resolution": 384, "reference_resolution": 256, "by_resolution": by_size,
            "per_seed": rows, "exploratory_paired_patient_comparison": comparison, "secondary": secondary,
            "primary_positive_patient_ranking": sorted([{"patient_id": k, "mean_dice": v} for k, v in patient_values[384].items()], key=lambda r: r["mean_dice"]),
            "primary_normal_patient_ranking": sorted([{"patient_id": k, "mean_foreground_fraction": v} for k, v in normal_values[384].items()], key=lambda r: -r["mean_foreground_fraction"])}


def plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colours = {256: "#0072b2", 384: "#d55e00"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
    for x, cohort in enumerate(("baseline", "external")):
        for size, offset in ((256, -0.13), (384, 0.13)):
            summary = report["by_resolution"][str(size)]
            value = summary["baseline_test_dice"] if cohort == "baseline" else summary["external"]["dice"]
            if cohort == "external":
                points = [r["dice"] for r in report["per_seed"] if r["image_size"] == size]
                for jitter, point in zip((-0.035, 0, 0.035), points):
                    axes[0].scatter(x + offset + jitter, point, color=colours[size], s=25, alpha=0.65)
            axes[0].errorbar(x + offset, value["mean"], yerr=value["sample_std"], fmt="s", color=colours[size], capsize=5, markersize=6)
    axes[0].set(xticks=(0, 1), xticklabels=("BUS-BRA test\n106 Cases", "BUS-UCLM positive\n35 eligible patients"),
                ylabel="Case/patient-macro Dice", ylim=(0, 1), xlim=(-0.45, 1.45), title="Lower performance on the external cohort")
    for x, size in enumerate(SIZES):
        value = report["by_resolution"][str(size)]["external"]["normal_foreground_fraction"]
        points = [100 * r["normal_foreground_fraction"] for r in report["per_seed"] if r["image_size"] == size]
        for jitter, point in zip((-0.06, 0, 0.06), points):
            axes[1].scatter(x + jitter, point, color=colours[size], s=30, alpha=0.65)
        axes[1].errorbar(x, 100 * value["mean"], yerr=100 * value["sample_std"], fmt="s", color=colours[size], capsize=5, markersize=6)
    axes[1].set(xticks=(0, 1), xticklabels=("256 px\nreference", "384 px\nselected primary"),
                ylabel="Patient-macro foreground area on normal images (%)", ylim=(0, 25), xlim=(-0.45, 1.45),
                title="Foreground predicted on all normal images")
    axes[1].text(0.5, 23.3, "419 normal images / 35 patients\nAny foreground: 100% of images in every run", ha="center", va="top", fontsize=9, color="#555555")
    for ax in axes:
        ax.grid(axis="y", color="#e4e7ed", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(handles=[Line2D([0], [0], marker="s", color=colours[s], label=f"{s} px", markersize=5) for s in SIZES],
                   loc="lower left", frameon=False)
    fig.suptitle("Frozen BUS-BRA U-Net models: external evaluation on BUS-UCLM V3", fontsize=14, y=0.995)
    fig.text(0.5, 0.93, "Same six checkpoints | seeds 42, 43, 44 | threshold 0.5 | no postprocessing | four positive-reference geometry exclusions", ha="center", fontsize=9, color="#555555")
    fig.text(0.5, 0.02, "Error bars are sample SD across three training seeds, not confidence intervals. Cohorts differ. Normal-image burden is a segmentation measure.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9), w_pad=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--baseline-test-summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--implementation-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a fresh report output directory")
    documents = [json.loads(p.read_text()) for p in (args.audit, args.external_summary, args.baseline_test_summary)]
    runs, old, eligible = audit(*documents, args.metadata, args.implementation_root)
    report = derive(runs, old, documents[1])
    report.update(protocol=documents[1]["protocol"],
                  verification={"export_consistency_checks": "passed", "source_images_audited_on_aries": 683,
                                "eligible_images": 679, "positive_images": 260, "positive_patients": 35,
                                "normal_images": 419, "normal_patients": 35, "quarantined_positive_pairs": 4,
                                "raw_prediction_masks_or_full_per_image_exports_received_for_this_review": False,
                                "limitation": "Source identities, metadata counts, patient-level arithmetic, subgroup partitions and bootstrap exports checked; raw pixel predictions have not been independently rescored here"},
                  input_sha256={"audit": sha(args.audit), "external_summary": sha(args.external_summary),
                                "baseline_test_summary": sha(args.baseline_test_summary), "metadata": sha(args.metadata)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "busuclm_external_metrics.json").write_bytes(encoded(report))
    plot(report, args.output_dir / "figures/busuclm_external_results.png")
    print("Export identities and patient/subgroup arithmetic verified.")
    print(json.dumps({s: report["by_resolution"][s]["external"]["dice"] for s in ("256", "384")}, indent=2))
    print("Saved report data and figure:", args.output_dir)


if __name__ == "__main__":
    main()
