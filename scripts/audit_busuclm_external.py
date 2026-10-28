"""Decode and audit the complete BUS-UCLM V3 dataset before external inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breastseg.busuclm_external import audit_dataset, json_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/BUSUCLM"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/busuclm_external_v2"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose a fresh audit output directory to preserve existing files")
    root, manifest, report = audit_dataset(args.data_root)
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Audit outputs must be outside the raw dataset")
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_bytes(json_bytes(report))
    if report["valid"]:
        (output / "manifest.json").write_bytes(json_bytes(manifest))
    print("Dataset root:", root)
    print("Metadata images:", report["metadata_images"])
    print("Decoded source pairs:", report["decoded_pairs"])
    print("Eligible pairs:", report["eligible_pairs"])
    print("Quarantined positive pairs:", report["quarantined_pairs"])
    print("Verified empty references on native image grids:", len(report["empty_reference_native_grid_ids"]))
    print("Patient prefixes:", report["patient_prefixes"])
    print("Source labels:", report["labels"])
    print("Eligible labels:", report["eligible_labels"])
    print("Native dimensions:", report["decoded_dimensions"])
    print("Warnings:", len(report["warnings"]), "Errors:", len(report["issues"]))
    for warning in report["warnings"][:10]:
        print("WARNING:", warning)
    for issue in report["issues"][:10]:
        print("ERROR:", issue)
    for row in report["quarantined_records"]:
        print("QUARANTINED:", row["id"], row["exclusion_reason"])
    print("Audit saved:", output / "audit.json")
    if not report["valid"]:
        raise SystemExit("Audit failed. Review audit.json before any inference; no valid manifest was saved.")
    print("Complete V3 source audit passed under geometry policy v2; only eligible images may be evaluated. No model predictions were run.")


if __name__ == "__main__":
    main()
