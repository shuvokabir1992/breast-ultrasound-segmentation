"""Audit the original CSVs and prepare a fixed Case-disjoint split."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breastseg.splits import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/BUSBRA"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Audit CSVs without claiming that image files were checked")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--val-fold", type=int, default=2)
    args = parser.parse_args()
    _, report = prepare(args.data_root, args.output_dir, args.metadata_dir,
                        args.metadata_only, args.test_fold, args.val_fold)
    print(json.dumps(report["baseline_protocol"], indent=2))
    check = report["dimension_check"]
    if check["status"] == "checked":
        print(f"CSV/PNG dimension discrepancies: {check['discrepancy_count']}")
        if check["discrepancy_count"]:
            print("Using verified matching image/mask PNG dimensions; source CSV unchanged.")
            for row in check["discrepancies"][:10]:
                print(f"  {row['id']}: CSV {row['metadata_width']}x{row['metadata_height']}; "
                      f"PNG pair {row['png_width']}x{row['png_height']}")
    else:
        print("Image dimensions have not been checked; manifest uses CSV dimensions.")
    print(f"Manifest and metadata audit saved in {args.output_dir}")


if __name__ == "__main__":
    main()
