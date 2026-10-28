# Add the completed project documentation to GitHub

This package prepares documentation and derived figures. It does not create
a repository, commit changes or publish anything automatically. Run the steps
below on the machine that contains the actual project checkout.

## 1. Extract and install the documentation

Save `breast-ultrasound-github-docs.zip` in the project root, then:

```bash
cd ~/breast-ultrasound-segmentation
python -m zipfile -e breast-ultrasound-github-docs.zip .
```

`README_GITHUB.md` is the proposed root README. The package uses this temporary
filename so extraction does not replace an unseen existing `README.md`.
Read it, then run the following to install it and merge the suggested ignore
rules. A different existing README is backed up outside the repository first;
existing `.gitignore` rules are preserved.

```bash
python - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil

root = Path.cwd()
source = root / "README_GITHUB.md"
target = root / "README.md"
if target.exists() and target.read_bytes() != source.read_bytes():
    folder = root.parent / (root.name + "-documentation-backups")
    folder.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    shutil.copy2(target, folder / ("README-" + stamp + ".md"))
shutil.copyfile(source, target)

ignore = root / ".gitignore"
old = ignore.read_text() if ignore.exists() else ""
rules = (root / ".gitignore.github").read_text()
if "# Project publication excludes" not in old:
    with ignore.open("a") as handle:
        handle.write(("\n" if old and not old.endswith("\n") else "") + rules)
print("README installed; ignore rules merged.")
PY
```

The reports and figures in this package use the same project-relative paths
as the earlier results bundles. They contain completed results, not new model
inference or modified research checkpoints.

## 2. Check the repository and ignored files

For an existing Git checkout:

```bash
git status --short
git remote -v
git branch --show-current
```

If this project has not been initialized as a repository, run `git init`
first. Do not initialize a second repository inside an existing checkout.

The suggested ignore rules exclude raw/processed data, virtual environments,
`runs/`, generated `reports/`, checkpoints, logs and downloaded archives.
Curated public outputs live under `figures/`, `docs/` and the small root result
JSON files instead.

Ignore rules do not untrack files already in Git. Check the current tracked
files before publishing:

```bash
git ls-files data runs reports .venv
git ls-files '*.pt' '*.pth' '*.ckpt' '*.safetensors' '*.zip'
```

If either command lists unwanted tracked files, remove only the confirmed
paths from Git's index with `git rm --cached` (and `-r` for a directory).
This preserves working files. For example, only if `data/` was listed:

```bash
git rm -r --cached -- data/
```

That change affects the next commit; it does not remove content from older
commits or already-published history. See the official
[GitHub ignore-file guidance](https://docs.github.com/en/get-started/git-basics/ignoring-files).

## 3. Stage the documentation explicitly

The following paths are included in this documentation package:

```bash
git add README.md .gitignore \
  docs/GITHUB_PUBLISHING.md \
  docs/REPRODUCIBILITY.md \
  docs/DATASETS_AND_ATTRIBUTION.md \
  docs/BUSUCLM_EXTERNAL_PROTOCOL.md \
  docs/BUSUCLM_EXTERNAL_PROTOCOL_V1.md \
  docs/BUSUCLM_GEOMETRY_AMENDMENT.md \
  docs/busuclm_geometry_review.json \
  docs/busuclm_results_verification.json \
  docs/busuclm_external_prediction_review.json \
  docs/busuclm_prediction_review_verification.json \
  BASELINE_RESULTS.md BUSUCLM_EXTERNAL_RESULTS.md README_BUSUCLM_RESULTS.md \
  baseline_final_metrics.json busuclm_external_metrics.json \
  figures/baseline_final_results.png \
  figures/busuclm_external_results.png \
  figures/busuclm_external_predictions_review.png
```

If the project code is not already tracked, also stage the existing source,
configuration and tests explicitly after reviewing those directories:

```bash
git add breastseg/ scripts/ configs/ tests/ requirements-baseline.txt
git add report_baseline_results.py report_busuclm_external_results.py
```

The documentation package does not contain a new `requirements-lock.txt` or
source-code snapshot. The original frozen implementation must come from the
existing project. A dependency lock file may be committed after reviewing it
for local paths or private package URLs.

Inspect the complete staged list, including anything staged before these steps:

```bash
git diff --cached --name-only
git diff --cached --stat
git diff --cached -- README.md .gitignore
```

The expected documentation changes are small Markdown/JSON files and three
figures. Datasets, checkpoints and delivery ZIPs should not appear. Open the
README preview and follow its relative report/figure links.

## 4. Commit, then push to the intended repository

Once the staged changes are correct:

```bash
git commit -m "Document fixed-holdout baseline and external evaluation"
```

For an existing remote, use `git remote -v` and `git branch --show-current`
to confirm the intended repository and branch. If that branch already has an
upstream, use `git push`. Otherwise set the upstream using the actual branch
name. No repository URL is supplied here because it has not been provided.

For a new GitHub repository, create an empty repository in your own account
and use the exact remote URL and existing-repository instructions shown by
GitHub. Avoid creating an unrelated README commit on GitHub before pushing
the local project. Follow any existing branch or pull-request workflow.

## Suggested GitHub description and topics

Description:

> MONAI residual U-Net for breast ultrasound lesion segmentation, with grouped
> BUS-BRA splits, three-seed resolution comparison, and frozen BUS-UCLM external
> evaluation with failure analysis.

Suggested topics: `pytorch`, `monai`, `medical-image-segmentation`, `ultrasound`,
`unet`, `reproducibility`, `computer-vision`.

Describe the work as one fixed holdout with three seeds per resolution, not
completed ten-fold CV. Keep the external false-positive findings visible and
retain dataset attribution with the derived image figure.
