# FLARE 2026 AutoMSC pipeline

## Quick Dataset008 test

From the `codebase/` directory, run:

```bash
chmod +x test_dataset008.sh
./test_dataset008.sh
```

This checks Dataset008, the command interface, automatic configuration, and
the unit tests without training. The default dataset location is
`/home/rc411/Data/renjure/AutoMSC2026Data/Dataset008_EGCT`. To use another
location:

```bash
./test_dataset008.sh check /absolute/path/to/Dataset008_EGCT
```

For a one-epoch segmentation, classification, and inference smoke test:

```bash
./test_dataset008.sh train /absolute/path/to/Dataset008_EGCT
```

## 1. Installation with uv

Run from the `codebase/` directory. Python 3.12 and a CUDA GPU are
recommended. Set `RAW_ROOT` to the local folder containing the datasets. The
other outputs default to writable folders under the current directory, but may
also be changed.

The lockfile selects CUDA 12.8 on x86-64 Linux and CUDA 12.9 on ARM64 Linux.

```bash
export RAW_ROOT=/absolute/path/to/folder/containing/datasets
export WORK_ROOT="$PWD/automsc_work"
export SUBMISSION_ROOT="$PWD/automsc_submission"
export UV_PROJECT_ENVIRONMENT="$WORK_ROOT/uv-environment"
export MPLBACKEND=Agg

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --python 3.12
uv run automsc --help
mkdir -p "$SUBMISSION_ROOT/predictions"
```
------------------------------------------
Notes:
`RAW_ROOT` must contain the dataset folders. 
`WORK_ROOT` stores preprocessing, feature caches, manifests, and trained checkpoints. 
Predictions are written below `SUBMISSION_ROOT/predictions`.
------------------------------------------

## 2. Reproduce Dataset008_EGCT and Dataset009_PHLF

The commands below perform segmentation training, full-data classification
training, and inference. The default segmentation fold is `all`.

```bash
uv run automsc train_seg Dataset008_EGCT \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda

uv run automsc train_cls Dataset008_EGCT \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda \
  --epochs 250 --rebuild-cache

uv run automsc infer Dataset008_EGCT \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda \
  --output-root "$SUBMISSION_ROOT/predictions"

uv run automsc train_seg Dataset009_PHLF \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda

uv run automsc train_cls Dataset009_PHLF \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda \
  --epochs 250 --rebuild-cache

uv run automsc infer Dataset009_PHLF \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda \
  --output-root "$SUBMISSION_ROOT/predictions"
```
------------------------------------------
The resulting submission folders are:

```text
automsc_submission/predictions/
├── Dataset008_EGCT_prediction/
│   ├── predictions.csv
│   └── *.nii.gz
└── Dataset009_PHLF_prediction/
    ├── predictions.csv
    └── *.nii.gz
```
------------------------------------------

## 3. Apply to a new dataset

Place the new dataset directly under `RAW_ROOT` using this structure:

```text
RAW_ROOT/
└── DatasetXXX_NAME/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    ├── cls_data.csv
    └── dataset.json
```

`dataset.json` must follow the nnU-Net format. `cls_data.csv` must contain one
case-identifier column and one or more classification-label columns.

Run the same three commands:

```bash
uv run automsc train_seg DatasetXXX_NAME \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda

uv run automsc train_cls DatasetXXX_NAME \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda \
  --epochs 250 --rebuild-cache

uv run automsc infer DatasetXXX_NAME \
  --raw-root "$RAW_ROOT" --work-root "$WORK_ROOT" --device cuda \
  --output-root "$SUBMISSION_ROOT/predictions"
```

------------------------------------------

Notes:

Segmentation training is configured for 2,500 epochs but feel free to early stop. 
If it is interrupted, resume it by adding `--continue` to the corresponding `train_seg` command.
`train_cls` requires the segmentation `checkpoint_best.pth`, which is selected automatically when `train_seg` completes.
- `Demo.ipynb` is included as an equivalent Google Colab example.

-------------------------------------------

### Manual early stopping

Segmentation training is configured for 2,500 epochs. For exact reproduction, allow training to complete. For an early-stopped test run, press `Ctrl+C` only after `checkpoint_best.pth` has been saved.

Because manual interruption occurs before `train_seg` writes the segmentation manifest, create it manually before running `train_cls`:

```bash
export DATASET = Dataset008_EGCT
export MODEL_NAME=nnUNetTrainerQuickSeg__nnUNetResEncUNetMPlans__3d_fullres

python3 - <<'PY'
import json
import os
from pathlib import Path

work_root = Path(os.environ["WORK_ROOT"])
dataset = os.environ["DATASET"]
model_name = os.environ["MODEL_NAME"]

checkpoint = (
    work_root
    / "nnUNet_results"
    / dataset
    / model_name
    / "fold_all"
    / "checkpoint_best.pth"
)
assert checkpoint.is_file(), f"Checkpoint not found: {checkpoint}"

manifest = {
    "version": 1,
    "dataset_name": dataset,
    "configuration": "3d_fullres",
    "planner": "nnUNetPlannerResEncM",
    "plans_identifier": "nnUNetResEncUNetMPlans",
    "trainer": "nnUNetTrainerQuickSeg",
    "fold": "all",
    "model_name": model_name,
    "checkpoint_name": "checkpoint_best.pth",
    "checkpoint_path": str(checkpoint),
}

manifest_path = (
    work_root
    / "manifests"
    / f"{dataset}.segmentation.json"
)
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Created: {manifest_path}")
PY
```

Change `DATASET` when applying the procedure to another dataset. For example:

```bash
export DATASET=Dataset009_PHLF
```
