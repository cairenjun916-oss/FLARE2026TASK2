from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DATASET_PATTERN = re.compile(r"^Dataset(?P<id>\d{3})_.+$")


@dataclass(frozen=True)
class DatasetContext:
    name: str
    dataset_id: int
    raw_root: Path
    folder: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label_column: str
    class_values: tuple[Any, ...]
    output_columns: tuple[str, ...]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(requested)


def _resolve_dataset(dataset: str, raw_root: str | Path | None) -> DatasetContext:
    supplied = Path(dataset).expanduser()
    if supplied.is_dir():
        folder = supplied.resolve()
        root = folder.parent
        name = folder.name
        if raw_root is not None and Path(raw_root).expanduser().resolve() != root:
            raise ValueError(
                f"Dataset path {folder} is not directly under raw root "
                f"{Path(raw_root).expanduser().resolve()}."
            )
    else:
        name = dataset
        raw_value = raw_root or os.environ.get("nnUNet_raw")
        if raw_value is None:
            raise ValueError(
                "Provide --raw-root, or set nnUNet_raw, when the dataset is "
                "given by name."
            )
        root = Path(raw_value).expanduser().resolve()
        folder = root / name

    match = DATASET_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Dataset must use nnU-Net naming (DatasetXXX_NAME); received {name!r}."
        )
    metadata_path = folder / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in ("channel_names", "labels", "file_ending"):
        if key not in metadata:
            raise KeyError(f"{metadata_path} is missing required key {key!r}.")
    return DatasetContext(
        name=name,
        dataset_id=int(match.group("id")),
        raw_root=root,
        folder=folder,
        metadata=metadata,
    )


def _configure_environment(context: DatasetContext, work_root: Path) -> None:
    os.environ.update(
        {
            "nnUNet_raw": str(context.raw_root),
            "nnUNet_preprocessed": str(work_root / "nnUNet_preprocessed"),
            "nnUNet_results": str(work_root / "nnUNet_results"),
            "nnUNet_compile": "false",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            # Colab exports its notebook-only matplotlib_inline backend. The
            # isolated uv environment may not contain that IPython backend,
            # and the training pipeline does not need an interactive display.
            "MPLBACKEND": "Agg",
        }
    )


def _classification_csv(context: DatasetContext) -> Path:
    filename = context.metadata.get("classification_labels_file", "cls_data.csv")
    path = context.folder / str(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Classification labels must be in the raw dataset: {path}"
        )
    return path


def _identifier_column(columns: Sequence[str]) -> str:
    for candidate in ("case_id", "identifier", "case", "id"):
        if candidate in columns:
            return candidate
    if not columns:
        raise ValueError("The classification CSV has no columns.")
    return str(columns[0])


def _sort_class_values(values: Sequence[Any]) -> tuple[Any, ...]:
    unique = list(dict.fromkeys(values))
    try:
        return tuple(sorted(unique))
    except TypeError:
        return tuple(sorted(unique, key=lambda value: str(value)))


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    if not cleaned:
        raise ValueError(f"Cannot form a safe task name from {value!r}.")
    return cleaned


def _discover_tasks(context: DatasetContext, requested: Sequence[str] | None):
    import pandas as pd

    csv_path = _classification_csv(context)
    frame = pd.read_csv(csv_path)
    identifier = _identifier_column(frame.columns.tolist())
    label_columns = list(requested or [c for c in frame.columns if c != identifier])
    if not label_columns:
        raise ValueError(f"No classification label columns were found in {csv_path}.")
    missing = [column for column in label_columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Unknown label columns {missing}; available columns: "
            f"{frame.columns.tolist()}"
        )

    declared = context.metadata.get("classification_labels", {})
    declared_names = list(declared) if isinstance(declared, dict) else []
    tasks: list[TaskSpec] = []
    multiple = len(label_columns) > 1
    for index, column in enumerate(label_columns):
        series = frame[column].dropna()
        values = _sort_class_values(series.unique().tolist())
        if len(values) < 2:
            raise ValueError(f"Classification column {column!r} has fewer than 2 classes.")
        if column == "label" and len(label_columns) == 1 and len(declared_names) == 1:
            task_name = str(declared_names[0])
        else:
            task_name = str(column)
        if multiple:
            prefix = _safe_name(task_name).lower()
            output = (
                (f"{prefix}_label",)
                if len(values) == 2
                else tuple(f"{prefix}_label_{i}" for i in range(len(values)))
            )
        else:
            output = (
                ("label",)
                if len(values) == 2
                else tuple(f"label_{i}" for i in range(len(values)))
            )
        tasks.append(TaskSpec(task_name, str(column), values, output))
    return csv_path, frame, identifier, tasks


def _preprocessed_identifiers(preprocessed_folder: Path) -> set[str]:
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    dataset_class = infer_dataset_class(str(preprocessed_folder))
    return set(map(str, dataset_class.get_identifiers(str(preprocessed_folder))))


def _write_all_case_classification_split(
    frame,
    identifier_column: str,
    task: TaskSpec,
    preprocessed_folder: Path,
    output_path: Path,
) -> tuple[Path, dict[str, list[str]]]:
    """Write the one permitted full-data split.

    ``val`` intentionally repeats ``train`` because downstream nnU-Net data
    helpers require both keys. It is used only for descriptive center-crop
    monitoring; it is not a held-out validation set and never selects an epoch.
    """
    labelled = frame[[identifier_column, task.label_column]].dropna().copy()
    labelled[identifier_column] = labelled[identifier_column].astype(str)
    if labelled[identifier_column].duplicated().any():
        raise ValueError(
            f"Duplicate identifiers were found in classification column "
            f"{task.label_column!r}."
        )
    available = _preprocessed_identifiers(preprocessed_folder)
    labelled_ids = set(labelled[identifier_column])
    missing_preprocessed = sorted(labelled_ids - available)
    if missing_preprocessed:
        raise ValueError(
            f"{len(missing_preprocessed)} labelled cases are absent from the "
            f"preprocessed data, including {missing_preprocessed[:5]}."
        )
    all_labelled_ids = labelled[identifier_column].tolist()
    if not all_labelled_ids:
        raise ValueError(f"Task {task.name!r} has no labelled cases.")
    split = {"train": all_labelled_ids, "val": all_labelled_ids}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([split], indent=2) + "\n", encoding="utf-8")
    return output_path, split


def _manifest_path(work_root: Path, dataset_name: str, kind: str) -> Path:
    return work_root / "manifests" / f"{dataset_name}.{kind}.json"
