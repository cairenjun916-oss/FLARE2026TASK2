from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .common import (
    DatasetContext,
    _configure_environment,
    _manifest_path,
    _resolve_dataset,
    _resolve_device,
)


def _group_test_images(context: DatasetContext) -> list[tuple[str, list[Path]]]:
    images_folder = context.folder / "imagesTs"
    if not images_folder.is_dir():
        raise FileNotFoundError(f"Missing test image folder: {images_folder}")
    ending = str(context.metadata["file_ending"])
    channel_count = len(context.metadata["channel_names"])
    grouped: dict[str, dict[int, Path]] = {}
    for path in sorted(images_folder.glob(f"*{ending}")):
        stem = path.name[: -len(ending)]
        match = re.fullmatch(r"(.+)_(\d{4})", stem)
        if match is None:
            raise ValueError(f"Invalid nnU-Net channel filename: {path.name}")
        case_id, channel_text = match.groups()
        channel = int(channel_text)
        if channel in grouped.setdefault(case_id, {}):
            raise ValueError(f"Duplicate channel {channel:04d} for {case_id}.")
        grouped[case_id][channel] = path
    if not grouped:
        raise ValueError(f"No test images ending in {ending!r} were found.")
    expected = set(range(channel_count))
    cases: list[tuple[str, list[Path]]] = []
    for case_id in sorted(grouped):
        actual = set(grouped[case_id])
        if actual != expected:
            raise ValueError(
                f"{case_id} has channels {sorted(actual)}; expected "
                f"{sorted(expected)} from dataset.json."
            )
        cases.append((case_id, [grouped[case_id][i] for i in sorted(expected)]))
    return cases


def _center_patch(data, patch_size: Sequence[int]):
    import torch
    import torch.nn.functional as functional

    tensor = torch.as_tensor(data, dtype=torch.float32)
    spatial = tensor.shape[1:]
    if len(spatial) != len(patch_size):
        raise ValueError(
            f"Preprocessed image has {len(spatial)} spatial dimensions, but "
            f"the plan has patch size {tuple(patch_size)}."
        )
    padding: list[int] = []
    for size, target in reversed(list(zip(spatial, patch_size))):
        missing = max(int(target) - int(size), 0)
        padding.extend((missing // 2, missing - missing // 2))
    if any(padding):
        tensor = functional.pad(tensor, padding)
    slices = [slice(None)]
    for size, target in zip(tensor.shape[1:], patch_size):
        start = max((int(size) - int(target)) // 2, 0)
        slices.append(slice(start, start + int(target)))
    return tensor[tuple(slices)].contiguous()


def _build_classifier_head(checkpoint: dict[str, Any], device):
    import torch
    from torch import nn

    state = checkpoint["classifier_state_dict"]
    first_weight = state["0.weight"]
    last_weight = state["3.weight"]
    head = nn.Sequential(
        nn.Linear(int(first_weight.shape[1]), int(first_weight.shape[0])),
        nn.ReLU(inplace=True),
        nn.Dropout(float(checkpoint.get("dropout", 0.0))),
        nn.Linear(int(last_weight.shape[1]), int(last_weight.shape[0])),
    )
    head.load_state_dict(state, strict=True)
    return head.to(device).eval()


def _validate_prediction_geometry(
    source: Path, prediction: Path, allowed_labels: set[int]
) -> None:
    import numpy as np
    import SimpleITK as sitk

    source_image = sitk.ReadImage(str(source))
    predicted_image = sitk.ReadImage(str(prediction))
    if source_image.GetSize() != predicted_image.GetSize():
        raise ValueError(f"Shape mismatch for {prediction.name}.")
    for field, source_value, predicted_value in (
        ("spacing", source_image.GetSpacing(), predicted_image.GetSpacing()),
        ("origin", source_image.GetOrigin(), predicted_image.GetOrigin()),
        ("direction", source_image.GetDirection(), predicted_image.GetDirection()),
    ):
        if not np.allclose(source_value, predicted_value, rtol=0.0, atol=1e-5):
            raise ValueError(f"{field.title()} mismatch for {prediction.name}.")
    observed = set(map(int, np.unique(sitk.GetArrayViewFromImage(predicted_image))))
    if not observed <= allowed_labels:
        raise ValueError(
            f"{prediction.name} contains labels {sorted(observed - allowed_labels)} "
            f"outside dataset.json."
        )


def _command_infer(args: argparse.Namespace) -> None:
    import pandas as pd
    import torch

    context = _resolve_dataset(args.dataset, args.raw_root)
    work_root = Path(args.work_root).expanduser().resolve()
    _configure_environment(context, work_root)
    device = _resolve_device(args.device)
    from nnunetv2.inference.export_prediction import export_prediction_from_logits

    from ..classifier import setup_a0
    manifest_path = _manifest_path(work_root, context.name, "classification")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing classification manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segmentation = manifest["segmentation"]
    if segmentation.get("checkpoint_name") != "checkpoint_best.pth":
        raise ValueError("Inference requires the best segmentation checkpoint.")
    tasks = manifest.get("tasks", [])
    if not tasks:
        raise ValueError(f"No classification tasks are recorded in {manifest_path}.")
    for task in tasks:
        if "models" in task:
            raise ValueError(
                "This manifest contains a classifier ensemble. Retrain with the "
                "single-full-data policy before competition inference."
            )
        if task.get("checkpoint_name") not in {
            "classifier_final.pth",
            "classifier_best.pth",
        }:
            raise ValueError(f"Invalid classifier checkpoint for {task.get('name')}.")

    first = tasks[0]
    setup = setup_a0(
        project_root=work_root,
        dataset_name=context.name,
        segmentation_fold=segmentation["fold"],
        classification_fold=manifest["classification_fold"],
        device=device,
        model_name=segmentation["model_name"],
        checkpoint_names=("checkpoint_best.pth",),
        feature_mode=manifest["feature_mode"],
        label_column=first["label_column"],
        identifier_column=manifest["identifier_column"],
        task_name=first["task_key"],
        split_json_path=first["split_path"],
        perform_everything_on_device=False,
    )
    heads: list[tuple[dict[str, Any], Any]] = []
    for task in tasks:
        checkpoint_path = Path(task["checkpoint_path"])
        if (
            checkpoint_path.name != task["checkpoint_name"]
            or not checkpoint_path.is_file()
        ):
            raise FileNotFoundError(
                f"Classifier checkpoint is missing: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("feature_mode") != manifest["feature_mode"]:
            raise ValueError(f"Feature-mode mismatch in {checkpoint_path}.")
        if list(checkpoint.get("label_values", [])) != list(task["class_values"]):
            raise ValueError(f"Class-order mismatch in {checkpoint_path}.")
        heads.append((task, _build_classifier_head(checkpoint, device)))

    prediction_root = Path(args.output_root).expanduser().resolve()
    output_dir = prediction_root / f"{context.name}_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _group_test_images(context)
    ending = str(context.metadata["file_ending"])
    expected_files = {f"{case_id}{ending}" for case_id, _ in cases}
    stale = sorted(
        path.name
        for path in output_dir.glob(f"*{ending}")
        if path.name not in expected_files
    )
    if stale:
        raise ValueError(
            f"Output folder contains stale segmentation files: {stale[:5]}. "
            "Remove them before inference so the submission has exactly one "
            "mask per test case."
        )

    preprocessor = setup.predictor.configuration_manager.preprocessor_class(
        verbose=False
    )
    rows: list[dict[str, Any]] = []
    for case_index, (case_id, channels) in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case_id}")
        data, _, properties = preprocessor.run_case(
            [str(path) for path in channels],
            None,
            setup.predictor.plans_manager,
            setup.predictor.configuration_manager,
            setup.predictor.dataset_json,
        )
        segmentation_logits = setup.predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(data)
        ).cpu()
        output_truncated = output_dir / case_id
        export_prediction_from_logits(
            segmentation_logits,
            properties,
            setup.predictor.configuration_manager,
            setup.predictor.plans_manager,
            setup.predictor.dataset_json,
            str(output_truncated),
            save_probabilities=False,
            num_threads_torch=args.export_threads,
        )

        patch = _center_patch(data, setup.patch_size).unsqueeze(0).to(device)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            features = setup.model.extract_features(patch)
            task_probabilities = []
            for _, head in heads:
                probabilities = torch.softmax(head(features).float(), dim=1)[0]
                task_probabilities.append(probabilities.cpu().tolist())
        row: dict[str, Any] = {"case_id": case_id}
        for (task, _), probabilities in zip(heads, task_probabilities):
            columns = task["output_columns"]
            if len(columns) == 1:
                row[columns[0]] = float(probabilities[1])
            else:
                row.update(
                    {column: float(value) for column, value in zip(columns, probabilities)}
                )
        rows.append(row)
        del data, segmentation_logits, patch, features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    columns = ["case_id"] + [
        column for task in tasks for column in task["output_columns"]
    ]
    predictions = pd.DataFrame(rows, columns=columns)
    predictions_csv = output_dir / "predictions.csv"
    predictions.to_csv(predictions_csv, index=False)

    if predictions["case_id"].duplicated().any() or len(predictions) != len(cases):
        raise RuntimeError("Classification output is not one-to-one with test cases.")
    for task in tasks:
        output_columns = task["output_columns"]
        values = predictions[output_columns].to_numpy(dtype=float)
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise ValueError(f"Probabilities are outside [0, 1] for {task['name']}.")
        if len(output_columns) > 1:
            import numpy as np

            if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-5):
                raise ValueError(f"Probabilities do not sum to 1 for {task['name']}.")
    allowed_labels: set[int] = set()
    for value in context.metadata["labels"].values():
        values = value if isinstance(value, (list, tuple)) else (value,)
        allowed_labels.update(
            int(item) for item in values if isinstance(item, (int, float))
        )
    for case_id, channels in cases:
        _validate_prediction_geometry(
            channels[0], output_dir / f"{case_id}{ending}", allowed_labels
        )
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    required_files = expected_files | {"predictions.csv"}
    if actual_files != required_files:
        raise ValueError(
            f"Submission folder has unexpected or missing files. "
            f"Unexpected={sorted(actual_files - required_files)}, "
            f"missing={sorted(required_files - actual_files)}"
        )
    print(f"Prediction folder: {output_dir}")
    print(f"Classification probabilities: {predictions_csv}")
