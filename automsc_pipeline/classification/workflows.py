from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .core import (
    FROZEN_EXPERIMENT_NAMES,
    A0Data,
    A0Setup,
    FeatureData,
    InferenceResult,
    TrainingResult,
    _normalise_feature_mode,
    balanced_accuracy_details,
)
from .data import (
    build_classification_data,
    build_finetuning_data,
    cache_frozen_features,
    configure_environment,
    setup_a0,
)
from .finetuning import (
    load_classifier_checkpoint,
    setup_partial_unfreeze,
    train_partial_unfreeze,
)
from .frozen_training import train_cached_classifier


def run_fold_inference(
    setup: A0Setup,
    data: A0Data,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    raw_images_folder: str | Path | None = None,
    runtime_limit_seconds: float = 60.0,
    gpu_memory_tolerance_mb: float = 4096.0,
    overwrite: bool = True,
    save_probabilities: bool = False,
    preprocessing_processes: int = 1,
    export_processes: int = 1,
    experiment_name: str = "A0",
) -> InferenceResult:
    """Run original-space segmentation and classification per fold case."""
    checkpoint_path = Path(checkpoint_path)
    loaded_checkpoint = load_classifier_checkpoint(setup, checkpoint_path)
    output_dir = Path(
        output_dir
        or setup.project_root
        / "outputs"
        / experiment_name
        / f"fold{setup.classification_fold}_train"
        / setup.dataset_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_metadata = {
        "experiment_name": experiment_name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": loaded_checkpoint.get("epoch"),
        "checkpoint_metric": loaded_checkpoint.get("checkpoint_metric"),
        "best_checkpoint_score": loaded_checkpoint.get(
            "best_checkpoint_score"
        ),
        "overwrite": bool(overwrite),
        "predictor_weight_installation": loaded_checkpoint.get(
            "predictor_weight_installation",
            {
                "installed": False,
                "reason": "frozen_backbone_experiment",
            },
        ),
    }
    (output_dir / "inference_checkpoint.json").write_text(
        json.dumps(inference_metadata, indent=2),
        encoding="utf-8",
    )
    raw_images_folder = Path(
        raw_images_folder
        or Path(os.environ["nnUNet_raw"]) / setup.dataset_name / "imagesTr"
    )
    val_index = {v: i for i, v in enumerate(data.val_dataset.identifiers)}
    cls_rows, efficiency_rows = [], []

    for identifier in tqdm(
        data.val_identifiers, desc=f"{experiment_name} inference"
    ):
        if setup.device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(setup.device)
            torch.cuda.synchronize(setup.device)
        start = time.perf_counter()
        channels = sorted(raw_images_folder.glob(f"{identifier}_*.nii.gz"))
        setup.predictor.predict_from_files(
            list_of_lists_or_source_folder=[[str(v) for v in channels]],
            output_folder_or_list_of_truncated_output_files=[
                str(output_dir / identifier)
            ],
            save_probabilities=save_probabilities,
            overwrite=overwrite,
            num_processes_preprocessing=preprocessing_processes,
            num_processes_segmentation_export=export_processes,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        case = data.val_dataset[val_index[identifier]]
        image = case[0]
        image = image.unsqueeze(0).to(setup.device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=setup.device.type, enabled=setup.device.type == "cuda"
        ):
            _, logits = setup.model(image, return_seg=False)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
        predicted_index = int(np.argmax(probabilities))
        probability_output: float | list[float] = (
            float(probabilities[1])
            if setup.num_classes == 2
            else [float(v) for v in probabilities]
        )
        if setup.device.type == "cuda":
            torch.cuda.synchronize(setup.device)
        runtime = time.perf_counter() - start
        peak_mb = (
            torch.cuda.max_memory_allocated(setup.device) / 1024**2
            if setup.device.type == "cuda"
            else 0.0
        )
        cls_rows.append(
            {
                "identifier": identifier,
                "probs": str(probability_output)
                if isinstance(probability_output, list)
                else probability_output,
                "predicted_index": predicted_index,
                "predicted_class": setup.label_values[predicted_index],
                **{f"probability_class_{i}": p for i, p in enumerate(probabilities)},
            }
        )
        efficiency_rows.append(
            {
                "identifier": identifier,
                "runtime_seconds": runtime,
                "runtime_limit_seconds": runtime_limit_seconds,
                "runtime_exceeded": runtime > runtime_limit_seconds,
                "raw_peak_gpu_memory_mb": peak_mb,
                "gpu_memory_tolerance_mb": gpu_memory_tolerance_mb,
                "adjusted_peak_gpu_memory_mb": max(
                    peak_mb - gpu_memory_tolerance_mb, 0.0
                ),
                "raw_gpu_memory_time_mb_seconds": peak_mb * runtime,
                "adjusted_gpu_memory_time_mb_seconds": max(
                    peak_mb - gpu_memory_tolerance_mb, 0.0
                )
                * runtime,
                "success": True,
                "challenge_case_pass": runtime <= runtime_limit_seconds,
            }
        )

    classification = pd.DataFrame(cls_rows)
    efficiency = pd.DataFrame(efficiency_rows)
    results_csv = output_dir / f"fold{setup.classification_fold}_results.csv"
    efficiency_csv = output_dir / f"fold{setup.classification_fold}_efficiency.csv"
    classification.to_csv(results_csv, index=False)
    classification.to_csv(output_dir / "results.csv", index=False)
    efficiency.to_csv(efficiency_csv, index=False)
    efficiency.to_csv(output_dir / "efficiency.csv", index=False)
    return InferenceResult(
        output_dir,
        results_csv,
        efficiency_csv,
        classification,
        efficiency,
    )


def evaluate_predictions(
    setup: A0Setup,
    inference: InferenceResult,
    gt_seg_folder: str | Path | None = None,
    gt_cls_csv: str | Path | None = None,
    eval_script: str | Path | None = None,
    num_seg_classes: int | None = None,
    output_dir: str | Path | None = None,
    experiment_name: str = "A0",
) -> str:
    """Call AutoMSC eval_metrics.py and append efficiency statistics."""
    gt_seg_folder = Path(
        gt_seg_folder
        or Path(os.environ["nnUNet_raw"]) / setup.dataset_name / "labelsTr"
    )
    gt_cls_csv = Path(gt_cls_csv or setup.cls_csv)
    eval_script = Path(eval_script or setup.project_root / "eval_metrics.py")
    if num_seg_classes is None:
        manager = setup.predictor.label_manager
        labels = (
            list(manager.all_labels)
            if hasattr(manager, "all_labels")
            else [0, *list(manager.foreground_labels)]
        )
        num_seg_classes = int(max(labels) + 1)
    command = [
        sys.executable,
        str(eval_script),
        "--pred_seg_path",
        str(inference.output_dir),
        "--gt_seg_path",
        str(gt_seg_folder),
        "--pred_cls_csv",
        str(inference.results_csv),
        "--gt_cls_csv",
        str(gt_cls_csv),
        "--num_seg_classes",
        str(num_seg_classes),
        "--num_cls_classes",
        str(setup.num_classes),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    text = completed.stdout
    if completed.stderr.strip():
        text += "\n--- STDERR ---\n" + completed.stderr

    # The official evaluator reports ordinary accuracy but not balanced
    # accuracy. Calculate it from the same validation predictions and insert
    # it directly below Accuracy without changing any other evaluator output.
    classification_ground_truth = pd.read_csv(gt_cls_csv)
    required_columns = {setup.identifier_column, setup.label_column}
    missing_columns = required_columns - set(classification_ground_truth.columns)
    if missing_columns:
        raise KeyError(
            f"{gt_cls_csv} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if classification_ground_truth[setup.identifier_column].astype(str).duplicated().any():
        raise ValueError(
            f"{gt_cls_csv} contains duplicate classification identifiers."
        )
    if inference.classification["identifier"].astype(str).duplicated().any():
        raise ValueError("Classification predictions contain duplicate identifiers.")

    label_by_identifier = dict(
        zip(
            classification_ground_truth[setup.identifier_column].astype(str),
            classification_ground_truth[setup.label_column],
        )
    )
    prediction_identifiers = (
        inference.classification["identifier"].astype(str).tolist()
    )
    missing_identifiers = [
        identifier
        for identifier in prediction_identifiers
        if identifier not in label_by_identifier
    ]
    if missing_identifiers:
        raise KeyError(
            "Classification ground truth is missing for identifiers: "
            + ", ".join(missing_identifiers[:10])
        )
    classification_targets = np.asarray(
        [
            setup.label_to_index[label_by_identifier[identifier]]
            for identifier in prediction_identifiers
        ],
        dtype=int,
    )
    classification_predictions = inference.classification[
        "predicted_index"
    ].to_numpy(dtype=int)
    balanced_details = balanced_accuracy_details(
        classification_targets,
        classification_predictions,
        setup.num_classes,
    )
    balanced_accuracy = float(balanced_details["balanced_accuracy"])
    accuracy_line = next(
        (
            line
            for line in text.splitlines()
            if line.strip().startswith("Accuracy:")
        ),
        None,
    )
    balanced_accuracy_line = (
        f"  Balanced Accuracy: {balanced_accuracy:.4f}"
    )
    if accuracy_line is not None:
        text = text.replace(
            accuracy_line,
            accuracy_line + "\n" + balanced_accuracy_line,
            1,
        )
    else:
        text += (
            "\n--- Additional Classification Metrics ---\n"
            + balanced_accuracy_line
            + "\n"
        )

    recall_parts = []
    for class_index, (support, recall) in enumerate(
        zip(
            balanced_details["class_support"],
            balanced_details["per_class_recall"],
        )
    ):
        recall_text = "NA" if np.isnan(recall) else f"{recall:.4f}"
        recall_parts.append(
            f"class {class_index}: recall={recall_text}, n={support}"
        )
    text += (
        "\n--- Balanced Accuracy Verification ---\n"
        "  Definition: unweighted mean of per-class recall\n"
        "  "
        + " | ".join(recall_parts)
        + f"\n  Verified Balanced Accuracy: {balanced_accuracy:.4f}\n"
    )

    efficiency = inference.efficiency
    text += (
        "\n--- Efficiency Metrics ---\n"
        f"Mean inference time: {efficiency.runtime_seconds.mean():.4f} s\n"
        f"Maximum inference time: {efficiency.runtime_seconds.max():.4f} s\n"
        f"Mean peak GPU memory: {efficiency.raw_peak_gpu_memory_mb.mean():.2f} MB\n"
        f"Maximum peak GPU memory: {efficiency.raw_peak_gpu_memory_mb.max():.2f} MB\n"
    )
    output_dir = Path(
        output_dir
        or setup.project_root
        / "eval_results"
        / experiment_name
        / f"fold{setup.classification_fold}_train"
        / setup.dataset_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval_log.txt"
    log_path.write_text(text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Evaluation failed. See {log_path}\n{text}")
    print(text)
    return text


def run_frozen_feature_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    feature_mode: str,
    segmentation_fold: str | int = "all",
    classification_fold: int = 0,
    train_variants: int = 2,
    epochs: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    image_batch_size: int = 1,
    feature_batch_size: int = 64,
    num_workers: int = 0,
    model_name: str | None = None,
    device: str | torch.device | None = None,
    hidden_channels: int = 256,
    dropout: float = 0.30,
    checkpoint_metric: str | None = None,
    reuse_cache: bool = True,
    label_column: str = "label",
    identifier_column: str | None = None,
    task_name: str | None = None,
    split_json_path: str | Path | None = None,
) -> tuple[A0Setup, A0Data, FeatureData, TrainingResult]:
    """Run one frozen-backbone A0-A3 experiment end to end."""
    feature_mode = _normalise_feature_mode(feature_mode)
    experiment_name = FROZEN_EXPERIMENT_NAMES[feature_mode]
    configure_environment(project_root, raw_root)
    setup = setup_a0(
        project_root=project_root,
        dataset_name=dataset_name,
        segmentation_fold=segmentation_fold,
        classification_fold=classification_fold,
        device=device,
        model_name=model_name,
        hidden_channels=hidden_channels,
        dropout=dropout,
        feature_mode=feature_mode,
        label_column=label_column,
        identifier_column=identifier_column,
        task_name=task_name,
        split_json_path=split_json_path,
    )
    data = build_classification_data(
        setup, image_batch_size=image_batch_size, num_workers=num_workers
    )
    features = cache_frozen_features(
        setup,
        data,
        train_variants=train_variants,
        feature_batch_size=feature_batch_size,
        extraction_num_workers=num_workers,
        reuse_cache=reuse_cache,
        experiment_name=experiment_name,
    )
    training = train_cached_classifier(
        setup,
        data,
        features,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_metric=checkpoint_metric,
        experiment_name=experiment_name,
    )
    return setup, data, features, training


def run_a0_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    **kwargs: Any,
) -> tuple[A0Setup, A0Data, FeatureData, TrainingResult]:
    """Run A0: frozen encoder with global pooled bottleneck features."""
    return run_frozen_feature_training(
        project_root=project_root,
        raw_root=raw_root,
        dataset_name=dataset_name,
        feature_mode="A0",
        **kwargs,
    )


def run_a1_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    **kwargs: Any,
) -> tuple[A0Setup, A0Data, FeatureData, TrainingResult]:
    """Run A1: A0 global features plus segmentation-guided ROI features."""
    return run_frozen_feature_training(
        project_root=project_root,
        raw_root=raw_root,
        dataset_name=dataset_name,
        feature_mode="A1",
        **kwargs,
    )


def run_a2_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    **kwargs: Any,
) -> tuple[A0Setup, A0Data, FeatureData, TrainingResult]:
    """Run A2: A1 features plus predicted foreground-volume fraction."""
    return run_frozen_feature_training(
        project_root=project_root,
        raw_root=raw_root,
        dataset_name=dataset_name,
        feature_mode="A2",
        **kwargs,
    )


def run_a3_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    **kwargs: Any,
) -> tuple[A0Setup, A0Data, FeatureData, TrainingResult]:
    """Run A3: A2 features plus segmentation-confidence feature."""
    return run_frozen_feature_training(
        project_root=project_root,
        raw_root=raw_root,
        dataset_name=dataset_name,
        feature_mode="A3",
        **kwargs,
    )


def run_a4_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    segmentation_fold: str | int = "all",
    classification_fold: int = 0,
    epochs: int = 100,
    classifier_learning_rate: float = 1e-4,
    backbone_learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    image_batch_size: int = 1,
    num_workers: int = 0,
    unfreeze_last_n_stages: int = 1,
    a0_checkpoint_path: str | Path | None = None,
    model_name: str | None = None,
    device: str | torch.device | None = None,
    checkpoint_metric: str = "val_balanced_accuracy",
    use_class_weights: bool = True,
) -> tuple[A0Setup, A0Data, TrainingResult]:
    """Run A4: classification with partial encoder unfreezing."""
    configure_environment(project_root, raw_root)
    setup = setup_partial_unfreeze(
        project_root=project_root,
        dataset_name=dataset_name,
        segmentation_fold=segmentation_fold,
        classification_fold=classification_fold,
        device=device,
        model_name=model_name,
        unfreeze_last_n_stages=unfreeze_last_n_stages,
        a0_checkpoint_path=a0_checkpoint_path,
    )
    data = build_finetuning_data(
        setup=setup,
        include_segmentation=False,
        image_batch_size=image_batch_size,
        num_workers=num_workers,
        use_class_weights=use_class_weights,
    )
    training = train_partial_unfreeze(
        setup=setup,
        data=data,
        experiment_name="A4_partial_unfreeze",
        epochs=epochs,
        classifier_learning_rate=classifier_learning_rate,
        backbone_learning_rate=backbone_learning_rate,
        weight_decay=weight_decay,
        auxiliary_segmentation_weight=0.0,
        checkpoint_metric=checkpoint_metric,
    )
    return setup, data, training


def run_a5_training(
    project_root: str | Path,
    raw_root: str | Path,
    dataset_name: str,
    segmentation_fold: str | int = "all",
    classification_fold: int = 0,
    epochs: int = 100,
    classifier_learning_rate: float = 1e-4,
    backbone_learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    auxiliary_segmentation_weight: float = 0.2,
    image_batch_size: int = 1,
    num_workers: int = 0,
    unfreeze_last_n_stages: int = 1,
    a0_checkpoint_path: str | Path | None = None,
    initial_finetuned_checkpoint_path: str | Path | None = None,
    model_name: str | None = None,
    device: str | torch.device | None = None,
    checkpoint_metric: str = "val_balanced_accuracy",
    use_class_weights: bool = True,
) -> tuple[A0Setup, A0Data, TrainingResult]:
    """Run A5: A4 plus an auxiliary Dice and CE/BCE segmentation loss."""
    configure_environment(project_root, raw_root)
    setup = setup_partial_unfreeze(
        project_root=project_root,
        dataset_name=dataset_name,
        segmentation_fold=segmentation_fold,
        classification_fold=classification_fold,
        device=device,
        model_name=model_name,
        unfreeze_last_n_stages=unfreeze_last_n_stages,
        a0_checkpoint_path=a0_checkpoint_path,
        initial_finetuned_checkpoint_path=initial_finetuned_checkpoint_path,
    )
    data = build_finetuning_data(
        setup=setup,
        include_segmentation=True,
        image_batch_size=image_batch_size,
        num_workers=num_workers,
        use_class_weights=use_class_weights,
    )
    training = train_partial_unfreeze(
        setup=setup,
        data=data,
        experiment_name="A5_auxiliary_segmentation",
        epochs=epochs,
        classifier_learning_rate=classifier_learning_rate,
        backbone_learning_rate=backbone_learning_rate,
        weight_decay=weight_decay,
        auxiliary_segmentation_weight=auxiliary_segmentation_weight,
        checkpoint_metric=checkpoint_metric,
    )
    return setup, data, training
