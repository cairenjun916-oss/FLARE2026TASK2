from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


FROZEN_EXPERIMENT_NAMES = {
    "A0": "A0_baseline",
    "A1": "A1_global_roi",
    "A2": "A2_global_roi_volume",
    "A3": "A3_global_roi_volume_confidence",
}


def _normalise_feature_mode(feature_mode: str) -> str:
    """Return one of A0-A3 from a short or descriptive mode name."""
    aliases = {
        "A0": "A0",
        "GLOBAL": "A0",
        "BASELINE": "A0",
        "A1": "A1",
        "GLOBAL_ROI": "A1",
        "ROI": "A1",
        "A2": "A2",
        "GLOBAL_ROI_VOLUME": "A2",
        "ROI_VOLUME": "A2",
        "A3": "A3",
        "GLOBAL_ROI_VOLUME_CONFIDENCE": "A3",
        "ROI_VOLUME_CONFIDENCE": "A3",
    }
    key = str(feature_mode).strip().upper().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        raise ValueError(
            f"Unknown feature_mode={feature_mode!r}. Use A0, A1, A2, or A3."
        )
    return aliases[key]


def _frozen_feature_dimension(feature_channels: int, feature_mode: str) -> int:
    """Calculate classifier input width for a cumulative A0-A3 design."""
    mode = _normalise_feature_mode(feature_mode)
    if mode == "A0":
        return feature_channels
    if mode == "A1":
        return 2 * feature_channels
    if mode == "A2":
        return (2 * feature_channels) + 1
    return (2 * feature_channels) + 2


@dataclass
class A0Setup:
    project_root: Path
    raw_dataset_folder: Path
    dataset_name: str
    task_name: str
    identifier_column: str
    label_column: str
    segmentation_fold: str | int
    classification_fold: int
    device: torch.device
    predictor: nnUNetPredictor
    model: nn.Module
    segmentation_model_dir: Path
    segmentation_checkpoint: str
    segmentation_checkpoint_path: Path
    preprocessed_folder: Path
    cls_csv: Path
    split_json: Path
    cls_df: pd.DataFrame
    label_values: list[Any]
    label_to_index: dict[Any, int]
    num_classes: int
    feature_channels: int
    feature_mode: str
    feature_dimension: int
    patch_size: tuple[int, ...]


@dataclass
class A0Data:
    train_dataset: Dataset
    val_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    train_identifiers: list[str]
    val_identifiers: list[str]
    label_by_identifier: dict[str, Any]
    class_counts: np.ndarray
    class_weights: torch.Tensor
    class_weight_strategy: str = "inverse"
    label_smoothing: float = 0.0


@dataclass
class FeatureData:
    train_loader: DataLoader
    val_loader: DataLoader
    train_cache: Path
    val_cache: Path
    train_metadata: dict[str, Any]
    val_metadata: dict[str, Any]


@dataclass
class TrainingResult:
    best_model_path: Path
    last_model_path: Path
    history_path: Path
    history: pd.DataFrame
    best_val_loss: float


@dataclass
class InferenceResult:
    output_dir: Path
    results_csv: Path
    efficiency_csv: Path
    classification: pd.DataFrame
    efficiency: pd.DataFrame


def balanced_accuracy_details(
    targets: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Calculate and independently verify macro-average class recall.

    Balanced accuracy is the unweighted mean of recall calculated separately
    for every class represented in the ground truth. For binary data, this is
    ``(sensitivity + specificity) / 2``. The explicit calculation is checked
    against scikit-learn so training and final evaluation cannot silently use
    different definitions.
    """
    targets_array = np.asarray(targets, dtype=int).reshape(-1)
    predictions_array = np.asarray(predictions, dtype=int).reshape(-1)
    if targets_array.size == 0:
        raise ValueError("Balanced accuracy cannot be calculated with no cases.")
    if targets_array.shape != predictions_array.shape:
        raise ValueError(
            "Targets and predictions must have the same shape; received "
            f"{targets_array.shape} and {predictions_array.shape}."
        )
    if num_classes < 2:
        raise ValueError("Classification requires at least two classes.")
    if (
        np.any(targets_array < 0)
        or np.any(targets_array >= num_classes)
        or np.any(predictions_array < 0)
        or np.any(predictions_array >= num_classes)
    ):
        raise ValueError(
            f"Targets and predictions must be in [0, {num_classes - 1}]."
        )

    support = np.bincount(targets_array, minlength=num_classes)
    correct = np.bincount(
        targets_array[targets_array == predictions_array],
        minlength=num_classes,
    )
    present_mask = support > 0
    per_class_recall = np.full(num_classes, np.nan, dtype=float)
    per_class_recall[present_mask] = (
        correct[present_mask] / support[present_mask]
    )
    explicit_score = float(per_class_recall[present_mask].mean())
    sklearn_score = float(
        balanced_accuracy_score(targets_array, predictions_array)
    )
    if not np.isclose(explicit_score, sklearn_score, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Internal balanced-accuracy verification failed: explicit macro "
            f"recall={explicit_score}, sklearn={sklearn_score}."
        )

    return {
        "balanced_accuracy": explicit_score,
        "class_support": support.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "present_class_indices": np.flatnonzero(present_mask).tolist(),
        "all_classes_present": bool(present_mask.all()),
    }


def _classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    """Return hard-decision and ranking metrics used by all experiments."""
    details = balanced_accuracy_details(targets, predictions, num_classes)
    average = "binary" if num_classes == 2 else "weighted"
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(details["balanced_accuracy"]),
        "f1": float(
            f1_score(
                targets,
                predictions,
                average=average,
                zero_division=0,
            )
        ),
    }
    if probabilities is None:
        return metrics

    targets_array = np.asarray(targets, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    try:
        if num_classes == 2:
            positive_scores = probability_array[:, 1]
            metrics["auroc"] = float(
                roc_auc_score(targets_array, positive_scores)
            )
            metrics["auprc"] = float(
                average_precision_score(targets_array, positive_scores)
            )
        else:
            metrics["auroc"] = float(
                roc_auc_score(
                    targets_array,
                    probability_array,
                    labels=np.arange(num_classes),
                    multi_class="ovr",
                    average="macro",
                )
            )
            one_hot = np.eye(num_classes, dtype=float)[targets_array]
            metrics["auprc"] = float(
                average_precision_score(one_hot, probability_array, average="macro")
            )
    except ValueError:
        # AUROC/AUPRC are undefined if an evaluated view lacks a class.
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    auroc = metrics["auroc"]
    metrics["selection_score"] = (
        0.5 * (metrics["balanced_accuracy"] + auroc)
        if np.isfinite(auroc)
        else metrics["balanced_accuracy"]
    )
    return metrics


def automatic_class_weights(
    counts: Sequence[int] | np.ndarray,
    strategy: str = "auto",
) -> tuple[np.ndarray, str]:
    """Derive stable class weights from the observed training distribution.

    ``auto`` uses effective-number weighting for imbalanced data. Unlike raw
    inverse-frequency weights, effective-number weights stop a very small class
    from dominating every gradient while still increasing its contribution.
    """
    class_counts = np.asarray(counts, dtype=float)
    if class_counts.ndim != 1 or class_counts.size < 2 or np.any(class_counts <= 0):
        raise ValueError(
            f"Every training class needs at least one case; counts={class_counts.tolist()}."
        )
    strategy = str(strategy).lower().replace("-", "_")
    imbalance = float(class_counts.max() / class_counts.min())
    if strategy == "auto":
        strategy = "none" if imbalance < 1.5 else "effective_number"
    if strategy == "none":
        weights = np.ones_like(class_counts)
    elif strategy == "inverse":
        weights = 1.0 / class_counts
    elif strategy == "sqrt_inverse":
        weights = 1.0 / np.sqrt(class_counts)
    elif strategy == "effective_number":
        beta = 1.0 - (1.0 / float(class_counts.sum()))
        effective_counts = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
        weights = 1.0 / effective_counts
    else:
        raise ValueError(
            "class-weight strategy must be auto, effective_number, inverse, "
            f"sqrt_inverse, or none; received {strategy!r}."
        )
    # Keep the expected per-sample loss scale at one. Only relative weights
    # matter to cross entropy, but stable scale helps optimizer portability.
    weights /= np.average(weights, weights=class_counts)
    return weights.astype(np.float32), strategy


def automatic_label_smoothing(counts: Sequence[int] | np.ndarray) -> float:
    """Choose mild regularization from scarcity and imbalance, capped at 0.05."""
    class_counts = np.asarray(counts, dtype=float)
    minority = float(class_counts.min())
    imbalance = float(class_counts.max() / minority)
    scarcity_factor = min(1.0, 20.0 / minority)
    imbalance_factor = min(1.0, np.log(max(imbalance, 1.0)) / np.log(10.0))
    return float(0.05 * max(scarcity_factor, imbalance_factor))


def automatic_head_configuration(
    counts: Sequence[int] | np.ndarray,
) -> tuple[int, float]:
    """Choose a small MLP capacity and dropout from dataset size/imbalance."""
    class_counts = np.asarray(counts, dtype=float)
    if class_counts.ndim != 1 or class_counts.size < 2 or np.any(class_counts <= 0):
        raise ValueError(
            f"Every training class needs at least one case; counts={class_counts.tolist()}."
        )
    total = float(class_counts.sum())
    minority = float(class_counts.min())
    imbalance = float(class_counts.max() / minority)
    target_width = np.sqrt(total) * 8.0
    hidden_channels = int(
        np.clip(2 ** round(np.log2(max(target_width, 1.0))), 32, 256)
    )
    scarcity_factor = min(1.0, 256.0 / total)
    imbalance_factor = min(1.0, np.log(max(imbalance, 1.0)) / np.log(10.0))
    dropout = float(0.25 + 0.15 * max(scarcity_factor, imbalance_factor))
    return hidden_channels, dropout

