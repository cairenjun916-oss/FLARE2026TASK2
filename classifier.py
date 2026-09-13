"""A0-A5 segmentation-backbone workflows for AutoMSC.

A0-A3 cache frozen segmentation-backbone features and train only the
classification head. A1 adds a segmentation-guided ROI feature, A2 adds a
foreground-volume feature, and A3 adds a segmentation-confidence feature.
A4 trains the classification head while partially unfreezing the last encoder
stage. A5 uses the same partial unfreezing and adds an auxiliary segmentation
loss through the frozen decoder.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureTyped,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


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


class A0FrozenSegClassifier(nn.Module):
    """Frozen nnU-Net feature builder and MLP classifier for A0-A3.

    Feature modes are cumulative:

    - A0: global pooled bottleneck feature.
    - A1: A0 plus soft segmentation-guided ROI pooled feature.
    - A2: A1 plus predicted foreground-volume fraction.
    - A3: A2 plus foreground-weighted segmentation confidence.

    A soft foreground mask is used so very small lesions do not produce an
    empty ROI feature. The same implementation supports ordinary softmax
    labels and nnU-Net region-based sigmoid outputs.
    """

    def __init__(
        self,
        segmentation_network: nn.Module,
        feature_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        dropout: float = 0.30,
        feature_mode: str = "A0",
        label_manager: Any | None = None,
    ) -> None:
        super().__init__()
        self.feature_mode = _normalise_feature_mode(feature_mode)
        self.feature_channels = int(feature_channels)
        self.feature_dimension = _frozen_feature_dimension(
            self.feature_channels, self.feature_mode
        )
        self.label_manager = label_manager
        self.segmentation_network = segmentation_network
        for parameter in self.segmentation_network.parameters():
            parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dimension, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.segmentation_network.eval()
        return self

    @staticmethod
    def _full_resolution_logits(segmentation_logits: Any) -> torch.Tensor:
        if isinstance(segmentation_logits, (tuple, list)):
            segmentation_logits = segmentation_logits[0]
        if not isinstance(segmentation_logits, torch.Tensor):
            raise TypeError("nnU-Net decoder did not return a tensor.")
        return segmentation_logits

    def _segmentation_maps(
        self, segmentation_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return soft foreground probability and foreground confidence maps."""
        has_regions = bool(
            getattr(self.label_manager, "has_regions", False)
        )
        if has_regions:
            foreground_channels = torch.sigmoid(segmentation_logits)
            foreground_probability = foreground_channels.amax(dim=1, keepdim=True)
            foreground_confidence = foreground_probability
        elif segmentation_logits.shape[1] > 1:
            probabilities = torch.softmax(segmentation_logits, dim=1)
            foreground_channels = probabilities[:, 1:]
            foreground_probability = 1.0 - probabilities[:, :1]
            foreground_confidence = foreground_channels.amax(dim=1, keepdim=True)
        else:
            foreground_probability = torch.sigmoid(segmentation_logits)
            foreground_confidence = foreground_probability
        return foreground_probability, foreground_confidence

    def _build_features(
        self,
        encoder_features: Sequence[torch.Tensor],
        segmentation_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        bottleneck = encoder_features[-1]
        spatial_axes = tuple(range(2, bottleneck.ndim))
        global_feature = bottleneck.mean(dim=spatial_axes)
        if self.feature_mode == "A0":
            return global_feature
        if segmentation_logits is None:
            raise RuntimeError(f"{self.feature_mode} requires segmentation logits.")

        foreground_probability, foreground_confidence = self._segmentation_maps(
            segmentation_logits
        )
        roi_weight = nn.functional.interpolate(
            foreground_probability.float(),
            size=tuple(bottleneck.shape[2:]),
            mode="area",
        ).to(dtype=bottleneck.dtype)
        roi_denominator = roi_weight.sum(dim=spatial_axes).clamp_min(1e-4)
        roi_feature = (bottleneck * roi_weight).sum(dim=spatial_axes)
        roi_feature = roi_feature / roi_denominator
        parts = [global_feature, roi_feature]

        if self.feature_mode in {"A2", "A3"}:
            volume = foreground_probability.float().mean(
                dim=tuple(range(2, foreground_probability.ndim))
            )
            parts.append(volume.to(dtype=global_feature.dtype))

        if self.feature_mode == "A3":
            probability = foreground_probability.float()
            confidence = foreground_confidence.float()
            confidence_axes = tuple(range(2, probability.ndim))
            weighted_confidence = (probability * confidence).sum(
                dim=confidence_axes
            ) / probability.sum(dim=confidence_axes).clamp_min(1e-6)
            parts.append(weighted_confidence.to(dtype=global_feature.dtype))

        features = torch.cat(parts, dim=1)
        if features.shape[1] != self.feature_dimension:
            raise RuntimeError(
                f"Built {features.shape[1]} features for {self.feature_mode}, "
                f"but classifier expects {self.feature_dimension}."
            )
        return features

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            encoder_features = self.segmentation_network.encoder(image)
            segmentation_logits = None
            if self.feature_mode != "A0":
                segmentation_logits = self._full_resolution_logits(
                    self.segmentation_network.decoder(encoder_features)
                )
            return self._build_features(encoder_features, segmentation_logits)

    def forward(self, image: torch.Tensor, return_seg: bool = False):
        with torch.no_grad():
            encoder_features = self.segmentation_network.encoder(image)
            segmentation_logits = None
            if return_seg or self.feature_mode != "A0":
                segmentation_logits = self._full_resolution_logits(
                    self.segmentation_network.decoder(encoder_features)
                )
            features = self._build_features(encoder_features, segmentation_logits)
        return segmentation_logits if return_seg else None, self.classifier(features)


class A0PreprocessedDataset(Dataset):
    """Loads nnU-Net preprocessed images and returns classification labels."""

    def __init__(
        self,
        identifiers: Sequence[str],
        preprocessed_folder: str | Path,
        patch_size: Sequence[int],
        label_by_identifier: dict[str, Any],
        label_to_index: dict[Any, int],
        training: bool,
        flip_probability: float = 0.5,
    ) -> None:
        self.identifiers = list(identifiers)
        self.label_by_identifier = label_by_identifier
        self.label_to_index = label_to_index
        dataset_class = infer_dataset_class(str(preprocessed_folder))
        self.nnunet_dataset = dataset_class(
            str(preprocessed_folder), self.identifiers
        )

        transforms = [
            EnsureTyped(keys=["image"], dtype=torch.float32),
            SpatialPadd(keys=["image"], spatial_size=tuple(patch_size)),
        ]
        if training:
            transforms.append(
                RandSpatialCropd(
                    keys=["image"], roi_size=tuple(patch_size), random_size=False
                )
            )
            transforms.extend(
                RandFlipd(keys=["image"], spatial_axis=axis, prob=flip_probability)
                for axis in range(len(tuple(patch_size)))
            )
        else:
            transforms.append(
                CenterSpatialCropd(keys=["image"], roi_size=tuple(patch_size))
            )
        self.transform = Compose(transforms)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int):
        identifier = self.identifiers[index]
        loaded_case = self.nnunet_dataset.load_case(identifier)
        image = np.asarray(loaded_case[0], dtype=np.float32).copy()
        image = self.transform({"image": image})["image"]
        label = self.label_to_index[self.label_by_identifier[identifier]]
        return image, torch.tensor(label, dtype=torch.long)


class PartialUnfreezeSegClassifier(nn.Module):
    """Classifier that fine-tunes only the final nnU-Net encoder stages."""

    def __init__(
        self,
        segmentation_network: nn.Module,
        feature_channels: int,
        num_classes: int,
        hidden_channels: int = 256,
        dropout: float = 0.30,
        unfreeze_last_n_stages: int = 1,
        spatial_dimensions: int = 3,
    ) -> None:
        super().__init__()
        if unfreeze_last_n_stages < 1:
            raise ValueError("unfreeze_last_n_stages must be at least 1.")

        self.unfreeze_last_n_stages = int(unfreeze_last_n_stages)
        self.segmentation_network = segmentation_network
        for parameter in self.segmentation_network.parameters():
            parameter.requires_grad = False

        encoder = self.segmentation_network.encoder
        stages = list(getattr(encoder, "stages", encoder.children()))
        if not stages:
            raise AttributeError(
                "Could not find encoder stages in the loaded nnU-Net network."
            )
        if unfreeze_last_n_stages > len(stages):
            raise ValueError(
                f"Requested {unfreeze_last_n_stages} encoder stages, but the "
                f"network contains only {len(stages)}."
            )

        self.encoder_stage_count = len(stages)
        self.unfrozen_stage_indices = tuple(
            range(
                self.encoder_stage_count - self.unfreeze_last_n_stages,
                self.encoder_stage_count,
            )
        )
        self.unfrozen_stages = tuple(stages[-self.unfreeze_last_n_stages :])
        for stage in self.unfrozen_stages:
            for parameter in stage.parameters():
                parameter.requires_grad = True

        if spatial_dimensions == 3:
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif spatial_dimensions == 2:
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError("Only 2D and 3D nnU-Net configurations are supported.")

        self.classifier = nn.Sequential(
            nn.Linear(feature_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.segmentation_network.eval()
        self.classifier.train(mode)
        for stage in self.unfrozen_stages:
            stage.train(mode)
        return self

    def forward(self, image: torch.Tensor, return_seg: bool = False):
        encoder_features = self.segmentation_network.encoder(image)
        segmentation_logits = (
            self.segmentation_network.decoder(encoder_features)
            if return_seg
            else None
        )
        if isinstance(segmentation_logits, (tuple, list)):
            # nnU-Net orders deep-supervision outputs from highest to lowest
            # resolution. A5 uses the full-resolution output.
            segmentation_logits = segmentation_logits[0]
        pooled = self.pool(encoder_features[-1]).flatten(start_dim=1)
        return segmentation_logits, self.classifier(pooled)


def verify_partial_unfreeze_configuration(
    setup: A0Setup,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Return a short summary of the A4/A5 trainable parameters."""
    if not isinstance(setup.model, PartialUnfreezeSegClassifier):
        raise TypeError("setup.model must be a PartialUnfreezeSegClassifier.")
    model = setup.model
    trainable_backbone_parameters = [
        parameter
        for parameter in model.segmentation_network.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    return {
        "encoder_stage_count": model.encoder_stage_count,
        "unfreeze_last_n_stages": model.unfreeze_last_n_stages,
        "unfrozen_stage_indices": list(model.unfrozen_stage_indices),
        "trainable_backbone_tensors": len(trainable_backbone_parameters),
        "trainable_backbone_parameters": int(
            sum(parameter.numel() for parameter in trainable_backbone_parameters)
        ),
        "trainable_classifier_tensors": len(classifier_parameters),
        "trainable_classifier_parameters": int(
            sum(parameter.numel() for parameter in classifier_parameters)
        ),
        "optimizer_groups": len(optimizer.param_groups) if optimizer else 0,
    }


def _unwrapped_network(network: nn.Module) -> nn.Module:
    """Return the underlying module when nnU-Net used torch.compile."""
    return getattr(network, "_orig_mod", network)


def install_finetuned_segmentation_weights_in_predictor(
    setup: A0Setup,
) -> dict[str, Any]:
    """Make nnU-Net sliding-window inference use current A4/A5 weights.

    nnUNetPredictor stores fold weights in ``list_of_parameters`` and reloads
    one of those states immediately before each sliding-window prediction.
    Updating only ``predictor.network`` is therefore insufficient: the original
    pretrained fold state would overwrite the fine-tuned encoder. This function
    replaces that retained list with one complete state built from the current
    fine-tuned segmentation network.
    """
    if not isinstance(setup.model, PartialUnfreezeSegClassifier):
        return {
            "installed": False,
            "reason": "frozen_backbone_experiment",
        }
    model_network = _unwrapped_network(setup.model.segmentation_network)
    predictor_network = _unwrapped_network(setup.predictor.network)
    if model_network is not predictor_network:
        raise RuntimeError(
            "Cannot synchronize predictor weights because model and predictor "
            "do not share the same underlying segmentation network."
        )

    # A full state is required because nnU-Net uses strict=True internally when
    # it reloads each entry in list_of_parameters.
    predictor_state = {
        name: value.detach().cpu().clone()
        for name, value in model_network.state_dict().items()
    }
    predictor_network.load_state_dict(predictor_state, strict=True)
    setup.predictor.list_of_parameters = [predictor_state]

    return {
        "installed": True,
        "parameter_sets": len(setup.predictor.list_of_parameters),
        "state_tensors": len(predictor_state),
        "unfreeze_last_n_stages": setup.model.unfreeze_last_n_stages,
    }


class FineTuningPreprocessedDataset(Dataset):
    """Loads aligned image, class label, and optional segmentation patches."""

    def __init__(
        self,
        identifiers: Sequence[str],
        preprocessed_folder: str | Path,
        patch_size: Sequence[int],
        label_by_identifier: dict[str, Any],
        label_to_index: dict[Any, int],
        training: bool,
        include_segmentation: bool,
        flip_probability: float = 0.5,
    ) -> None:
        self.identifiers = list(identifiers)
        self.label_by_identifier = label_by_identifier
        self.label_to_index = label_to_index
        self.include_segmentation = include_segmentation
        dataset_class = infer_dataset_class(str(preprocessed_folder))
        self.nnunet_dataset = dataset_class(
            str(preprocessed_folder), self.identifiers
        )

        keys = ["image", "seg"] if include_segmentation else ["image"]
        transforms = [
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
        if include_segmentation:
            transforms.append(EnsureTyped(keys=["seg"], dtype=torch.long))
        transforms.append(SpatialPadd(keys=keys, spatial_size=tuple(patch_size)))
        if training:
            transforms.append(
                RandSpatialCropd(
                    keys=keys, roi_size=tuple(patch_size), random_size=False
                )
            )
            transforms.extend(
                RandFlipd(keys=keys, spatial_axis=axis, prob=flip_probability)
                for axis in range(len(tuple(patch_size)))
            )
        else:
            transforms.append(
                CenterSpatialCropd(keys=keys, roi_size=tuple(patch_size))
            )
        self.transform = Compose(transforms)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int):
        identifier = self.identifiers[index]
        loaded_case = self.nnunet_dataset.load_case(identifier)
        sample: dict[str, Any] = {
            "image": np.asarray(loaded_case[0], dtype=np.float32).copy()
        }
        if self.include_segmentation:
            if loaded_case[1] is None:
                raise ValueError(
                    f"No preprocessed segmentation was found for {identifier}."
                )
            sample["seg"] = np.asarray(loaded_case[1]).copy()
        transformed = self.transform(sample)
        label = torch.tensor(
            self.label_to_index[self.label_by_identifier[identifier]],
            dtype=torch.long,
        )
        if self.include_segmentation:
            return transformed["image"], label, transformed["seg"].long()
        return transformed["image"], label


def configure_environment(
    project_root: str | Path,
    raw_root: str | Path,
    preprocessed_root: str | Path | None = None,
    results_root: str | Path | None = None,
    disable_wandb: bool = True,
    disable_compile: bool = True,
) -> dict[str, str]:
    """Configure paths required by nnU-Net and return their values."""
    project_root = Path(project_root)
    values = {
        "nnUNet_raw": str(Path(raw_root)),
        "nnUNet_preprocessed": str(
            Path(preprocessed_root or project_root / "nnUNet_preprocessed")
        ),
        "nnUNet_results": str(Path(results_root or project_root / "nnUNet_results")),
    }
    os.environ.update(values)
    if disable_wandb:
        os.environ.update(WANDB_MODE="disabled", WANDB_SILENT="true")
    if disable_compile:
        os.environ["nnUNet_compile"] = "false"
    return values


def find_segmentation_model(
    results_root: str | Path,
    dataset_name: str,
    segmentation_fold: str | int = "all",
    checkpoint_names: Sequence[str] = (
        "checkpoint_best.pth",
    ),
    model_name: str | None = None,
) -> tuple[Path, str, Path]:
    """Resolve an nnU-Net model folder and checkpoint."""
    dataset_root = Path(results_root) / dataset_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"No nnU-Net results directory exists for {dataset_name}: "
            f"{dataset_root}. Run train_seg first."
        )
    if model_name is not None:
        candidates = [dataset_root / model_name]
    else:
        candidates = [
            folder
            for folder in dataset_root.iterdir()
            if folder.is_dir()
            and (folder / "plans.json").exists()
            and (folder / "dataset.json").exists()
        ]
        candidates.sort(
            key=lambda folder: (
                "cls" in folder.name.lower() or "mtl" in folder.name.lower(),
                "resenc" not in folder.name.lower(),
                folder.name,
            )
        )

    fold_folder = f"fold_{segmentation_fold}"
    for folder in candidates:
        for checkpoint_name in checkpoint_names:
            checkpoint_path = folder / fold_folder / checkpoint_name
            if checkpoint_path.is_file():
                return folder, checkpoint_name, checkpoint_path
    raise FileNotFoundError(
        f"No requested checkpoint was found for {dataset_name}/{fold_folder}."
    )


def setup_a0(
    project_root: str | Path,
    dataset_name: str,
    segmentation_fold: str | int = "all",
    classification_fold: int = 0,
    device: str | torch.device | None = None,
    model_name: str | None = None,
    checkpoint_names: Sequence[str] = (
        "checkpoint_best.pth",
    ),
    tile_step_size: float = 0.5,
    use_gaussian: bool = False,
    use_mirroring: bool = False,
    perform_everything_on_device: bool = True,
    hidden_channels: int = 256,
    dropout: float = 0.30,
    feature_mode: str = "A0",
    label_column: str = "label",
    identifier_column: str | None = None,
    task_name: str | None = None,
    split_json_path: str | Path | None = None,
) -> A0Setup:
    """Load pretrained segmentation network and construct an A0-A3 model."""
    project_root = Path(project_root)
    feature_mode = _normalise_feature_mode(feature_mode)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dir, checkpoint_name, checkpoint_path = find_segmentation_model(
        os.environ["nnUNet_results"],
        dataset_name,
        segmentation_fold,
        checkpoint_names,
        model_name,
    )

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=perform_everything_on_device,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_dir),
        use_folds=(segmentation_fold,),
        checkpoint_name=checkpoint_name,
    )

    dataset_root = Path(os.environ["nnUNet_preprocessed"]) / dataset_name
    raw_dataset_folder = Path(os.environ["nnUNet_raw"]) / dataset_name
    dataset_json_path = raw_dataset_folder / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_json_path}")
    with dataset_json_path.open(encoding="utf-8") as file:
        raw_dataset_json = json.load(file)
    cls_filename = raw_dataset_json.get(
        "classification_labels_file", "cls_data.csv"
    )
    cls_csv = raw_dataset_folder / cls_filename
    if not cls_csv.is_file():
        raise FileNotFoundError(
            f"Missing classification labels in the raw dataset: {cls_csv}"
        )
    preprocessed_folder = dataset_root / predictor.configuration_manager.data_identifier
    split_json = Path(split_json_path or dataset_root / "splits_final.json")
    if not split_json.is_file():
        raise FileNotFoundError(
            f"Missing classification split file: {split_json}. Run train_cls "
            "through the AutoMSC command so it can create the all-cases split."
        )
    cls_df = pd.read_csv(cls_csv)
    if identifier_column is None:
        for candidate in ("case_id", "identifier", "case", "id"):
            if candidate in cls_df.columns:
                identifier_column = candidate
                break
        else:
            identifier_column = str(cls_df.columns[0])
    if identifier_column not in cls_df.columns:
        raise KeyError(
            f"Identifier column {identifier_column!r} is not present in "
            f"{cls_csv}; available columns: {cls_df.columns.tolist()}"
        )
    if label_column not in cls_df.columns:
        raise KeyError(
            f"Label column {label_column!r} is not present in {cls_csv}; "
            f"available columns: {cls_df.columns.tolist()}"
        )
    cls_df = cls_df[[identifier_column, label_column]].dropna().copy()
    cls_df[identifier_column] = cls_df[identifier_column].astype(str)
    if cls_df[identifier_column].duplicated().any():
        duplicates = cls_df.loc[
            cls_df[identifier_column].duplicated(), identifier_column
        ].tolist()
        raise ValueError(
            f"Duplicate case identifiers in {cls_csv}: {duplicates[:5]}"
        )
    label_values = cls_df[label_column].dropna().unique().tolist()
    try:
        label_values = sorted(label_values)
    except TypeError:
        label_values = sorted(label_values, key=lambda value: str(value))
    if len(label_values) < 2:
        raise ValueError(
            f"Classification task {label_column!r} has fewer than two classes."
        )
    label_to_index = {label: index for index, label in enumerate(label_values)}
    task_name = str(task_name or label_column)
    feature_channels = int(
        predictor.configuration_manager.network_arch_init_kwargs[
            "features_per_stage"
        ][-1]
    )
    patch_size = tuple(int(v) for v in predictor.configuration_manager.patch_size)
    model = A0FrozenSegClassifier(
        predictor.network.to(device).eval(),
        feature_channels,
        len(label_values),
        hidden_channels,
        dropout,
        feature_mode=feature_mode,
        label_manager=predictor.label_manager,
    ).to(device)

    return A0Setup(
        project_root=project_root,
        raw_dataset_folder=raw_dataset_folder,
        dataset_name=dataset_name,
        task_name=task_name,
        identifier_column=identifier_column,
        label_column=label_column,
        segmentation_fold=segmentation_fold,
        classification_fold=classification_fold,
        device=device,
        predictor=predictor,
        model=model,
        segmentation_model_dir=model_dir,
        segmentation_checkpoint=checkpoint_name,
        segmentation_checkpoint_path=checkpoint_path,
        preprocessed_folder=preprocessed_folder,
        cls_csv=cls_csv,
        split_json=split_json,
        cls_df=cls_df,
        label_values=label_values,
        label_to_index=label_to_index,
        num_classes=len(label_values),
        feature_channels=feature_channels,
        feature_mode=feature_mode,
        feature_dimension=model.feature_dimension,
        patch_size=patch_size,
    )


def setup_partial_unfreeze(
    project_root: str | Path,
    dataset_name: str,
    segmentation_fold: str | int = "all",
    classification_fold: int = 0,
    device: str | torch.device | None = None,
    model_name: str | None = None,
    checkpoint_names: Sequence[str] = (
        "checkpoint_best.pth",
    ),
    tile_step_size: float = 0.5,
    use_gaussian: bool = False,
    use_mirroring: bool = False,
    perform_everything_on_device: bool = True,
    hidden_channels: int = 256,
    dropout: float = 0.30,
    unfreeze_last_n_stages: int = 1,
    a0_checkpoint_path: str | Path | None = None,
    initial_finetuned_checkpoint_path: str | Path | None = None,
) -> A0Setup:
    """Construct the A4/A5 model from the pretrained segmentation network.

    ``a0_checkpoint_path`` optionally initializes the classification head from
    A0. ``initial_finetuned_checkpoint_path`` optionally resumes or warm-starts
    from an A4/A5 checkpoint.
    """
    setup = setup_a0(
        project_root=project_root,
        dataset_name=dataset_name,
        segmentation_fold=segmentation_fold,
        classification_fold=classification_fold,
        device=device,
        model_name=model_name,
        checkpoint_names=checkpoint_names,
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=perform_everything_on_device,
        hidden_channels=hidden_channels,
        dropout=dropout,
    )
    setup.model = PartialUnfreezeSegClassifier(
        segmentation_network=setup.predictor.network,
        feature_channels=setup.feature_channels,
        num_classes=setup.num_classes,
        hidden_channels=hidden_channels,
        dropout=dropout,
        unfreeze_last_n_stages=unfreeze_last_n_stages,
        spatial_dimensions=len(setup.patch_size),
    ).to(setup.device)

    if a0_checkpoint_path is not None:
        checkpoint = torch.load(
            a0_checkpoint_path, map_location=setup.device, weights_only=False
        )
        if "classifier_state_dict" not in checkpoint:
            raise KeyError(
                "The A0 checkpoint does not contain classifier_state_dict."
            )
        checkpoint_mode = _normalise_feature_mode(
            checkpoint.get("feature_mode", "A0")
        )
        if checkpoint_mode != "A0":
            raise ValueError(
                "A4/A5 must initialize from an A0 global-feature checkpoint; "
                f"received {checkpoint_mode}."
            )
        setup.model.classifier.load_state_dict(checkpoint["classifier_state_dict"])

    if initial_finetuned_checkpoint_path is not None:
        load_classifier_checkpoint(setup, initial_finetuned_checkpoint_path)

    return setup


def build_classification_data(
    setup: A0Setup,
    image_batch_size: int = 1,
    num_workers: int = 0,
    flip_probability: float = 0.5,
    pin_memory: bool | None = None,
    use_class_weights: bool = True,
    class_weight_strategy: str = "auto",
    label_smoothing: float | None = None,
) -> A0Data:
    """Create fold-specific image datasets, loaders, and class weights."""
    with setup.split_json.open() as file:
        split = json.load(file)[setup.classification_fold]
    train_ids = [str(v) for v in split["train"]]
    val_ids = [str(v) for v in split["val"]]
    label_by_id = dict(
        zip(
            setup.cls_df[setup.identifier_column].astype(str),
            setup.cls_df[setup.label_column],
        )
    )
    missing = sorted((set(train_ids) | set(val_ids)) - set(label_by_id))
    if missing:
        raise ValueError(
            f"{setup.cls_csv} has no {setup.label_column!r} label for "
            f"{len(missing)} preprocessed cases, including {missing[:5]}."
        )
    common = dict(
        preprocessed_folder=setup.preprocessed_folder,
        patch_size=setup.patch_size,
        label_by_identifier=label_by_id,
        label_to_index=setup.label_to_index,
        flip_probability=flip_probability,
    )
    train_dataset = A0PreprocessedDataset(train_ids, training=True, **common)
    val_dataset = A0PreprocessedDataset(val_ids, training=False, **common)
    pin_memory = setup.device.type == "cuda" if pin_memory is None else pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=image_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_labels = [setup.label_to_index[label_by_id[v]] for v in train_ids]
    counts = np.bincount(train_labels, minlength=setup.num_classes)
    validation_labels = [
        setup.label_to_index[label_by_id[value]] for value in val_ids
    ]
    validation_counts = np.bincount(
        validation_labels,
        minlength=setup.num_classes,
    )
    if np.any(validation_counts == 0):
        warnings.warn(
            "Validation fold does not contain every classification class. "
            "Balanced accuracy will average recall only over classes present "
            f"in the ground truth. Validation counts={validation_counts.tolist()}.",
            RuntimeWarning,
            stacklevel=2,
        )
    requested_strategy = class_weight_strategy if use_class_weights else "none"
    weights, resolved_strategy = automatic_class_weights(
        counts, requested_strategy
    )
    resolved_smoothing = (
        automatic_label_smoothing(counts)
        if label_smoothing is None
        else float(label_smoothing)
    )
    if not 0.0 <= resolved_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    return A0Data(
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        train_ids,
        val_ids,
        label_by_id,
        counts,
        torch.tensor(weights, dtype=torch.float32, device=setup.device),
        resolved_strategy,
        resolved_smoothing,
    )


def build_finetuning_data(
    setup: A0Setup,
    include_segmentation: bool,
    image_batch_size: int = 1,
    num_workers: int = 0,
    flip_probability: float = 0.5,
    pin_memory: bool | None = None,
    use_class_weights: bool = True,
    class_weight_strategy: str = "auto",
    label_smoothing: float | None = None,
) -> A0Data:
    """Create image loaders for A4 or aligned image/segmentation loaders for A5."""
    with setup.split_json.open() as file:
        split = json.load(file)[setup.classification_fold]
    train_ids = [str(value) for value in split["train"]]
    val_ids = [str(value) for value in split["val"]]
    label_by_id = dict(
        zip(
            setup.cls_df[setup.identifier_column].astype(str),
            setup.cls_df[setup.label_column],
        )
    )
    missing = sorted((set(train_ids) | set(val_ids)) - set(label_by_id))
    if missing:
        raise ValueError(
            f"{setup.cls_csv} has no {setup.label_column!r} label for "
            f"{len(missing)} preprocessed cases, including {missing[:5]}."
        )
    common = dict(
        preprocessed_folder=setup.preprocessed_folder,
        patch_size=setup.patch_size,
        label_by_identifier=label_by_id,
        label_to_index=setup.label_to_index,
        include_segmentation=include_segmentation,
        flip_probability=flip_probability,
    )
    train_dataset = FineTuningPreprocessedDataset(
        train_ids, training=True, **common
    )
    val_dataset = FineTuningPreprocessedDataset(val_ids, training=False, **common)
    pin_memory = setup.device.type == "cuda" if pin_memory is None else pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=image_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=image_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_labels = [setup.label_to_index[label_by_id[value]] for value in train_ids]
    counts = np.bincount(train_labels, minlength=setup.num_classes)
    validation_labels = [
        setup.label_to_index[label_by_id[value]] for value in val_ids
    ]
    validation_counts = np.bincount(
        validation_labels,
        minlength=setup.num_classes,
    )
    if np.any(validation_counts == 0):
        warnings.warn(
            "Validation fold does not contain every classification class. "
            "Balanced accuracy will average recall only over classes present "
            f"in the ground truth. Validation counts={validation_counts.tolist()}.",
            RuntimeWarning,
            stacklevel=2,
        )
    requested_strategy = class_weight_strategy if use_class_weights else "none"
    weights, resolved_strategy = automatic_class_weights(
        counts, requested_strategy
    )
    resolved_smoothing = (
        automatic_label_smoothing(counts)
        if label_smoothing is None
        else float(label_smoothing)
    )
    if not 0.0 <= resolved_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    return A0Data(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        train_identifiers=train_ids,
        val_identifiers=val_ids,
        label_by_identifier=label_by_id,
        class_counts=counts,
        class_weights=torch.tensor(
            weights, dtype=torch.float32, device=setup.device
        ),
        class_weight_strategy=resolved_strategy,
        label_smoothing=resolved_smoothing,
    )


def _extract_feature_tensors(
    model: A0FrozenSegClassifier,
    loader: DataLoader,
    device: torch.device,
    variants: int,
    description: str,
) -> dict[str, Any]:
    features, labels = [], []
    model.eval()
    start = time.perf_counter()
    for variant in range(variants):
        for images, targets in tqdm(
            loader, desc=f"{description} {variant + 1}/{variants}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                batch_features = model.extract_features(images)
            features.append(batch_features.float().cpu())
            labels.append(targets.long().cpu())
    tensor = torch.cat(features)
    return {
        "features": tensor,
        "labels": torch.cat(labels),
        "number_of_variants": variants,
        "feature_dimension": int(tensor.shape[1]),
        "number_of_samples": int(tensor.shape[0]),
        "extraction_seconds": float(time.perf_counter() - start),
    }


def cache_frozen_features(
    setup: A0Setup,
    data: A0Data,
    train_variants: int = 2,
    feature_batch_size: int = 64,
    extraction_batch_size: int = 1,
    extraction_num_workers: int = 0,
    cache_dir: str | Path | None = None,
    reuse_cache: bool = True,
    experiment_name: str | None = None,
) -> FeatureData:
    """Build and cache the selected frozen A0-A3 feature representation."""
    experiment_name = experiment_name or FROZEN_EXPERIMENT_NAMES[
        setup.feature_mode
    ]
    cache_dir = Path(
        cache_dir
        or setup.project_root
        / "experiments"
        / experiment_name
        / setup.dataset_name
        / setup.task_name
        / f"fold_{setup.classification_fold}/feature_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = (
        cache_dir / f"train_features_{train_variants}_augmented_plus_center.pt"
    )
    val_cache = cache_dir / "val_features.pt"
    pin = setup.device.type == "cuda"

    def extraction_loader(dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=extraction_batch_size,
            shuffle=False,
            num_workers=extraction_num_workers,
            pin_memory=pin,
        )

    train_cache_matches = False
    if reuse_cache and train_cache.is_file():
        train_meta = torch.load(train_cache, map_location="cpu", weights_only=False)
        train_cache_matches = list(train_meta.get("identifiers", [])) == list(
            data.train_identifiers
        )
    if not train_cache_matches:
        train_meta = _extract_feature_tensors(
            setup.model,
            extraction_loader(data.train_dataset),
            setup.device,
            train_variants,
            "Training features",
        )
        center_train_dataset = A0PreprocessedDataset(
            data.train_identifiers,
            preprocessed_folder=setup.preprocessed_folder,
            patch_size=setup.patch_size,
            label_by_identifier=data.label_by_identifier,
            label_to_index=setup.label_to_index,
            training=False,
        )
        center_meta = _extract_feature_tensors(
            setup.model,
            extraction_loader(center_train_dataset),
            setup.device,
            1,
            "Center-crop training features",
        )
        train_meta["features"] = torch.cat(
            (train_meta["features"], center_meta["features"])
        )
        train_meta["labels"] = torch.cat(
            (train_meta["labels"], center_meta["labels"])
        )
        train_meta["number_of_variants"] = train_variants + 1
        train_meta["number_of_samples"] = int(train_meta["features"].shape[0])
        train_meta["includes_center_crop"] = True
        train_meta["extraction_seconds"] += center_meta["extraction_seconds"]
        train_meta.update(
            dataset=setup.dataset_name,
            fold=setup.classification_fold,
            feature_mode=setup.feature_mode,
            task_name=setup.task_name,
            label_column=setup.label_column,
            label_values=setup.label_values,
            identifiers=list(data.train_identifiers),
        )
        torch.save(train_meta, train_cache)

    val_cache_matches = False
    if reuse_cache and val_cache.is_file():
        val_meta = torch.load(val_cache, map_location="cpu", weights_only=False)
        val_cache_matches = list(val_meta.get("identifiers", [])) == list(
            data.val_identifiers
        )
    if not val_cache_matches:
        val_meta = _extract_feature_tensors(
            setup.model,
            extraction_loader(data.val_dataset),
            setup.device,
            1,
            "Validation features",
        )
        val_meta.update(
            dataset=setup.dataset_name,
            fold=setup.classification_fold,
            feature_mode=setup.feature_mode,
            task_name=setup.task_name,
            label_column=setup.label_column,
            label_values=setup.label_values,
            identifiers=list(data.val_identifiers),
        )
        torch.save(val_meta, val_cache)

    for cache_path, metadata, identifiers in (
        (train_cache, train_meta, data.train_identifiers),
        (val_cache, val_meta, data.val_identifiers),
    ):
        cached_mode = _normalise_feature_mode(metadata.get("feature_mode", "A0"))
        cached_dimension = int(metadata["feature_dimension"])
        if (
            cached_mode != setup.feature_mode
            or cached_dimension != setup.feature_dimension
            or metadata.get("task_name") != setup.task_name
            or metadata.get("label_column") != setup.label_column
            or list(metadata.get("label_values", [])) != list(setup.label_values)
            or list(metadata.get("identifiers", [])) != list(identifiers)
        ):
            raise ValueError(
                f"Incompatible feature cache at {cache_path}: mode={cached_mode}, "
                f"dimension={cached_dimension}, task={metadata.get('task_name')!r}; "
                f"expected mode={setup.feature_mode}, "
                f"dimension={setup.feature_dimension}, task={setup.task_name!r}. "
                "Set reuse_cache=False."
            )

    def feature_loader(metadata: dict[str, Any], shuffle: bool) -> DataLoader:
        dataset = TensorDataset(metadata["features"], metadata["labels"])
        return DataLoader(
            dataset,
            batch_size=min(feature_batch_size, len(dataset)),
            shuffle=shuffle,
            num_workers=0,
            pin_memory=pin,
        )

    gc.collect()
    if setup.device.type == "cuda":
        torch.cuda.empty_cache()
    return FeatureData(
        feature_loader(train_meta, True),
        feature_loader(val_meta, False),
        train_cache,
        val_cache,
        train_meta,
        val_meta,
    )


def _classification_epoch(
    classifier: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
) -> dict[str, float]:
    training = optimizer is not None
    classifier.train(training)
    loss_sum, count, targets, predictions, probabilities = 0.0, 0, [], [], []
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = classifier(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.shape[0]
            count += labels.shape[0]
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(1).detach().cpu().tolist())
            probabilities.extend(
                torch.softmax(logits.detach().float(), dim=1).cpu().tolist()
            )
    metrics = _classification_metrics(
        targets,
        predictions,
        num_classes,
        probabilities=np.asarray(probabilities, dtype=float),
    )
    return {
        "loss": loss_sum / max(count, 1),
        **metrics,
    }


def train_cached_classifier(
    setup: A0Setup,
    data: A0Data,
    features: FeatureData,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    save_dir: str | Path | None = None,
    checkpoint_metric: str = "val_loss",
    selection_window: int = 1,
    minimum_epochs: int = 1,
    early_stopping_patience: int | None = None,
    verbose: bool = True,
    experiment_name: str | None = None,
) -> TrainingResult:
    """Train an A0-A3 classification head using cached frozen features."""
    experiment_name = experiment_name or FROZEN_EXPERIMENT_NAMES[
        setup.feature_mode
    ]
    save_dir = Path(
        save_dir
        or setup.project_root
        / "experiments"
        / experiment_name
        / setup.dataset_name
        / setup.task_name
        / f"fold_{setup.classification_fold}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "classifier_best.pth"
    last_path = save_dir / "classifier_last.pth"
    history_path = save_dir / "training_history.csv"
    if selection_window < 1:
        raise ValueError("selection_window must be at least 1.")
    if minimum_epochs < 1:
        raise ValueError("minimum_epochs must be at least 1.")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive or None.")
    criterion = nn.CrossEntropyLoss(
        weight=data.class_weights,
        label_smoothing=data.label_smoothing,
    )
    classifier = setup.model.classifier.to(setup.device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    minimize = checkpoint_metric.endswith("loss")
    best_score = float("inf") if minimize else float("-inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = _classification_epoch(
            classifier,
            features.train_loader,
            criterion,
            setup.device,
            optimizer,
            setup.num_classes,
        )
        val_metrics = _classification_epoch(
            classifier,
            features.val_loader,
            criterion,
            setup.device,
            None,
            setup.num_classes,
        )
        row = {"epoch": epoch, "epoch_seconds": time.perf_counter() - start}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        if checkpoint_metric not in row:
            raise KeyError(
                f"Unknown checkpoint metric {checkpoint_metric!r}; "
                f"available metrics={sorted(row)}."
            )
        previous_window = (
            history[-(selection_window - 1) :] if selection_window > 1 else []
        )
        recent_scores = [
            previous[checkpoint_metric] for previous in previous_window
        ] + [row[checkpoint_metric]]
        finite_recent_scores = [
            float(value) for value in recent_scores if np.isfinite(value)
        ]
        score = (
            float(np.mean(finite_recent_scores))
            if finite_recent_scores
            else float("nan")
        )
        smoothed_metric = f"{checkpoint_metric}_smoothed"
        row[smoothed_metric] = score
        eligible = epoch >= min(minimum_epochs, epochs)
        improved = eligible and np.isfinite(score) and (
            score < best_score if minimize else score > best_score
        )
        row["saved_best"] = improved
        checkpoint = {
            "epoch": epoch,
            "classifier_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dataset": setup.dataset_name,
            "task_name": setup.task_name,
            "identifier_column": setup.identifier_column,
            "label_column": setup.label_column,
            "classification_fold": setup.classification_fold,
            "segmentation_model_dir": str(setup.segmentation_model_dir),
            "segmentation_checkpoint": setup.segmentation_checkpoint,
            "segmentation_fold": setup.segmentation_fold,
            "feature_channels": setup.feature_channels,
            "feature_mode": setup.feature_mode,
            "feature_dimension": features.train_metadata["feature_dimension"],
            "hidden_channels": int(classifier[0].out_features),
            "dropout": float(classifier[2].p),
            "patch_size": setup.patch_size,
            "label_values": setup.label_values,
            "label_to_index": setup.label_to_index,
            "class_counts": data.class_counts.tolist(),
            "class_weights": data.class_weights.detach().cpu().tolist(),
            "class_weight_strategy": data.class_weight_strategy,
            "label_smoothing": data.label_smoothing,
            **row,
        }
        torch.save(checkpoint, last_path)
        if improved:
            best_score = score
            epochs_without_improvement = 0
            checkpoint["best_checkpoint_score"] = best_score
            checkpoint["checkpoint_metric"] = smoothed_metric
            torch.save(checkpoint, best_path)
        elif eligible:
            epochs_without_improvement += 1
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if verbose:
            print(
                f"Epoch {epoch:03d} | train loss {row['train_loss']:.4f} | "
                f"val loss {row['val_loss']:.4f} | val bal acc "
                f"{row['val_balanced_accuracy']:.4f} | val AUROC "
                f"{row['val_auroc']:.4f} | val F1 {row['val_f1']:.4f}"
                + (" | saved best" if improved else "")
            )
        if (
            early_stopping_patience is not None
            and eligible
            and epochs_without_improvement >= early_stopping_patience
        ):
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}; {smoothed_metric} did not "
                    f"improve for {early_stopping_patience} epochs."
                )
            break
    return TrainingResult(
        best_path,
        last_path,
        history_path,
        pd.DataFrame(history),
        float(min(row["val_loss"] for row in history)),
    )


def train_full_data_classifier(
    setup: A0Setup,
    data: A0Data,
    features: FeatureData,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    averaging_fraction: float = 0.40,
    save_dir: str | Path | None = None,
    verbose: bool = True,
    experiment_name: str | None = None,
) -> TrainingResult:
    """Train one classifier on all cases and save one averaged final model.

    There is intentionally no best-epoch search: the deterministic monitoring
    view contains the same patients as training and must not be presented as
    held-out validation. A cosine learning-rate schedule and arithmetic weight
    average over the final trajectory provide a stable single checkpoint
    without generating a fold ensemble or selecting a noisy metric spike.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if not 0.0 < averaging_fraction <= 1.0:
        raise ValueError("averaging_fraction must be in (0, 1].")
    experiment_name = experiment_name or FROZEN_EXPERIMENT_NAMES[
        setup.feature_mode
    ]
    save_dir = Path(
        save_dir
        or setup.project_root
        / "experiments"
        / experiment_name
        / setup.dataset_name
        / setup.task_name
        / "full_data"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "classifier_final.pth"
    history_path = save_dir / "training_history.csv"
    criterion = nn.CrossEntropyLoss(
        weight=data.class_weights,
        label_smoothing=data.label_smoothing,
    )
    classifier = setup.model.classifier.to(setup.device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01,
    )
    averaging_start = max(1, int(np.floor(epochs * (1.0 - averaging_fraction))) + 1)
    averaged_state: dict[str, torch.Tensor] = {}
    averaged_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = _classification_epoch(
            classifier,
            features.train_loader,
            criterion,
            setup.device,
            optimizer,
            setup.num_classes,
        )
        monitor_metrics = _classification_epoch(
            classifier,
            features.val_loader,
            criterion,
            setup.device,
            None,
            setup.num_classes,
        )
        row = {
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - start,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_averaged": epoch >= averaging_start,
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update(
            {f"monitor_{key}": value for key, value in monitor_metrics.items()}
        )
        if epoch >= averaging_start:
            averaged_epochs += 1
            for name, value in classifier.state_dict().items():
                cpu_value = value.detach().cpu()
                if name not in averaged_state:
                    averaged_state[name] = cpu_value.clone()
                elif torch.is_floating_point(cpu_value):
                    averaged_state[name].add_(
                        (cpu_value - averaged_state[name]) / averaged_epochs
                    )
                else:
                    averaged_state[name] = cpu_value.clone()
        scheduler.step()
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if verbose:
            print(
                f"Epoch {epoch:03d} | train loss {row['train_loss']:.4f} | "
                f"monitor loss {row['monitor_loss']:.4f} | monitor bal acc "
                f"{row['monitor_balanced_accuracy']:.4f} | monitor AUROC "
                f"{row['monitor_auroc']:.4f}"
                + (" | averaging" if row["weight_averaged"] else "")
            )

    if not averaged_state or averaged_epochs == 0:
        raise RuntimeError("No classifier epochs were included in weight averaging.")
    classifier.load_state_dict(averaged_state)
    final_train_metrics = _classification_epoch(
        classifier,
        features.train_loader,
        criterion,
        setup.device,
        None,
        setup.num_classes,
    )
    final_monitor_metrics = _classification_epoch(
        classifier,
        features.val_loader,
        criterion,
        setup.device,
        None,
        setup.num_classes,
    )
    checkpoint = {
        "epoch": epochs,
        "classifier_state_dict": {
            name: value.detach().cpu()
            for name, value in classifier.state_dict().items()
        },
        "dataset": setup.dataset_name,
        "task_name": setup.task_name,
        "identifier_column": setup.identifier_column,
        "label_column": setup.label_column,
        "classification_fold": 0,
        "segmentation_model_dir": str(setup.segmentation_model_dir),
        "segmentation_checkpoint": setup.segmentation_checkpoint,
        "segmentation_fold": setup.segmentation_fold,
        "feature_channels": setup.feature_channels,
        "feature_mode": setup.feature_mode,
        "feature_dimension": features.train_metadata["feature_dimension"],
        "hidden_channels": int(classifier[0].out_features),
        "dropout": float(classifier[2].p),
        "patch_size": setup.patch_size,
        "label_values": setup.label_values,
        "label_to_index": setup.label_to_index,
        "class_counts": data.class_counts.tolist(),
        "class_weights": data.class_weights.detach().cpu().tolist(),
        "class_weight_strategy": data.class_weight_strategy,
        "label_smoothing": data.label_smoothing,
        "training_scope": "all_labelled_cases",
        "monitoring_scope": "all_labelled_cases_center_crop",
        "averaging_fraction": averaging_fraction,
        "weight_averaging_start_epoch": averaging_start,
        "averaged_epoch_count": averaged_epochs,
        "learning_rate_schedule": "cosine_annealing",
        "final_train_metrics": final_train_metrics,
        "final_monitor_metrics": final_monitor_metrics,
    }
    torch.save(checkpoint, model_path)
    if verbose:
        print(
            f"Final averaged model | loss {final_monitor_metrics['loss']:.4f} | "
            f"balanced accuracy "
            f"{final_monitor_metrics['balanced_accuracy']:.4f} | "
            f"AUROC {final_monitor_metrics['auroc']:.4f}"
        )
    return TrainingResult(
        model_path,
        model_path,
        history_path,
        pd.DataFrame(history),
        float(final_monitor_metrics["loss"]),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _dice_from_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    include_background: bool,
    smooth: float = 1e-5,
) -> torch.Tensor:
    mask = valid_mask.unsqueeze(1).to(dtype=probabilities.dtype)
    probabilities = probabilities * mask
    targets = targets.to(dtype=probabilities.dtype) * mask
    reduction_axes = (0, *range(2, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=reduction_axes)
    denominator = probabilities.sum(dim=reduction_axes) + targets.sum(
        dim=reduction_axes
    )
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    if not include_background and dice.numel() > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


def auxiliary_segmentation_loss(
    segmentation_logits: torch.Tensor,
    segmentation_target: torch.Tensor,
    label_manager: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute Dice+CE or Dice+BCE using the loaded nnU-Net label scheme."""
    if segmentation_target.ndim == segmentation_logits.ndim:
        integer_target = segmentation_target[:, 0].long()
    elif segmentation_target.ndim == segmentation_logits.ndim - 1:
        integer_target = segmentation_target.long()
    else:
        raise ValueError(
            "Segmentation target shape is incompatible with segmentation logits: "
            f"{tuple(segmentation_target.shape)} versus "
            f"{tuple(segmentation_logits.shape)}."
        )

    ignore_label = getattr(label_manager, "ignore_label", None)
    # nnU-Net preprocessed segmentations can contain -1 outside the valid
    # image/crop area. These voxels must not be passed to CE or one_hot.
    valid_mask = integer_target >= 0
    if ignore_label is not None:
        valid_mask &= integer_target != int(ignore_label)
    safe_target = integer_target.clone()
    safe_target[~valid_mask] = 0

    if bool(getattr(label_manager, "has_regions", False)):
        regions = getattr(label_manager, "foreground_regions", None)
        if regions is None:
            regions = getattr(label_manager, "regions", None)
        if regions is None:
            raise AttributeError("The region-based label definitions were not found.")
        regions = list(regions)
        if len(regions) != segmentation_logits.shape[1]:
            raise ValueError(
                "Region count does not match segmentation output channels: "
                f"{len(regions)} versus {segmentation_logits.shape[1]}."
            )
        region_targets = []
        for region in regions:
            region_labels = (
                list(region) if isinstance(region, (tuple, list, set)) else [region]
            )
            channel = torch.zeros_like(safe_target, dtype=torch.bool)
            for region_label in region_labels:
                channel |= safe_target == int(region_label)
            region_targets.append(channel)
        target_channels = torch.stack(region_targets, dim=1).to(
            dtype=segmentation_logits.dtype
        )
        bce_voxels = nn.functional.binary_cross_entropy_with_logits(
            segmentation_logits, target_channels, reduction="none"
        )
        bce_mask = valid_mask.unsqueeze(1).expand_as(bce_voxels)
        distribution_loss = _masked_mean(bce_voxels, bce_mask)
        dice_loss = _dice_from_probabilities(
            torch.sigmoid(segmentation_logits),
            target_channels,
            valid_mask,
            include_background=True,
        )
        distribution_name = "bce"
    else:
        number_of_channels = segmentation_logits.shape[1]
        # Also ignore any label that has no matching output channel.
        valid_mask &= integer_target < number_of_channels
        safe_target[~valid_mask] = 0
        ce_voxels = nn.functional.cross_entropy(
            segmentation_logits, safe_target, reduction="none"
        )
        distribution_loss = _masked_mean(ce_voxels, valid_mask)
        target_channels = nn.functional.one_hot(
            safe_target, num_classes=number_of_channels
        ).movedim(-1, 1)
        dice_loss = _dice_from_probabilities(
            torch.softmax(segmentation_logits, dim=1),
            target_channels,
            valid_mask,
            include_background=False,
        )
        distribution_name = "ce"

    total = distribution_loss + dice_loss
    return total, {
        distribution_name: float(distribution_loss.detach().cpu()),
        "dice_loss": float(dice_loss.detach().cpu()),
    }


def _finetuning_epoch(
    setup: A0Setup,
    loader: DataLoader,
    classification_criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    auxiliary_segmentation_weight: float,
    gradient_clip_norm: float | None,
) -> dict[str, float]:
    training = optimizer is not None
    setup.model.train(training)
    loss_sum = 0.0
    classification_loss_sum = 0.0
    segmentation_loss_sum = 0.0
    count = 0
    targets: list[int] = []
    predictions: list[int] = []
    grad_context = torch.enable_grad() if training else torch.inference_mode()

    with grad_context:
        for batch in loader:
            if len(batch) == 3:
                images, labels, segmentation_target = batch
                segmentation_target = segmentation_target.to(
                    setup.device, non_blocking=True
                )
            else:
                images, labels = batch
                segmentation_target = None
            images = images.to(setup.device, non_blocking=True)
            labels = labels.to(setup.device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=setup.device.type,
                enabled=setup.device.type == "cuda",
            ):
                segmentation_logits, classification_logits = setup.model(
                    images,
                    return_seg=auxiliary_segmentation_weight > 0,
                )
                classification_loss = classification_criterion(
                    classification_logits, labels
                ) / labels.shape[0]
                if auxiliary_segmentation_weight > 0:
                    if segmentation_target is None:
                        raise ValueError(
                            "A5 requires include_segmentation=True when building data."
                        )
                    segmentation_loss, _ = auxiliary_segmentation_loss(
                        segmentation_logits,
                        segmentation_target,
                        setup.predictor.label_manager,
                    )
                else:
                    segmentation_loss = classification_loss.new_zeros(())
                total_loss = classification_loss + (
                    auxiliary_segmentation_weight * segmentation_loss
                )

            if training:
                scaler.scale(total_loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in setup.model.parameters()
                            if parameter.requires_grad
                        ],
                        gradient_clip_norm,
                    )
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.shape[0]
            loss_sum += float(total_loss.detach().cpu()) * batch_size
            classification_loss_sum += (
                float(classification_loss.detach().cpu()) * batch_size
            )
            segmentation_loss_sum += (
                float(segmentation_loss.detach().cpu()) * batch_size
            )
            count += batch_size
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(
                classification_logits.argmax(1).detach().cpu().tolist()
            )

    metrics: dict[str, float] = {
        "loss": loss_sum / max(count, 1),
        "classification_loss": classification_loss_sum / max(count, 1),
        "segmentation_loss": segmentation_loss_sum / max(count, 1),
        **_classification_metrics(targets, predictions, setup.num_classes),
    }
    target_counts = np.bincount(targets, minlength=setup.num_classes)
    prediction_counts = np.bincount(predictions, minlength=setup.num_classes)
    for class_index in range(setup.num_classes):
        metrics[f"target_count_class_{class_index}"] = float(
            target_counts[class_index]
        )
        metrics[f"prediction_count_class_{class_index}"] = float(
            prediction_counts[class_index]
        )
    return metrics


def train_partial_unfreeze(
    setup: A0Setup,
    data: A0Data,
    experiment_name: str,
    epochs: int = 100,
    classifier_learning_rate: float = 1e-4,
    backbone_learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    auxiliary_segmentation_weight: float = 0.0,
    gradient_clip_norm: float | None = 12.0,
    save_dir: str | Path | None = None,
    checkpoint_metric: str = "val_balanced_accuracy",
    verbose: bool = True,
) -> TrainingResult:
    """Train A4 or A5 without cached features and without early stopping."""
    if not isinstance(setup.model, PartialUnfreezeSegClassifier):
        raise TypeError("setup.model must be created by setup_partial_unfreeze().")
    if auxiliary_segmentation_weight < 0:
        raise ValueError("auxiliary_segmentation_weight cannot be negative.")
    if classifier_learning_rate <= 0 or backbone_learning_rate <= 0:
        raise ValueError(
            "classifier_learning_rate and backbone_learning_rate must both "
            "be greater than zero for partial-unfreeze training."
        )

    save_dir = Path(
        save_dir
        or setup.project_root
        / "experiments"
        / experiment_name
        / setup.dataset_name
        / f"fold_{setup.classification_fold}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "classifier_best.pth"
    last_path = save_dir / "classifier_last.pth"
    history_path = save_dir / "training_history.csv"

    classification_parameters = [
        parameter
        for parameter in setup.model.classifier.parameters()
        if parameter.requires_grad
    ]
    backbone_parameters = [
        parameter
        for name, parameter in setup.model.named_parameters()
        if parameter.requires_grad and not name.startswith("classifier.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": classification_parameters,
                "lr": classifier_learning_rate,
                "name": "classifier",
            },
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "name": "unfrozen_encoder",
            },
        ],
        weight_decay=weight_decay,
    )
    # reduction="sum" is intentional. _finetuning_epoch divides by ordinary
    # batch size, preserving class weights even when image_batch_size == 1.
    criterion = nn.CrossEntropyLoss(
        weight=data.class_weights,
        reduction="sum",
        label_smoothing=data.label_smoothing,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=setup.device.type == "cuda"
    )
    minimize = checkpoint_metric.endswith("loss")
    best_score = float("inf") if minimize else float("-inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = _finetuning_epoch(
            setup,
            data.train_loader,
            criterion,
            optimizer,
            scaler,
            auxiliary_segmentation_weight,
            gradient_clip_norm,
        )
        val_metrics = _finetuning_epoch(
            setup,
            data.val_loader,
            criterion,
            None,
            scaler,
            auxiliary_segmentation_weight,
            gradient_clip_norm,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - start,
            "classifier_learning_rate": optimizer.param_groups[0]["lr"],
            "backbone_learning_rate": optimizer.param_groups[1]["lr"],
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        if checkpoint_metric not in row:
            raise KeyError(
                f"Unknown checkpoint_metric={checkpoint_metric!r}. "
                f"Available values are {sorted(row)}."
            )
        score = float(row[checkpoint_metric])
        improved = score < best_score if minimize else score > best_score
        row["saved_best"] = improved

        checkpoint = {
            "epoch": epoch,
            "checkpoint_type": "partial_unfreeze",
            "segmentation_state_dict": {
                name: value.detach().cpu()
                for name, value
                in setup.model.segmentation_network.state_dict().items()
            },
            "classifier_state_dict": {
                name: value.detach().cpu()
                for name, value in setup.model.classifier.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "experiment_name": experiment_name,
            "dataset": setup.dataset_name,
            "classification_fold": setup.classification_fold,
            "segmentation_model_dir": str(setup.segmentation_model_dir),
            "segmentation_checkpoint": setup.segmentation_checkpoint,
            "segmentation_fold": setup.segmentation_fold,
            "feature_channels": setup.feature_channels,
            "patch_size": setup.patch_size,
            "label_values": setup.label_values,
            "label_to_index": setup.label_to_index,
            "auxiliary_segmentation_weight": auxiliary_segmentation_weight,
            "checkpoint_metric": checkpoint_metric,
            "unfreeze_last_n_stages": setup.model.unfreeze_last_n_stages,
            "encoder_stage_count": setup.model.encoder_stage_count,
            "unfrozen_stage_indices": list(
                setup.model.unfrozen_stage_indices
            ),
            **row,
        }
        if improved:
            best_score = score
        checkpoint["best_checkpoint_score"] = best_score
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        if verbose:
            prediction_counts = [
                int(row[f"val_prediction_count_class_{class_index}"])
                for class_index in range(setup.num_classes)
            ]
            print(
                f"Epoch {epoch:03d} | train loss {row['train_loss']:.4f} | "
                f"val loss {row['val_loss']:.4f} | val bal acc "
                f"{row['val_balanced_accuracy']:.4f}"
                + f" | val predictions {prediction_counts}"
                + (" | saved best" if improved else "")
            )

    return TrainingResult(
        best_model_path=best_path,
        last_model_path=last_path,
        history_path=history_path,
        history=pd.DataFrame(history),
        best_val_loss=float(min(row["val_loss"] for row in history)),
    )


def load_classifier_checkpoint(
    setup: A0Setup, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Load an A0-A5 checkpoint into the matching model."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=setup.device, weights_only=False
    )

    if "segmentation_state_dict" in checkpoint:
        if not isinstance(setup.model, PartialUnfreezeSegClassifier):
            raise TypeError("A4/A5 checkpoints require a partial-unfreeze setup.")
        setup.model.segmentation_network.load_state_dict(
            checkpoint["segmentation_state_dict"],
            strict=True,
        )
        setup.model.classifier.load_state_dict(
            checkpoint["classifier_state_dict"],
            strict=True,
        )

    elif "adapted_model_state_dict" in checkpoint:
        # Backward compatibility with older A4/A5 checkpoints.
        if not isinstance(setup.model, PartialUnfreezeSegClassifier):
            raise TypeError("A4/A5 checkpoints require a partial-unfreeze setup.")
        setup.model.load_state_dict(
            checkpoint["adapted_model_state_dict"],
            strict=False,
        )

    elif "classifier_state_dict" in checkpoint:
        checkpoint_mode = _normalise_feature_mode(
            checkpoint.get("feature_mode", "A0")
        )
        checkpoint_dimension = int(
            checkpoint.get("feature_dimension", setup.feature_dimension)
        )
        if (
            checkpoint_mode != setup.feature_mode
            or checkpoint_dimension != setup.feature_dimension
        ):
            raise ValueError(
                f"Checkpoint uses {checkpoint_mode} features with dimension "
                f"{checkpoint_dimension}, but current setup uses "
                f"{setup.feature_mode} with dimension {setup.feature_dimension}."
            )
        setup.model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        raise KeyError(
            "Checkpoint does not contain model weights."
        )

    setup.model.to(setup.device).eval()
    if isinstance(setup.model, PartialUnfreezeSegClassifier):
        checkpoint["predictor_weight_installation"] = (
            install_finetuned_segmentation_weights_in_predictor(setup)
        )
    return checkpoint


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
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    image_batch_size: int = 1,
    feature_batch_size: int = 64,
    num_workers: int = 0,
    model_name: str | None = None,
    device: str | torch.device | None = None,
    hidden_channels: int = 256,
    dropout: float = 0.30,
    checkpoint_metric: str = "val_loss",
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
