from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .core import (
    A0Data,
    A0Setup,
    TrainingResult,
    _classification_metrics,
    _normalise_feature_mode,
)
from .data import setup_a0
from .models import (
    PartialUnfreezeSegClassifier,
    install_finetuned_segmentation_weights_in_predictor,
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
