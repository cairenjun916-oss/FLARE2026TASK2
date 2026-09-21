from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .configuration import automatic_cached_training_configuration
from .core import (
    FROZEN_EXPERIMENT_NAMES,
    A0Data,
    A0Setup,
    FeatureData,
    TrainingResult,
    _classification_metrics,
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
    epochs: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    save_dir: str | Path | None = None,
    checkpoint_metric: str | None = None,
    selection_window: int | None = None,
    minimum_epochs: int | None = None,
    early_stopping_patience: int | None | Literal["auto"] = "auto",
    verbose: bool = True,
    experiment_name: str | None = None,
) -> TrainingResult:
    """Train an A0-A3 classification head using cached frozen features."""
    automatic = automatic_cached_training_configuration(data.class_counts)
    epochs = automatic.epochs if epochs is None else int(epochs)
    learning_rate = (
        automatic.learning_rate
        if learning_rate is None
        else float(learning_rate)
    )
    weight_decay = (
        automatic.weight_decay if weight_decay is None else float(weight_decay)
    )
    checkpoint_metric = checkpoint_metric or automatic.checkpoint_metric
    selection_window = (
        automatic.selection_window
        if selection_window is None
        else int(selection_window)
    )
    minimum_epochs = (
        automatic.minimum_epochs
        if minimum_epochs is None
        else int(minimum_epochs)
    )
    if early_stopping_patience == "auto":
        early_stopping_patience = automatic.early_stopping_patience
    elif early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)

    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
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
    resolved_configuration = {
        **asdict(automatic),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "checkpoint_metric": checkpoint_metric,
        "selection_window": selection_window,
        "minimum_epochs": minimum_epochs,
        "early_stopping_patience": early_stopping_patience,
    }
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
            "training_configuration": resolved_configuration,
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
