from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CachedTrainingConfiguration:
    sample_count: int
    minority_count: int
    imbalance_ratio: float
    difficulty: float
    epochs: int
    learning_rate: float
    weight_decay: float
    checkpoint_metric: str
    selection_window: int
    minimum_epochs: int
    early_stopping_patience: int


def automatic_cached_training_configuration(
    counts: Sequence[int] | np.ndarray,
) -> CachedTrainingConfiguration:
    class_counts = np.asarray(counts, dtype=float)
    if (
        class_counts.ndim != 1
        or class_counts.size < 2
        or not np.all(np.isfinite(class_counts))
        or np.any(class_counts <= 0)
    ):
        raise ValueError(
            "Every training class needs at least one finite case; "
            f"counts={class_counts.tolist()}."
        )

    sample_count = int(class_counts.sum())
    minority_count = int(class_counts.min())
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    scarcity_factor = min(1.0, 256.0 / float(sample_count))
    imbalance_factor = min(
        1.0,
        np.log(max(imbalance_ratio, 1.0)) / np.log(10.0),
    )
    difficulty = float(max(scarcity_factor, imbalance_factor))

    epochs = int(round(150 + (100 * difficulty)))
    learning_rate = float(1e-3 / (1.0 + difficulty))
    weight_decay = float(1e-4 * (1.0 + difficulty))
    selection_window = max(1, int(round(np.sqrt(epochs) / 3.0)))
    minimum_epochs = max(selection_window, int(round(epochs * 0.20)))
    early_stopping_patience = max(
        selection_window * 2,
        int(round(epochs * 0.15)),
    )

    return CachedTrainingConfiguration(
        sample_count=sample_count,
        minority_count=minority_count,
        imbalance_ratio=imbalance_ratio,
        difficulty=difficulty,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_metric="val_selection_score",
        selection_window=selection_window,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
    )
