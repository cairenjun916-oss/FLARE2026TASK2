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

from .classification.configuration import (
    CachedTrainingConfiguration,
    automatic_cached_training_configuration,
)
from .classification.core import (
    FROZEN_EXPERIMENT_NAMES,
    A0Data,
    A0Setup,
    FeatureData,
    InferenceResult,
    TrainingResult,
    _classification_metrics,
    _frozen_feature_dimension,
    _normalise_feature_mode,
    automatic_class_weights,
    automatic_head_configuration,
    automatic_label_smoothing,
    balanced_accuracy_details,
)
from .classification.data import (
    A0PreprocessedDataset,
    FineTuningPreprocessedDataset,
    _extract_feature_tensors,
    build_classification_data,
    build_finetuning_data,
    cache_frozen_features,
    configure_environment,
    find_segmentation_model,
    setup_a0,
)
from .classification.finetuning import (
    _dice_from_probabilities,
    _finetuning_epoch,
    _masked_mean,
    auxiliary_segmentation_loss,
    load_classifier_checkpoint,
    setup_partial_unfreeze,
    train_partial_unfreeze,
)
from .classification.frozen_training import (
    _classification_epoch,
    train_cached_classifier,
    train_full_data_classifier,
)
from .classification.models import (
    A0FrozenSegClassifier,
    PartialUnfreezeSegClassifier,
    _unwrapped_network,
    install_finetuned_segmentation_weights_in_predictor,
    verify_partial_unfreeze_configuration,
)
from .classification.workflows import (
    evaluate_predictions,
    run_a0_training,
    run_a1_training,
    run_a2_training,
    run_a3_training,
    run_a4_training,
    run_a5_training,
    run_fold_inference,
    run_frozen_feature_training,
)

_COMPATIBILITY_CLASSES = (
    A0Setup,
    A0Data,
    FeatureData,
    TrainingResult,
    InferenceResult,
    A0FrozenSegClassifier,
    A0PreprocessedDataset,
    PartialUnfreezeSegClassifier,
    FineTuningPreprocessedDataset,
)
for _compatibility_class in _COMPATIBILITY_CLASSES:
    _compatibility_class.__module__ = __name__
del _compatibility_class, _COMPATIBILITY_CLASSES
