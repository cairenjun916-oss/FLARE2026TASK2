from __future__ import annotations

import unittest
import importlib.util
import sys
import types

import numpy as np


class _Placeholder:
    pass


if importlib.util.find_spec("torch") is None:
    # The policy tests do not exercise image tensors. Lightweight stubs let the
    # pure weighting/metric functions run in a minimal CPU test environment;
    # the normal uv environment imports the real dependencies.
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = _Placeholder
    nn_module = types.ModuleType("torch.nn")
    nn_module.Module = _Placeholder
    torch_module.nn = nn_module
    utils_module = types.ModuleType("torch.utils")
    data_module = types.ModuleType("torch.utils.data")
    data_module.DataLoader = _Placeholder
    data_module.Dataset = _Placeholder
    data_module.TensorDataset = _Placeholder
    utils_module.data = data_module
    sys.modules.update(
        {
            "torch": torch_module,
            "torch.nn": nn_module,
            "torch.utils": utils_module,
            "torch.utils.data": data_module,
        }
    )

for module_name in ("pandas", "monai", "monai.transforms", "tqdm", "tqdm.auto"):
    root_name = module_name.split(".")[0]
    try:
        available = importlib.util.find_spec(root_name) is not None
    except (ImportError, ValueError):
        available = False
    if not available:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

if not hasattr(sys.modules["monai.transforms"], "CenterSpatialCropd"):
    for transform_name in (
        "CenterSpatialCropd",
        "Compose",
        "EnsureTyped",
        "RandFlipd",
        "RandSpatialCropd",
        "SpatialPadd",
    ):
        setattr(sys.modules["monai.transforms"], transform_name, _Placeholder)
if not hasattr(sys.modules["tqdm.auto"], "tqdm"):
    sys.modules["tqdm.auto"].tqdm = lambda iterable, **_: iterable

for module_name in (
    "nnunetv2.inference.predict_from_raw_data",
    "nnunetv2.training.dataloading.nnunet_dataset",
):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))
sys.modules["nnunetv2.inference.predict_from_raw_data"].nnUNetPredictor = _Placeholder
sys.modules[
    "nnunetv2.training.dataloading.nnunet_dataset"
].infer_dataset_class = lambda *_: None

from automsc_pipeline.classifier import (
    _classification_metrics,
    automatic_cached_training_configuration,
    automatic_class_weights,
    automatic_head_configuration,
    automatic_label_smoothing,
)


class ClassificationAutoconfigurationTests(unittest.TestCase):
    def test_effective_number_weights_are_balanced_but_bounded(self) -> None:
        weights, strategy = automatic_class_weights([147, 12], "auto")
        ratio = float(weights[1] / weights[0])
        self.assertEqual(strategy, "effective_number")
        self.assertGreater(ratio, 1.0)
        self.assertLess(ratio, 147 / 12)
        self.assertAlmostEqual(
            float(np.average(weights, weights=[147, 12])), 1.0, places=6
        )

    def test_balanced_data_is_not_needlessly_reweighted(self) -> None:
        weights, strategy = automatic_class_weights([50, 48], "auto")
        self.assertEqual(strategy, "none")
        np.testing.assert_allclose(weights, np.ones(2))

    def test_automatic_smoothing_is_mild_and_scarcity_aware(self) -> None:
        rare = automatic_label_smoothing([147, 12])
        common = automatic_label_smoothing([500, 450])
        self.assertGreater(rare, common)
        self.assertLessEqual(rare, 0.05)
        self.assertGreaterEqual(common, 0.0)

    def test_probability_metrics_reward_ranking(self) -> None:
        targets = [0, 0, 0, 1, 1]
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.2, 0.8]]
        )
        predictions = probabilities.argmax(axis=1)
        metrics = _classification_metrics(
            targets, predictions, 2, probabilities=probabilities
        )
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["selection_score"], 1.0)

    def test_head_capacity_and_dropout_follow_data_difficulty(self) -> None:
        scarce_width, scarce_dropout = automatic_head_configuration([147, 12])
        larger_width, larger_dropout = automatic_head_configuration([331, 84, 97])
        self.assertEqual(scarce_width, 128)
        self.assertEqual(larger_width, 256)
        self.assertLessEqual(scarce_width, larger_width)
        self.assertGreater(scarce_dropout, larger_dropout)
        self.assertGreaterEqual(scarce_dropout, 0.25)
        self.assertLessEqual(scarce_dropout, 0.40)

    def test_cached_training_configuration_adapts_to_data_difficulty(self) -> None:
        scarce = automatic_cached_training_configuration([147, 12])
        abundant = automatic_cached_training_configuration([500, 450])
        self.assertGreater(scarce.epochs, abundant.epochs)
        self.assertLess(scarce.learning_rate, abundant.learning_rate)
        self.assertGreater(scarce.weight_decay, abundant.weight_decay)
        self.assertGreater(scarce.minimum_epochs, abundant.minimum_epochs)
        self.assertGreater(
            scarce.early_stopping_patience,
            abundant.early_stopping_patience,
        )
        self.assertEqual(scarce.checkpoint_metric, "val_selection_score")

    def test_cached_training_configuration_rejects_missing_classes(self) -> None:
        with self.assertRaises(ValueError):
            automatic_cached_training_configuration([20, 0])


if __name__ == "__main__":
    unittest.main()
