# AutoMSC package guide

## Layout

| Path | Responsibility |
|---|---|
| `__main__.py` | Supports `python -m automsc_pipeline`. |
| `cli.py` | Defines the stable `automsc` parser and command entry point. |
| `commands/common.py` | Resolves datasets, paths, tasks, devices, manifests, and random seeds. |
| `commands/training.py` | Implements `train_seg` and `train_cls`. |
| `commands/inference.py` | Implements `infer` and validates prediction outputs. |
| `classifier.py` | Preserves the original classifier import surface. |
| `classification/configuration.py` | Calculates cached-classifier training settings from class counts. |
| `classification/core.py` | Contains shared data records, metrics, class weighting, label smoothing, and head configuration. |
| `classification/models.py` | Contains the frozen and partially unfrozen segmentation-backed classifiers. |
| `classification/data.py` | Builds datasets, initializes models, and caches frozen features. |
| `classification/frozen_training.py` | Trains classifier heads from cached features. |
| `classification/finetuning.py` | Handles partial unfreezing, auxiliary segmentation loss, and checkpoint loading. |
| `classification/workflows.py` | Provides A0-A5 workflows, inference, and evaluation. |

`cli.py` and `classifier.py` are compatibility facades. Existing commands and
imports continue to use these paths even though their implementations are split
into subpackages.

## Automatic cached-training configuration

Calling `train_cached_classifier` without optimization overrides calculates
its settings from the training class counts. The calculation uses the same
sample-scarcity and class-imbalance signals already used by the classifier head
configuration.

Let `N` be the number of labelled cases and `r` the largest class count divided
by the smallest class count:

```text
scarcity  = min(1, 256 / N)
imbalance = min(1, log(max(r, 1)) / log(10))
difficulty = max(scarcity, imbalance)

epochs       = round(150 + 100 * difficulty)
learning rate = 1e-3 / (1 + difficulty)
weight decay  = 1e-4 * (1 + difficulty)
```

The metric is `val_selection_score`, which is the existing balanced-accuracy
and AUROC selection score. The smoothing window, minimum training duration,
and early-stopping patience scale with the calculated epoch count. Every
resolved value is saved in the checkpoint under `training_configuration`.

Explicit arguments always override automatic values. Passing `None` requests
automatic resolution, while passing `early_stopping_patience=None` explicitly
disables early stopping.

```python
from automsc_pipeline.classifier import train_cached_classifier

automatic_result = train_cached_classifier(setup, data, features)

overridden_result = train_cached_classifier(
    setup,
    data,
    features,
    epochs=220,
    learning_rate=7e-4,
    early_stopping_patience=None,
)
```

`setup`, `data`, and `features` are pipeline objects created before training.
`save_dir` and `experiment_name` are inferred when omitted. `verbose` remains
a user preference rather than a dataset-derived value.

## Reproduction command policy

The packaged `automsc train_cls` command uses the separate full-data training
path. Its fixed 250-epoch, `3e-4` learning-rate, `1e-4` weight-decay, and final
40% weight-averaging schedule remains unchanged so published reproduction runs
stay comparable. Class weights, label smoothing, hidden width, and dropout are
still calculated independently for every target.

## Commands

```bash
automsc train_seg DatasetXXX_NAME --raw-root /data --work-root /work
automsc train_cls DatasetXXX_NAME --raw-root /data --work-root /work
automsc infer DatasetXXX_NAME --raw-root /data --work-root /work \
  --output-root /submission/predictions
```
