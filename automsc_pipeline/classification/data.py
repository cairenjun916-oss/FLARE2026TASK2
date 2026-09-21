from __future__ import annotations

import gc
import json
import os
import time
import warnings
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

from .core import (
    FROZEN_EXPERIMENT_NAMES,
    A0Data,
    A0Setup,
    FeatureData,
    _normalise_feature_mode,
    automatic_class_weights,
    automatic_label_smoothing,
)
from .models import A0FrozenSegClassifier


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
