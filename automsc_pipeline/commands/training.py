from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from .common import (
    _configure_environment,
    _discover_tasks,
    _manifest_path,
    _resolve_dataset,
    _resolve_device,
    _safe_name,
    _seed_everything,
    _write_all_case_classification_split,
    _write_json,
)


def _command_train_seg(args: argparse.Namespace) -> None:
    import torch

    context = _resolve_dataset(args.dataset, args.raw_root)
    work_root = Path(args.work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    _configure_environment(context, work_root)
    device = _resolve_device(args.device)
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )
    from nnunetv2.run.run_training import run_training

    from ..classifier import find_segmentation_model

    extract_fingerprints(
        [context.dataset_id],
        "DatasetFingerprintExtractor",
        args.processes,
        True,
        args.clean,
        False,
    )
    plans_identifier = plan_experiments(
        [context.dataset_id], args.planner
    )
    preprocess(
        [context.dataset_id],
        plans_identifier,
        configurations=(args.configuration,),
        num_processes=(args.processes,),
        verbose=False,
    )
    run_training(
        dataset_name_or_id=context.name,
        configuration=args.configuration,
        fold=args.fold,
        trainer_class_name=args.trainer,
        plans_identifier=plans_identifier,
        continue_training=args.continue_training,
        val_with_best=True,
        device=torch.device(device),
    )
    model_name = (
        f"{args.trainer}__{plans_identifier}__{args.configuration}"
    )
    model_dir, checkpoint_name, checkpoint_path = find_segmentation_model(
        work_root / "nnUNet_results",
        context.name,
        args.fold,
        checkpoint_names=("checkpoint_best.pth",),
        model_name=model_name,
    )
    manifest = {
        "version": 1,
        "dataset_name": context.name,
        "configuration": args.configuration,
        "planner": args.planner,
        "plans_identifier": plans_identifier,
        "trainer": args.trainer,
        "fold": args.fold,
        "model_name": model_dir.name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
    }
    manifest_path = _manifest_path(work_root, context.name, "segmentation")
    _write_json(manifest_path, manifest)
    print(f"Best segmentation checkpoint: {checkpoint_path}")
    print(f"Manifest: {manifest_path}")


def _load_segmentation_manifest(work_root: Path, dataset_name: str) -> dict[str, Any]:
    path = _manifest_path(work_root, dataset_name, "segmentation")
    if not path.is_file():
        raise FileNotFoundError(f"Missing segmentation manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_name") != "checkpoint_best.pth":
        raise ValueError(
            f"Segmentation manifest does not select checkpoint_best.pth: {path}"
        )
    checkpoint = Path(manifest["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Best segmentation checkpoint is missing: {checkpoint}")
    return manifest


def _command_train_cls(args: argparse.Namespace) -> None:
    import torch

    context = _resolve_dataset(args.dataset, args.raw_root)
    work_root = Path(args.work_root).expanduser().resolve()
    _configure_environment(context, work_root)
    segmentation = _load_segmentation_manifest(work_root, context.name)
    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    from ..classifier import (
        FROZEN_EXPERIMENT_NAMES,
        automatic_head_configuration,
        build_classification_data,
        cache_frozen_features,
        setup_a0,
        train_full_data_classifier,
    )
    csv_path, frame, identifier, tasks = _discover_tasks(
        context, args.label_columns
    )
    experiment_name = FROZEN_EXPERIMENT_NAMES[args.feature_mode]
    task_manifests: list[dict[str, Any]] = []
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    model_dir = (
        work_root
        / "nnUNet_results"
        / context.name
        / segmentation["model_name"]
    )
    plans_manager = PlansManager(str(model_dir / "plans.json"))
    configuration_manager = plans_manager.get_configuration(
        segmentation["configuration"]
    )
    preprocessed_folder = (
        work_root
        / "nnUNet_preprocessed"
        / context.name
        / configuration_manager.data_identifier
    )

    for task in tasks:
        task_key = _safe_name(task.name)
        class_counts = [
            int((frame[task.label_column] == value).sum())
            for value in task.class_values
        ]
        auto_hidden_channels, auto_dropout = automatic_head_configuration(
            class_counts
        )
        hidden_channels = (
            auto_hidden_channels
            if args.hidden_channels is None
            else int(args.hidden_channels)
        )
        dropout = auto_dropout if args.dropout is None else float(args.dropout)
        split_path = (
            work_root
            / "classification_splits"
            / context.name
            / f"{task_key}.json"
        )
        split_path, split = _write_all_case_classification_split(
            frame,
            identifier,
            task,
            preprocessed_folder,
            split_path,
        )
        print(
            f"{task.name}: one classifier using all {len(split['train'])} "
            f"labelled cases; hidden={hidden_channels}, dropout={dropout:.3f}."
        )
        setup = setup_a0(
            project_root=work_root,
            dataset_name=context.name,
            segmentation_fold=segmentation["fold"],
            classification_fold=0,
            device=device,
            model_name=segmentation["model_name"],
            checkpoint_names=("checkpoint_best.pth",),
            hidden_channels=hidden_channels,
            dropout=dropout,
            feature_mode=args.feature_mode,
            label_column=task.label_column,
            identifier_column=identifier,
            task_name=task_key,
            split_json_path=split_path,
        )
        data = build_classification_data(
            setup,
            image_batch_size=1,
            num_workers=args.workers,
            use_class_weights=True,
            class_weight_strategy=args.class_weighting,
            label_smoothing=args.label_smoothing,
        )
        if data.train_identifiers != data.val_identifiers:
            raise RuntimeError("Full-data training requires identical train/monitor IDs.")
        print(
            f"  counts={data.class_counts.tolist()}, "
            f"weights={data.class_weights.detach().cpu().tolist()}, "
            f"smoothing={data.label_smoothing:.4f}"
        )
        features = cache_frozen_features(
            setup,
            data,
            train_variants=args.train_variants,
            feature_batch_size=args.batch_size,
            extraction_batch_size=1,
            extraction_num_workers=args.workers,
            reuse_cache=not args.rebuild_cache,
            experiment_name=experiment_name,
        )
        save_dir = (
            work_root
            / "experiments"
            / experiment_name
            / context.name
            / task_key
            / "full_data"
        )
        training = train_full_data_classifier(
            setup,
            data,
            features,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            save_dir=save_dir,
            averaging_fraction=args.averaging_fraction,
            experiment_name=experiment_name,
        )
        if training.best_model_path.name != "classifier_final.pth":
            raise RuntimeError("Full-data training did not return its final checkpoint.")
        checkpoint = torch.load(
            training.best_model_path, map_location="cpu", weights_only=False
        )
        task_manifests.append(
            {
                "name": task.name,
                "task_key": task_key,
                "label_column": task.label_column,
                "class_values": list(task.class_values),
                "output_columns": list(task.output_columns),
                "split_path": str(split_path),
                "checkpoint_path": str(training.best_model_path),
                "checkpoint_name": training.best_model_path.name,
                "training_scope": "all_labelled_cases",
                "monitoring_scope": "same_cases_center_crop_diagnostic_only",
                "training_cases": len(split["train"]),
                "class_counts": checkpoint.get("class_counts"),
                "class_weights": checkpoint.get("class_weights"),
                "class_weight_strategy": checkpoint.get("class_weight_strategy"),
                "label_smoothing": checkpoint.get("label_smoothing"),
                "hidden_channels": checkpoint.get("hidden_channels"),
                "dropout": checkpoint.get("dropout"),
                "averaging_fraction": checkpoint.get("averaging_fraction"),
                "averaged_epoch_count": checkpoint.get("averaged_epoch_count"),
                "final_training_metrics": checkpoint.get("final_train_metrics"),
                "final_monitoring_metrics": checkpoint.get("final_monitor_metrics"),
            }
        )
        print(f"{task.name}: final single classifier: {training.best_model_path}")
        del setup, data, features, training, checkpoint
        gc.collect()

    manifest = {
        "version": 5,
        "dataset_name": context.name,
        "classification_csv": str(csv_path),
        "identifier_column": identifier,
        "feature_mode": args.feature_mode,
        "classification_fold": 0,
        "training_policy": (
            "one classifier per target, trained on every labelled case; "
            "late-epoch parameter averaging within that single training run"
        ),
        "model_selection": "none_full_data_fixed_schedule",
        "class_weighting": args.class_weighting,
        "segmentation": segmentation,
        "tasks": task_manifests,
    }
    manifest_path = _manifest_path(work_root, context.name, "classification")
    _write_json(manifest_path, manifest)
    print(f"Classification manifest: {manifest_path}")
