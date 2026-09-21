from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .commands.common import (
    DATASET_PATTERN,
    DatasetContext,
    TaskSpec,
    _classification_csv,
    _configure_environment,
    _discover_tasks,
    _identifier_column,
    _json_default,
    _manifest_path,
    _preprocessed_identifiers,
    _resolve_dataset,
    _resolve_device,
    _safe_name,
    _seed_everything,
    _sort_class_values,
    _write_all_case_classification_split,
    _write_json,
)
from .commands.inference import (
    _build_classifier_head,
    _center_patch,
    _command_infer,
    _group_test_images,
    _validate_prediction_geometry,
)
from .commands.training import (
    _command_train_cls,
    _command_train_seg,
    _load_segmentation_manifest,
)


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset", help="DatasetXXX_NAME or the path to that dataset folder."
    )
    parser.add_argument(
        "--raw-root",
        help="Folder containing DatasetXXX_NAME (defaults to nnUNet_raw).",
    )
    parser.add_argument(
        "--work-root",
        required=True,
        help="Writable folder for preprocessed data, checkpoints, and manifests.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu", "mps"), default="auto"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automsc",
        description=(
            "Dataset-agnostic FLARE AutoMSC segmentation/classification pipeline."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_seg = commands.add_parser(
        "train_seg", help="Plan, preprocess, and train segmentation."
    )
    _add_common_paths(train_seg)
    train_seg.add_argument("--configuration", default="3d_fullres")
    train_seg.add_argument("--planner", default="nnUNetPlannerResEncM")
    train_seg.add_argument("--trainer", default="nnUNetTrainerQuickSeg")
    train_seg.add_argument("--fold", default="all")
    train_seg.add_argument("--processes", type=int, default=4)
    train_seg.add_argument("--clean", action="store_true")
    train_seg.add_argument("--continue", dest="continue_training", action="store_true")
    train_seg.set_defaults(handler=_command_train_seg)

    train_cls = commands.add_parser(
        "train_cls",
        help="Train one auto-configured full-data classifier per target.",
    )
    _add_common_paths(train_cls)
    train_cls.add_argument("--feature-mode", choices=("A0", "A1", "A2", "A3"), default="A3")
    train_cls.add_argument("--label-columns", nargs="+")
    train_cls.add_argument(
        "--class-weighting",
        choices=("auto", "effective_number", "inverse", "sqrt_inverse", "none"),
        default="auto",
        help=(
            "Class-weight policy. Auto uses stable effective-number weights "
            "only when imbalance is material."
        ),
    )
    train_cls.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Override automatic label smoothing with a value in [0, 1).",
    )
    train_cls.add_argument("--epochs", type=int, default=250)
    train_cls.add_argument("--learning-rate", type=float, default=3e-4)
    train_cls.add_argument("--weight-decay", type=float, default=1e-4)
    train_cls.add_argument(
        "--hidden-channels",
        type=int,
        default=None,
        help="Override the head width inferred from labelled sample count.",
    )
    train_cls.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override dropout inferred from sample scarcity and imbalance.",
    )
    train_cls.add_argument("--train-variants", type=int, default=2)
    train_cls.add_argument("--batch-size", type=int, default=64)
    train_cls.add_argument(
        "--averaging-fraction",
        type=float,
        default=0.40,
        help="Fraction of late epochs averaged into the one final model.",
    )
    train_cls.add_argument("--workers", type=int, default=0)
    train_cls.add_argument("--seed", type=int, default=12345)
    train_cls.add_argument("--rebuild-cache", action="store_true")
    train_cls.set_defaults(handler=_command_train_cls)

    infer = commands.add_parser(
        "infer", help="Create the exact per-dataset prediction folder."
    )
    _add_common_paths(infer)
    infer.add_argument(
        "--output-root",
        required=True,
        help="The submission predictions/ directory.",
    )
    infer.add_argument("--export-threads", type=int, default=1)
    infer.set_defaults(handler=_command_infer)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except KeyboardInterrupt:
        raise
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
