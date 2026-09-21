from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .core import (
    A0Setup,
    _frozen_feature_dimension,
    _normalise_feature_mode,
)


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
