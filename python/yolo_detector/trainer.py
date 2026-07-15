from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK


PRETRAINED_ENV = "BARCODE_PRETRAINED_WEIGHTS"
PRETRAINED_BACKBONE_LAYERS_ENV = "BARCODE_PRETRAINED_BACKBONE_LAYERS"
FIRST_CONV_KEY = "model.0.conv.weight"
LAYER_KEY = re.compile(r"^model\.(\d+)\.")


def resolve_pretrained_weights(source: str | Path) -> Path:
    candidate = Path(source).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    pretrained = YOLO(str(source))
    checkpoint = Path(pretrained.ckpt_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Unable to resolve pretrained weights: {source}")
    return checkpoint.resolve()


def _is_transferable_backbone_key(key: str, max_layer: int) -> bool:
    match = LAYER_KEY.match(key)
    return match is not None and int(match.group(1)) <= max_layer


def transfer_grayscale_backbone(
    target: DetectionModel,
    pretrained: str | Path,
    max_layer: int = 9,
) -> dict[str, Any]:
    """Transfer shared layers and collapse RGB stem weights to grayscale."""
    source = YOLO(str(pretrained)).model.float().cpu()
    source_state = source.state_dict()
    target_state = target.state_dict()
    transferred: dict[str, torch.Tensor] = {}

    for key, target_value in target_state.items():
        if key == FIRST_CONV_KEY or not _is_transferable_backbone_key(key, max_layer):
            continue
        source_value = source_state.get(key)
        if source_value is not None and source_value.shape == target_value.shape:
            transferred[key] = source_value.detach().to(dtype=target_value.dtype)

    source_stem = source_state.get(FIRST_CONV_KEY)
    target_stem = target_state.get(FIRST_CONV_KEY)
    if source_stem is None or target_stem is None:
        raise KeyError(f"Missing first convolution key: {FIRST_CONV_KEY}")
    if source_stem.ndim != 4 or source_stem.shape[1] != 3 or target_stem.shape[1] != 1:
        raise ValueError(
            f"Expected RGB->gray stem transfer, got {tuple(source_stem.shape)} -> {tuple(target_stem.shape)}"
        )

    collapsed_stem = source_stem.sum(dim=1, keepdim=True)
    if collapsed_stem.shape != target_stem.shape:
        raise ValueError(
            f"Collapsed stem shape mismatch: {tuple(collapsed_stem.shape)} != {tuple(target_stem.shape)}"
        )
    transferred[FIRST_CONV_KEY] = collapsed_stem.to(dtype=target_stem.dtype)
    target.load_state_dict(transferred, strict=False)

    report = {
        "pretrained": str(pretrained),
        "max_source_layer": max_layer,
        "transferred_tensors": len(transferred),
        "target_tensors": len(target_state),
        "stem_source_shape": list(source_stem.shape),
        "stem_target_shape": list(target_stem.shape),
        "stem_reduction": "sum",
    }
    LOGGER.info(
        "Transferred %d/%d tensors from pretrained shared layers; RGB stem collapsed by channel sum.",
        len(transferred),
        len(target_state),
    )
    return report


def build_grayscale_model(
    config: str | Path,
    nc: int = 3,
    pretrained: str | Path | None = None,
    pretrained_max_layer: int = 9,
    verbose: bool = True,
) -> tuple[DetectionModel, dict[str, Any] | None]:
    model = DetectionModel(str(config), ch=1, nc=nc, verbose=verbose)
    report = (
        transfer_grayscale_backbone(model, pretrained, max_layer=pretrained_max_layer)
        if pretrained
        else None
    )
    return model, report


class GrayscalePretrainedDetectionTrainer(DetectionTrainer):
    """Detection trainer that performs a deterministic RGB-to-gray backbone transfer."""

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = DetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if self.data["channels"] != 1:
            raise ValueError(f"Barcode detector requires channels: 1, got {self.data['channels']}")

        if weights:
            model.load(weights)
            return model

        pretrained = os.environ.get(PRETRAINED_ENV)
        if pretrained:
            max_layer = int(os.environ.get(PRETRAINED_BACKBONE_LAYERS_ENV, "9"))
            self.grayscale_transfer_report = transfer_grayscale_backbone(model, pretrained, max_layer=max_layer)
        else:
            LOGGER.warning("No pretrained backbone configured; training the grayscale detector from scratch.")
        return model
