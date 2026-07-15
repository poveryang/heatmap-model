#!/usr/bin/env python3
"""Export fixed-shape YOLO detection heads without decode, thresholding, or NMS."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import onnx
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO


class FusedConvAct(nn.Module):
    """Fused convolution and activation used by the split-free C2f export."""

    def __init__(self, conv: nn.Conv2d, act: nn.Module, start: int, end: int) -> None:
        super().__init__()
        if conv.groups != 1:
            raise ValueError("Split-free C2f export requires an ungrouped cv1 convolution")
        self.conv = nn.Conv2d(
            conv.in_channels,
            end - start,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            groups=1,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            self.conv.weight.copy_(conv.weight[start:end])
            if conv.bias is not None:
                self.conv.bias.copy_(conv.bias[start:end])
        self.act = copy.deepcopy(act)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(image))


class SplitFreeC2f(nn.Module):
    """Mathematically equivalent C2f export without a runtime Split operator."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        channels = int(block.c)
        fused_conv = block.cv1.conv
        if fused_conv.out_channels != channels * 2:
            raise ValueError("Unexpected C2f cv1 output channel count")
        self.cv1a = FusedConvAct(fused_conv, block.cv1.act, 0, channels)
        self.cv1b = FusedConvAct(fused_conv, block.cv1.act, channels, channels * 2)
        self.cv2 = block.cv2
        self.m = block.m
        for attribute in ("f", "i", "type", "np"):
            if hasattr(block, attribute):
                setattr(self, attribute, getattr(block, attribute))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = [self.cv1a(image), self.cv1b(image)]
        outputs.extend(module(outputs[-1]) for module in self.m)
        return self.cv2(torch.cat(outputs, dim=1))


def replace_c2f_splits(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "C2f":
            setattr(module, name, SplitFreeC2f(child))
            replaced += 1
        else:
            replaced += replace_c2f_splits(child)
    return replaced


class StaticRawDetect(nn.Module):
    """Fixed-shape Detect head that exposes box and class logits."""

    def __init__(self, head: nn.Module, imgsz: int, per_level: bool = False) -> None:
        super().__init__()
        self.cv2 = head.cv2
        self.cv3 = head.cv3
        self.nc = int(head.nc)
        self.reg_max = int(head.reg_max)
        self.per_level = per_level
        self.stride = head.stride.detach().clone()
        self.level_elements = [(imgsz // int(stride)) ** 2 for stride in self.stride.tolist()]
        for attribute in ("f", "i", "type", "np"):
            if hasattr(head, attribute):
                setattr(self, attribute, getattr(head, attribute))

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        boxes = [self.cv2[index](features[index]) for index in range(len(features))]
        scores = [self.cv3[index](features[index]) for index in range(len(features))]
        if self.per_level:
            boxes = [value.reshape(1, 4 * self.reg_max, self.level_elements[index])
                     for index, value in enumerate(boxes)]
            scores = [value.reshape(1, self.nc, self.level_elements[index])
                      for index, value in enumerate(scores)]
            return tuple(value for pair in zip(boxes, scores) for value in pair)
        boxes = [value.reshape(1, 4 * self.reg_max, self.level_elements[index])
                 for index, value in enumerate(boxes)]
        scores = [value.reshape(1, self.nc, self.level_elements[index])
                  for index, value in enumerate(scores)]
        return torch.cat(boxes, dim=2), torch.cat(scores, dim=2)


class RawDetectionHead(nn.Module):
    def __init__(self, checkpoint: Path, imgsz: int, per_level: bool = False,
                 split_free_c2f: bool = False) -> None:
        super().__init__()
        self.core = YOLO(str(checkpoint)).model.float().eval()
        self.core.fuse(verbose=False)
        self.split_free_c2f_blocks = replace_c2f_splits(self.core) if split_free_c2f else 0
        original_head = self.core.model[-1]
        if not hasattr(original_head, "reg_max") or not hasattr(original_head, "stride"):
            raise TypeError(f"Unsupported detection head: {type(original_head).__name__}")
        for module in self.core.modules():
            if callable(getattr(module, "forward_split", None)):
                module.forward = module.forward_split
        self.head = StaticRawDetect(original_head, imgsz, per_level=per_level)
        self.core.model[-1] = self.head
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.core.eval()
        return self

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.core(image)
        if not isinstance(outputs, tuple):
            raise RuntimeError(f"Expected raw head tuple, got {type(outputs).__name__}")
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--per-level", action="store_true")
    parser.add_argument("--split-free-c2f", action="store_true")
    return parser.parse_args()


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    wrapper = RawDetectionHead(checkpoint, args.imgsz, per_level=args.per_level,
                               split_free_c2f=args.split_free_c2f)
    head = wrapper.head
    strides = [int(value) for value in head.stride.tolist()]
    output_names = (
        [name for stride in strides for name in (f"box_s{stride}", f"class_s{stride}")]
        if args.per_level else ["box_logits", "class_logits"]
    )
    torch.manual_seed(0)
    sample = torch.rand(1, 1, args.imgsz, args.imgsz, dtype=torch.float32)

    with torch.inference_mode():
        pytorch_outputs = wrapper(sample)
    output_shapes = [list(value.shape) for value in pytorch_outputs]

    torch.onnx.export(
        wrapper,
        sample,
        str(output),
        input_names=["images"],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        training=torch.onnx.TrainingMode.EVAL,
    )

    model = onnx.load(output)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, output)

    operators = Counter(node.op_type for node in model.graph.node)
    report = {
        "checkpoint": str(checkpoint),
        "onnx": str(output),
        "onnx_bytes": output.stat().st_size,
        "input": {"name": "images", "shape": [1, 1, args.imgsz, args.imgsz], "dtype": "float32"},
        "outputs": [
            {
                "name": item.name,
                "shape": tensor_shape(item),
                "pytorch_shape": output_shapes[index],
            }
            for index, item in enumerate(model.graph.output)
        ],
        "levels": [
            {
                "stride": stride,
                "height": args.imgsz // stride,
                "width": args.imgsz // stride,
                "candidates": (args.imgsz // stride) ** 2,
            }
            for stride in strides
        ],
        "classes": int(head.nc),
        "reg_max": int(head.reg_max),
        "operators": dict(sorted(operators.items())),
        "model_transform": {
            "split_free_c2f_blocks": wrapper.split_free_c2f_blocks,
            "split_free_c2f": "fused cv1 weights sliced into two equivalent convolutions"
            if wrapper.split_free_c2f_blocks
            else "none",
        },
        "postprocess": {
            "location": "C++",
            "class_activation": "sigmoid",
            "box_decode": "DFL softmax + distance decode" if int(head.reg_max) > 1 else "distance decode",
            "threshold_filter": "C++",
            "nms": "C++",
            "raw_output_layout": "per-level NCN" if args.per_level else "concatenated NCN",
        },
    }
    report_path = output.with_suffix(".operators.json")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_outputs = session.run(None, {"images": sample.numpy()})
        report["onnxruntime_parity"] = [
            {
                "output": output_names[index],
                "max_abs_error": float(np.max(np.abs(actual - expected.numpy()))),
                "mean_abs_error": float(np.mean(np.abs(actual - expected.numpy()))),
            }
            for index, (actual, expected) in enumerate(zip(ort_outputs, pytorch_outputs))
        ]
    except ImportError:
        report["onnxruntime_parity"] = "onnxruntime not installed"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
