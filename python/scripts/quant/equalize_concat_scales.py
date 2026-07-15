#!/usr/bin/env python3
"""Equalize calibration qparams across ONNX Concat boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--scale-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--exclude-output",
        action="append",
        default=[],
        help="Concat output name to leave unchanged; may be repeated.",
    )
    parser.add_argument(
        "--override-qparam",
        action="append",
        nargs=3,
        metavar=("TENSOR", "SCALE", "ZERO_POINT"),
        default=[],
        help="Override one tensor qparam after Concat equalization; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.scale_table.read_text(encoding="utf-8").splitlines()
    qparams: dict[str, tuple[str, str]] = {}
    for line in lines:
        fields = line.split()
        if len(fields) >= 3:
            qparams[fields[0]] = (fields[1], fields[2])

    model = onnx.load(args.onnx)
    replacements: dict[str, tuple[str, str]] = {}
    changed_nodes = 0
    excluded = set(args.exclude_output)
    for node in model.graph.node:
        if node.op_type != "Concat" or not node.output or node.output[0] in excluded:
            continue
        output_name = node.output[0]
        if output_name not in qparams:
            raise KeyError(f"Missing Concat output in scale table: {output_name}")
        output_qparams = qparams[output_name]
        for input_name in node.input:
            if input_name not in qparams:
                raise KeyError(f"Missing Concat input in scale table: {input_name}")
            replacements[input_name] = output_qparams
        changed_nodes += 1

    for name, scale, zero_point in args.override_qparam:
        if name not in qparams:
            raise KeyError(f"Missing tensor in scale table: {name}")
        replacements[name] = (scale, zero_point)

    output_lines = []
    changed_tensors = 0
    for line in lines:
        fields = line.split()
        if len(fields) >= 3 and fields[0] in replacements:
            scale, zero_point = replacements[fields[0]]
            output_lines.append(f"{fields[0]} {scale} {zero_point}")
            changed_tensors += 1
        else:
            output_lines.append(line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(
        f"equalized_concat_nodes={changed_nodes} changed_tensors={changed_tensors} "
        f"overridden_tensors={len(args.override_qparam)} output={args.output}"
    )


if __name__ == "__main__":
    main()
