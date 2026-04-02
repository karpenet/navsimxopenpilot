#!/usr/bin/env python3
"""
Convert comma ``driving_vision.onnx`` (FP16 weights + uint8→FP16 image casts) to FP32.

FP16 accumulation on NAVSIM-packed frames often overflows to NaN/Inf in the 1576-D
output; FP32 weights and internal math fixes that without masking outputs.

Usage:
  python convert_driving_vision_fp16_to_fp32.py \\
    --input /path/to/driving_vision.onnx \\
    --output /path/to/driving_vision_fp32.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, shape_inference


def convert(path_in: Path, path_out: Path) -> None:
    model = onnx.load(str(path_in))

    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(arr, name=init.name))

    for n in model.graph.node:
        if n.op_type != "Cast":
            continue
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT

    for o in model.graph.output:
        o.type.tensor_type.elem_type = TensorProto.FLOAT

    while len(model.graph.value_info):
        model.graph.value_info.pop()

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path_out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    convert(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
