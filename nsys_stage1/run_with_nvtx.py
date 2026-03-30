# -*- coding: utf-8 -*-
"""
使用 ONNX Runtime 运行模型，并在正式循环外层打 NVTX range。
"""
import argparse
import os
import sys

import numpy as np
import nvtx
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger("nsys_stage1.run_with_nvtx")
PROVIDERS = [
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
    "CPUExecutionProvider",
]


def _to_numpy_dtype(ort_type):
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    if "double" in ort_type:
        return np.float64
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "int8" in ort_type:
        return np.int8
    if "uint8" in ort_type:
        return np.uint8
    if "bool" in ort_type:
        return np.bool_
    return np.float32


def _normalize_shape(shape):
    dims = []
    for dim in shape:
        if dim in (None, 0, "None"):
            dims.append(1)
        else:
            dims.append(int(dim))
    return dims


def _build_feed(session):
    feed = {}
    for inp in session.get_inputs():
        shape = _normalize_shape(inp.shape)
        dtype = _to_numpy_dtype(inp.type)
        if np.issubdtype(dtype, np.bool_):
            data = np.random.randint(0, 2, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = np.random.randint(0, 8, size=shape).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        feed[inp.name] = data
    return feed


def run_model(model_path, warmup, loops, range_name):
    session = ort.InferenceSession(model_path, providers=PROVIDERS)
    feed = _build_feed(session)

    for _ in range(warmup):
        session.run(None, feed)

    with nvtx.annotate(range_name):
        for _ in range(loops):
            session.run(None, feed)


def main():
    parser = argparse.ArgumentParser(description="Run ONNX model with NVTX range.")
    parser.add_argument("--model", required=True, help="ONNX 模型路径")
    parser.add_argument("--warmup", type=int, default=20, help="warmup 次数")
    parser.add_argument("--loops", type=int, default=50, help="正式循环次数")
    parser.add_argument("--range-name", default="measure", help="NVTX range 名称")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        logger.error("模型文件不存在: %s", args.model)
        sys.exit(1)

    logger.info(
        "运行模型: model=%s warmup=%d loops=%d range=%s",
        args.model,
        args.warmup,
        args.loops,
        args.range_name,
    )
    run_model(args.model, args.warmup, args.loops, args.range_name)


if __name__ == "__main__":
    main()
