# -*- coding: utf-8 -*-
"""
Single Kernel Runner
使用 ONNX Runtime (CUDAExecutionProvider) 运行单个 kernel ONNX 模型，
支持 warmup 与 loop，输出平均推理耗时与启动/结束时间。
"""
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger("ort_kernel_runner")

PROVIDERS = ["CUDAExecutionProvider"]


def _to_numpy_dtype(ort_type):
    """将 ORT 输入类型映射到 numpy dtype"""
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
    # 默认 float32
    return np.float32


def _normalize_shape(shape):
    """将动态/None shape 归一化为可生成的形状"""
    norm = []
    for d in shape:
        if d is None or d == 0:
            norm.append(1)
        else:
            norm.append(int(d))
    return norm


def _prepare_inputs(session):
    """为 session 的所有输入生成随机 dummy 数据"""
    feed = {}
    for inp in session.get_inputs():
        shape = _normalize_shape(inp.shape)
        dtype = _to_numpy_dtype(inp.type)
        data = np.random.randn(*shape).astype(dtype)
        feed[inp.name] = data
    return feed


def run_kernel(model_path, warmup=10, loops=50, use_warmup=True):
    """
    运行单个 kernel ONNX 模型并返回平均推理耗时 (ms) 与起止时间。
    """
    sess = ort.InferenceSession(model_path, providers=PROVIDERS)
    feed = _prepare_inputs(sess)

    # Warmup
    if use_warmup and warmup > 0:
        for _ in range(warmup):
            _ = sess.run(None, feed)

    # 正式计时
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    start_time = datetime.now()
    t0 = time.perf_counter()
    for _ in range(loops):
        _ = sess.run(None, feed)
    t1 = time.perf_counter()
    end_time = datetime.now()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    avg_ms = ((t1 - t0) / max(loops, 1)) * 1000.0
    return avg_ms, start_time, end_time


def main():
    parser = argparse.ArgumentParser(description="Single Kernel Runner")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--loops", type=int, default=50, help="循环次数")
    parser.add_argument("--no-warmup", action="store_true", help="禁用预热")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        logger.error("模型文件不存在: %s", args.model)
        sys.exit(1)

    logger.info("运行模型: %s", args.model)
    avg_ms, start_time, end_time = run_kernel(
        args.model, warmup=args.warmup, loops=args.loops, use_warmup=not args.no_warmup
    )

    logger.info("启动时间: %s", start_time.isoformat(timespec="seconds"))
    logger.info("结束时间: %s", end_time.isoformat(timespec="seconds"))
    logger.info("平均推理耗时: %.3f ms", avg_ms)


if __name__ == "__main__":
    main()
