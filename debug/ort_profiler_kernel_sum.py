# -*- coding: utf-8 -*-
"""
独立测试脚本：统计 ORT profiler 的每个 kernel 节点平均执行时间。

默认口径：
- 先 warmup，再 loop
- 仅统计最后 `loops` 次（剔除 warmup）
- 输出每个 kernel 节点的平均执行时间（ms）
"""
import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def _to_numpy_dtype(ort_type: str):
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


def _normalize_shape(shape: List) -> List[int]:
    out = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            out.append(int(d))
        else:
            out.append(1)
    return out


def _prepare_inputs(session: ort.InferenceSession) -> Dict[str, np.ndarray]:
    feeds: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = _normalize_shape(inp.shape)
        dtype = _to_numpy_dtype(inp.type)
        if np.issubdtype(dtype, np.floating):
            arr = np.random.randn(*shape).astype(dtype)
        elif dtype == np.bool_:
            arr = (np.random.rand(*shape) > 0.5).astype(np.bool_)
        else:
            arr = np.random.randint(0, 10, size=shape).astype(dtype)
        feeds[inp.name] = arr
    return feeds


def _extract_kernel_time_avg_by_node_ms(profile_json_path: str, loops: int) -> Dict[str, float]:
    with open(profile_json_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    per_node_us = defaultdict(list)
    for e in events:
        if not isinstance(e, dict):
            continue
        if e.get("cat") != "Node":
            continue
        name = str(e.get("name", ""))
        if not name.endswith("_kernel_time"):
            continue
        dur = e.get("dur", None)
        if dur is None:
            continue
        try:
            dur_us = float(dur)
        except Exception:
            continue
        node = name[:-12]  # trim suffix "_kernel_time"
        per_node_us[node].append(dur_us)

    # 仅统计最后 loops 次，等价剔除 warmup 后的 loop 区间
    avg_by_node_ms: Dict[str, float] = {}
    for node, seq in per_node_us.items():
        tail = seq[-loops:] if len(seq) >= loops else seq
        if not tail:
            continue
        avg_by_node_ms[node] = (sum(tail) / float(len(tail))) / 1000.0
    return avg_by_node_ms


def _measure_direct_model_latency(
    model_path: str, providers: List, warmup: int, loops: int
) -> float:
    """测量整模型直接执行的平均时延（ms）"""
    sess = ort.InferenceSession(model_path, providers=providers)
    feeds = _prepare_inputs(sess)

    for _ in range(max(warmup, 0)):
        sess.run(None, feeds)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    t0 = time.perf_counter()
    for _ in range(max(loops, 1)):
        sess.run(None, feeds)
    t1 = time.perf_counter()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    avg_ms = ((t1 - t0) / max(loops, 1)) * 1000.0
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="ORT profiler per-kernel average time")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--warmup", type=int, default=50, help="预热次数")
    parser.add_argument("--loops", type=int, default=50, help="循环次数")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default="/cyq/hase/debug/ort_profile_kernel_sum",
        help="ORT profiling 文件名前缀",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        raise ValueError("模型文件不存在: {}".format(model_path))

    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = args.profile_prefix

    providers = [
        ("CUDAExecutionProvider", {"device_id": str(args.device_id)}),
        "CPUExecutionProvider",
    ]

    # 先测量整模型直接执行时间
    direct_model_ms = _measure_direct_model_latency(
        model_path=model_path,
        providers=providers,
        warmup=args.warmup,
        loops=args.loops,
    )

    # 再用 profiling 模式跑一次，提取各 kernel 时间
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    feeds = _prepare_inputs(sess)

    for _ in range(max(args.warmup, 0)):
        sess.run(None, feeds)

    for _ in range(max(args.loops, 1)):
        sess.run(None, feeds)

    # 确保 profiling 文件已落盘
    profile_path = sess.end_profiling()
    time.sleep(0.02)

    avg_by_node_ms = _extract_kernel_time_avg_by_node_ms(profile_path, loops=max(args.loops, 1))
    
    # 输出每个 kernel 的平均时间
    for node_name, avg_ms in sorted(avg_by_node_ms.items(), key=lambda x: x[1], reverse=True):
        print("{}\t{:.6f}".format(node_name, avg_ms))
    
    # 计算所有 kernel 平均时间的总和
    kernel_sum_ms = sum(avg_by_node_ms.values())
    
    # 输出对比结果
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print("  Direct Model Latency (ms):     {:.6f}".format(direct_model_ms))
    print("  Kernel Sum Latency (ms):       {:.6f}".format(kernel_sum_ms))
    print("  Difference (ms):               {:.6f}".format(kernel_sum_ms - direct_model_ms))
    print("  KernelSum / Model Ratio:       {:.6f}".format(kernel_sum_ms / direct_model_ms if direct_model_ms > 0 else 0.0))
    print("=" * 60)


if __name__ == "__main__":
    main()
