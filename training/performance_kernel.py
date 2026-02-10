# -*- coding: utf-8 -*-
"""
Task 5.1: 硬件性能特征算子定义与计时

定义计算密集型和显存密集型算子，并在当前硬件上测量执行时间。
结果会保存为 JSON，供后续训练阶段读取。
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch


def _measure_torch_op(op_fn, warmup=10, loops=50, use_cuda=False):
    """测量单个 Torch 算子的平均执行时间（毫秒）。"""
    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(warmup):
            op_fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(loops):
            starter.record()
            op_fn()
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
        return float(sum(times) / max(len(times), 1))

    for _ in range(warmup):
        op_fn()
    t0 = time.perf_counter()
    for _ in range(loops):
        op_fn()
    t1 = time.perf_counter()
    return ((t1 - t0) / max(loops, 1)) * 1000.0


def build_performance_kernels(device):
    """
    返回硬件特征算子定义。
    - compute_intensive: 计算密集
    - memory_intensive: 显存/内存带宽密集
    """
    kernels = {}

    # 计算密集: 大矩阵乘
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    kernels["compute_matmul_fp32"] = lambda: torch.matmul(a, b)

    # 计算密集: 卷积
    x = torch.randn(16, 64, 112, 112, device=device, dtype=torch.float32)
    w = torch.randn(128, 64, 3, 3, device=device, dtype=torch.float32)
    kernels["compute_conv2d_fp32"] = lambda: torch.nn.functional.conv2d(
        x, w, stride=1, padding=1
    )

    # 显存密集: 元素级加法
    m1 = torch.randn(16, 1024, 1024, device=device, dtype=torch.float32)
    m2 = torch.randn(16, 1024, 1024, device=device, dtype=torch.float32)
    kernels["memory_elementwise_add_fp32"] = lambda: m1 + m2

    # 显存密集: 显式拷贝
    src = torch.randn(16, 1024, 1024, device=device, dtype=torch.float32)
    kernels["memory_clone_fp32"] = lambda: src.clone()

    return kernels


def collect_performance_kernel_times(warmup=10, loops=50):
    """采集硬件性能特征算子的执行时间。"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    kernel_map = build_performance_kernels(device)
    results = {}
    for name, op_fn in kernel_map.items():
        avg_ms = _measure_torch_op(op_fn, warmup=warmup, loops=loops, use_cuda=use_cuda)
        results[name] = round(avg_ms, 6)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "warmup": warmup,
        "loops": loops,
        "latency_ms": results,
    }
    if use_cuda:
        payload["gpu_name"] = torch.cuda.get_device_name(0)
    return payload


def save_results(results, output_path):
    """保存测试结果到 JSON。"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Measure hardware performance kernels.")
    parser.add_argument("--warmup", type=int, default=10, help="warmup loops")
    parser.add_argument("--loops", type=int, default=50, help="measure loops")
    parser.add_argument(
        "--output",
        type=str,
        default="training/performance_kernel_times.json",
        help="output json path",
    )
    args = parser.parse_args()

    results = collect_performance_kernel_times(warmup=args.warmup, loops=args.loops)
    save_results(results, args.output)

    print("Performance kernel times saved to:", args.output)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
