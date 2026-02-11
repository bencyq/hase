# -*- coding: utf-8 -*-
"""
精简版 ORT profiler 解析脚本：
- 运行模型并生成 profile（或读取已有 profile）
- 输出 model_run / execute / node timeline 的分层耗时
"""
import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

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


def _run_direct_model_latency_ms(model_path: str, providers: List, warmup: int, loops: int) -> float:
    sess = ort.InferenceSession(model_path, providers=providers)
    feeds = _prepare_inputs(sess)

    for _ in range(max(warmup, 0)):
        sess.run(None, feeds)

    t0 = time.perf_counter()
    for _ in range(max(loops, 1)):
        sess.run(None, feeds)
    t1 = time.perf_counter()
    return ((t1 - t0) / max(loops, 1)) * 1000.0


def _generate_profile(model_path: str, providers: List, warmup: int, loops: int, profile_prefix: str) -> str:
    so = ort.SessionOptions()
    so.enable_profiling = True
    so.profile_file_prefix = profile_prefix

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    feeds = _prepare_inputs(sess)
    for _ in range(max(warmup, 0)):
        sess.run(None, feeds)
    for _ in range(max(loops, 1)):
        sess.run(None, feeds)

    out = sess.end_profiling()
    time.sleep(0.02)
    return out


def _parse_profile_breakdown_ms(profile_json: str, loops: int) -> Dict[str, float]:
    events = json.load(open(profile_json, "r", encoding="utf-8"))

    model_runs_us = []
    executes_us = []
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "Session":
            continue
        name = e.get("name")
        dur = e.get("dur")
        ts = e.get("ts")
        if dur is None or ts is None:
            continue
        try:
            dur_us = float(dur)
            ts_us = float(ts)
        except Exception:
            continue
        if name == "model_run":
            model_runs_us.append((ts_us, ts_us + dur_us, dur_us))
        elif name == "SequentialExecutor::Execute":
            executes_us.append((ts_us, ts_us + dur_us, dur_us))

    node_order = []
    node_seen = set()
    runs = defaultdict(lambda: defaultdict(lambda: {"before": None, "after": None, "kernel": None}))
    run_idx = defaultdict(int)

    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "Node":
            continue
        name = str(e.get("name", ""))
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None:
            continue
        try:
            ts_us = float(ts)
        except Exception:
            continue

        if name.endswith("_fence_before"):
            node = name[:-13]
            rid = run_idx[node]
            runs[node][rid]["before"] = ts_us
            if node not in node_seen:
                node_seen.add(node)
                node_order.append(node)
        elif name.endswith("_fence_after"):
            node = name[:-12]
            rid = run_idx[node]
            try:
                end_us = ts_us + (float(dur) if dur is not None else 0.0)
            except Exception:
                end_us = ts_us
            runs[node][rid]["after"] = end_us
            run_idx[node] += 1
        elif name.endswith("_kernel_time"):
            node = name[:-12]
            rid = run_idx[node]
            try:
                runs[node][rid]["kernel"] = float(dur) if dur is not None else None
            except Exception:
                runs[node][rid]["kernel"] = None

    max_runs = max((len(runs[n]) for n in node_order), default=0)
    n = min(max(loops, 1), len(model_runs_us), len(executes_us), max_runs)
    if n <= 0:
        raise ValueError("无法从 profile 中解析出足够的 run 数据")

    start_m = len(model_runs_us) - n
    start_e = len(executes_us) - n
    start_r = max_runs - n

    sum_model_us = 0.0
    sum_execute_us = 0.0
    sum_run_outside_execute_us = 0.0
    sum_execute_before_us = 0.0
    sum_execute_after_us = 0.0
    sum_span_us = 0.0
    sum_kernel_us = 0.0
    sum_node_nonkernel_us = 0.0
    sum_gap_us = 0.0
    sum_execute_outside_nodes_us = 0.0

    for i in range(n):
        m_start, m_end, m_dur = model_runs_us[start_m + i]
        e_start, e_end, e_dur = executes_us[start_e + i]
        rid = start_r + i

        sum_model_us += m_dur
        sum_execute_us += e_dur
        sum_run_outside_execute_us += max(0.0, m_dur - e_dur)
        sum_execute_before_us += max(0.0, e_start - m_start)
        sum_execute_after_us += max(0.0, m_end - e_end)

        node_intervals = []
        kernel_us = 0.0
        for node in node_order:
            item = runs[node].get(rid, {})
            b = item.get("before")
            a = item.get("after")
            k = item.get("kernel")
            if b is None or a is None or a < b:
                continue
            node_intervals.append((b, a))
            if k is not None:
                kernel_us += k

        if len(node_intervals) >= 2:
            span_us = node_intervals[-1][1] - node_intervals[0][0]
            wrapper_us = sum((a - b) for b, a in node_intervals)
            gap_us = 0.0
            for j in range(len(node_intervals) - 1):
                prev_after = node_intervals[j][1]
                next_before = node_intervals[j + 1][0]
                if next_before > prev_after:
                    gap_us += (next_before - prev_after)
            node_nonkernel_us = max(0.0, wrapper_us - kernel_us)
        else:
            span_us = 0.0
            gap_us = 0.0
            node_nonkernel_us = 0.0

        sum_span_us += span_us
        sum_kernel_us += kernel_us
        sum_node_nonkernel_us += node_nonkernel_us
        sum_gap_us += gap_us
        sum_execute_outside_nodes_us += max(0.0, e_dur - span_us)

    denom = 1000.0 * float(n)
    return {
        "model_run": sum_model_us / denom,
        "SequentialExecutor::Execute": sum_execute_us / denom,
        "node_timeline_span": sum_span_us / denom,
        "kernel_time_sum": sum_kernel_us / denom,
        "node内非kernel": sum_node_nonkernel_us / denom,
        "node间gap": sum_gap_us / denom,
        "Execute内节点外": sum_execute_outside_nodes_us / denom,
        "model_run内Execute外": sum_run_outside_execute_us / denom,
        "Execute前": sum_execute_before_us / denom,
        "Execute后": sum_execute_after_us / denom,
    }


def main():
    parser = argparse.ArgumentParser(description="ORT profile breakdown")
    parser.add_argument("--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("--warmup", type=int, default=50, help="预热次数")
    parser.add_argument("--loops", type=int, default=50, help="循环次数")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--profile-json", type=str, default="", help="可选：已有 profile json")
    parser.add_argument(
        "--profile-prefix",
        type=str,
        default="/cyq/hase/debug/ort_profile_kernel_sum",
        help="自动生成 profile 时的前缀",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        raise ValueError("模型文件不存在: {}".format(model_path))

    providers = [
        ("CUDAExecutionProvider", {"device_id": str(args.device_id)}),
        "CPUExecutionProvider",
    ]

    # 保留模型真实执行时间（如需可用于对比）
    _ = _run_direct_model_latency_ms(
        model_path=model_path,
        providers=providers,
        warmup=args.warmup,
        loops=args.loops,
    )

    if args.profile_json:
        profile_path = os.path.abspath(args.profile_json)
        if not os.path.isfile(profile_path):
            raise ValueError("profile json 不存在: {}".format(profile_path))
    else:
        profile_path = _generate_profile(
            model_path=model_path,
            providers=providers,
            warmup=args.warmup,
            loops=args.loops,
            profile_prefix=args.profile_prefix,
        )

    out = _parse_profile_breakdown_ms(profile_json=profile_path, loops=max(args.loops, 1))
    ordered_keys = [
        "model_run",
        "SequentialExecutor::Execute",
        "node_timeline_span",
        "kernel_time_sum",
        "node内非kernel",
        "node间gap",
        "Execute内节点外",
        "model_run内Execute外",
        "Execute前",
        "Execute后",
    ]
    for k in ordered_keys:
        print("{} = {:.5f} ms".format(k, out[k]))


if __name__ == "__main__":
    main()
