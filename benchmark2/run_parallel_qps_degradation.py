#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型并行退化实验：
1) 单模型 baseline
2) 多模型并行 (每模型多副本)
3) 采集 Prometheus/DCGM 指标并计算 QPS_drop
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.collector import PrometheusCollector, _get_stress_levels, _load_config, _start_stressor, _stop_stressor
from utils.logger import get_logger

logger = get_logger("benchmark2_parallel_qps")
PROVIDERS = ["CUDAExecutionProvider"]


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
    return np.float32


def _normalize_shape(shape: Sequence[int]) -> List[int]:
    out = []
    for d in shape:
        if d is None or d == 0:
            out.append(1)
        else:
            out.append(int(d))
    return out


def _prepare_inputs(session: ort.InferenceSession) -> Dict[str, np.ndarray]:
    feed = {}
    for inp in session.get_inputs():
        shape = _normalize_shape(inp.shape)
        dtype = _to_numpy_dtype(inp.type)
        feed[inp.name] = np.random.randn(*shape).astype(dtype)
    return feed


def _worker_run(model_path: str, warmup: int, duration_sec: float, result_q: mp.Queue, worker_name: str) -> None:
    try:
        sess = ort.InferenceSession(model_path, providers=PROVIDERS)
        feed = _prepare_inputs(sess)

        for _ in range(max(0, warmup)):
            _ = sess.run(None, feed)

        t_end = time.perf_counter() + max(0.1, float(duration_sec))
        cnt = 0
        total_ms = 0.0
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            _ = sess.run(None, feed)
            t1 = time.perf_counter()
            cnt += 1
            total_ms += (t1 - t0) * 1000.0

        avg_ms = total_ms / max(cnt, 1)
        qps = cnt / max(float(duration_sec), 1e-6)
        result_q.put({
            "ok": True,
            "worker_name": worker_name,
            "count": cnt,
            "avg_ms": avg_ms,
            "qps": qps,
            "error": "",
        })
    except Exception as exc:
        result_q.put({
            "ok": False,
            "worker_name": worker_name,
            "count": 0,
            "avg_ms": 0.0,
            "qps": 0.0,
            "error": str(exc),
        })


def _run_scenario(model_paths: Sequence[str], replicas_per_model: int, warmup: int, duration_sec: float) -> List[Dict[str, object]]:
    result_q: mp.Queue = mp.Queue()
    procs: List[mp.Process] = []
    job_map: Dict[str, str] = {}

    for model_path in model_paths:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        for ridx in range(replicas_per_model):
            worker_name = "{}#{}".format(model_name, ridx)
            p = mp.Process(
                target=_worker_run,
                args=(model_path, warmup, duration_sec, result_q, worker_name),
            )
            p.start()
            procs.append(p)
            job_map[worker_name] = model_name

    results = []
    for _ in procs:
        results.append(result_q.get())

    for p in procs:
        p.join()

    merged = []
    for item in results:
        worker_name = item["worker_name"]
        item["model_name"] = job_map.get(worker_name, worker_name.split("#")[0])
        merged.append(item)
    return merged


def _aggregate_by_model(worker_rows: Sequence[Dict[str, object]], duration_sec: float) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    for r in worker_rows:
        model_name = str(r["model_name"])
        agg.setdefault(model_name, {"count": 0.0, "qps": 0.0, "avg_ms_sum": 0.0, "workers": 0.0, "failures": 0.0})
        agg[model_name]["count"] += float(r["count"])
        agg[model_name]["qps"] += float(r["qps"])
        agg[model_name]["avg_ms_sum"] += float(r["avg_ms"])
        agg[model_name]["workers"] += 1.0
        if not bool(r["ok"]):
            agg[model_name]["failures"] += 1.0

    for model_name, d in agg.items():
        d["avg_ms"] = d["avg_ms_sum"] / max(d["workers"], 1.0)
        d["qps_recomputed"] = d["count"] / max(duration_sec, 1e-6)
    return agg


def _format_level_key(level: Dict[str, float]) -> str:
    return "sa{:.2f}_so{:.2f}_dr{:.2f}".format(level["sm_active"], level["sm_occ"], level["dram"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark2 多模型并行 QPS 退化实验")
    parser.add_argument("--models-dir", type=str, default=os.path.join(PROJECT_ROOT, "benchmark2"), help="ONNX 模型目录")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "benchmark", "config.yaml"), help="collector/stressor 配置")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--warmup", type=int, default=5, help="每个 worker warmup 次数")
    parser.add_argument("--duration", type=float, default=20.0, help="每个场景正式测量时长(秒)")
    parser.add_argument("--settle-sec", type=float, default=10.0, help="启动 stressor 后稳定等待秒数")
    parser.add_argument("--metric-window-sec", type=float, default=8.0, help="Prometheus 指标平均窗口")
    parser.add_argument("--replicas", type=str, default="1,2", help="并行副本阶梯，例如 '1,2,3'")
    parser.add_argument("--max-stress-levels", type=int, default=3, help="最多使用前 N 个 stress level")
    parser.add_argument("--output", type=str, default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    model_paths = sorted(
        os.path.join(args.models_dir, x)
        for x in os.listdir(args.models_dir)
        if x.endswith(".onnx")
    )
    if not model_paths:
        raise RuntimeError("未找到 ONNX 模型: {}".format(args.models_dir))

    cfg = _load_config(args.config)
    levels = _get_stress_levels(cfg)
    if args.max_stress_levels > 0:
        levels = levels[: args.max_stress_levels]
    if not levels:
        levels = [{"sm_active": 0.0, "sm_occ": 0.0, "dram": 0.0}]

    replicas_list = [int(x.strip()) for x in args.replicas.split(",") if x.strip()]
    replicas_list = sorted(set([x for x in replicas_list if x > 0]))
    if 1 not in replicas_list:
        replicas_list = [1] + replicas_list

    output = args.output
    if not output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(PROJECT_ROOT, "benchmark2", "parallel_qps_degradation_{}.csv".format(ts))
    os.makedirs(os.path.dirname(output), exist_ok=True)

    prom = PrometheusCollector(config_path=args.config, device_id=args.device)
    baseline_map: Dict[Tuple[str, str], float] = {}
    rows: List[Dict[str, object]] = []

    logger.info("实验开始: models=%d levels=%d replicas=%s", len(model_paths), len(levels), replicas_list)
    logger.info("输出文件: %s", output)

    for level_idx, level in enumerate(levels, 1):
        level_key = _format_level_key(level)
        logger.info("Level %d/%d: %s", level_idx, len(levels), level_key)
        proc = _start_stressor(args.device, level)
        try:
            time.sleep(max(0.0, args.settle_sec))

            # Phase A: baseline，逐模型独占
            for model_path in model_paths:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                worker_rows = _run_scenario([model_path], replicas_per_model=1, warmup=args.warmup, duration_sec=args.duration)
                agg = _aggregate_by_model(worker_rows, args.duration).get(model_name, {})
                qps = float(agg.get("qps_recomputed", 0.0))
                baseline_map[(level_key, model_name)] = qps

                t_end = time.time()
                t_start = t_end - max(1.0, args.metric_window_sec)
                metrics = prom.get_averages(t_start, t_end)

                rows.append({
                    "level_key": level_key,
                    "phase": "baseline_single",
                    "replicas_per_model": 1,
                    "model_name": model_name,
                    "duration_sec": args.duration,
                    "count": int(agg.get("count", 0)),
                    "qps": qps,
                    "avg_ms": float(agg.get("avg_ms", 0.0)),
                "workers_total": int(agg.get("workers", 0)),
                "workers_failed": int(agg.get("failures", 0)),
                    "qps_baseline": qps,
                    "qps_drop": 0.0,
                    "dcgm_sm_active": metrics.get("sm_active", 0.0),
                    "dcgm_sm_occupancy": metrics.get("sm_occupancy", 0.0),
                    "dcgm_dram_active": metrics.get("dram_active", 0.0),
                })

            # Phase B: 多模型并行，阶梯副本
            for replicas in replicas_list:
                worker_rows = _run_scenario(model_paths, replicas_per_model=replicas, warmup=args.warmup, duration_sec=args.duration)
                agg_map = _aggregate_by_model(worker_rows, args.duration)

                t_end = time.time()
                t_start = t_end - max(1.0, args.metric_window_sec)
                metrics = prom.get_averages(t_start, t_end)

                for model_path in model_paths:
                    model_name = os.path.splitext(os.path.basename(model_path))[0]
                    agg = agg_map.get(model_name, {})
                    qps = float(agg.get("qps_recomputed", 0.0))
                    baseline_qps = float(baseline_map.get((level_key, model_name), 0.0))
                    if baseline_qps > 0:
                        qps_drop = (baseline_qps - qps) / baseline_qps
                    else:
                        qps_drop = 0.0

                    rows.append({
                        "level_key": level_key,
                        "phase": "parallel_mixed",
                        "replicas_per_model": replicas,
                        "model_name": model_name,
                        "duration_sec": args.duration,
                        "count": int(agg.get("count", 0)),
                        "qps": qps,
                        "avg_ms": float(agg.get("avg_ms", 0.0)),
                        "workers_total": int(agg.get("workers", 0)),
                        "workers_failed": int(agg.get("failures", 0)),
                        "qps_baseline": baseline_qps,
                        "qps_drop": qps_drop,
                        "dcgm_sm_active": metrics.get("sm_active", 0.0),
                        "dcgm_sm_occupancy": metrics.get("sm_occupancy", 0.0),
                        "dcgm_dram_active": metrics.get("dram_active", 0.0),
                    })
        finally:
            _stop_stressor(proc)

    fieldnames = [
        "level_key",
        "phase",
        "replicas_per_model",
        "model_name",
        "duration_sec",
        "count",
        "qps",
        "avg_ms",
        "workers_total",
        "workers_failed",
        "qps_baseline",
        "qps_drop",
        "dcgm_sm_active",
        "dcgm_sm_occupancy",
        "dcgm_dram_active",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logger.info("实验完成，已写入: %s", output)


if __name__ == "__main__":
    main()
