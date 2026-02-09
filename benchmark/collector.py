# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import csv
import time
import json
import logging
import argparse
import subprocess
from datetime import datetime
from typing import Optional, Dict

import requests
import yaml
import onnx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from benchmark.ort_kernel_runner import run_kernel

logging = logging.getLogger("nn-Meter")
logger = get_logger("collector")

class PrometheusCollector:
    def __init__(self, config_path="nn_meter/hase/config.yaml"):
        self.url = "http://localhost:9090"
        # internal keys: sm_active / sm_occupancy / dram_active
        self.metrics_map = {
            "sm_active": "DCGM_FI_PROF_SM_ACTIVE",
            "sm_occupancy": "DCGM_FI_PROF_SM_OCCUPANCY",
            "dram_active": "DCGM_FI_PROF_DRAM_ACTIVE",
        }
        
        # Load config if available
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                    if "PROMETHEUS" in cfg:
                        self.url = cfg["PROMETHEUS"].get("URL", self.url)
                        metrics_cfg = cfg["PROMETHEUS"].get("METRICS", {}) or {}
                        # Support config keys in either UPPER_CASE or internal snake_case
                        for k, v in metrics_cfg.items():
                            if not v:
                                continue
                            key = str(k).strip()
                            metric = str(v).strip()
                            if key in ("SM_ACTIVE", "sm_active"):
                                self.metrics_map["sm_active"] = metric
                            elif key in ("SM_OCCUPANCY", "sm_occupancy"):
                                self.metrics_map["sm_occupancy"] = metric
                            elif key in ("DRAM_ACTIVE", "dram_active"):
                                self.metrics_map["dram_active"] = metric
        except Exception as e:
            logging.warning(f"Failed to load Prometheus config: {e}. Using defaults.")

    @staticmethod
    def _window_seconds_to_prom_duration(window_seconds: float) -> str:
        """
        Convert seconds to a PromQL duration string.
        We intentionally keep it in seconds for correctness (e.g., '90s').
        """
        try:
            sec = int(round(float(window_seconds)))
        except Exception:
            sec = 1
        sec = max(1, sec)
        return f"{sec}s"

    def query_instant(self, promql: str, ts: Optional[float] = None) -> Optional[float]:
        """
        Query Prometheus instant API and return the first scalar-like value.
        The averaging/aggregation should be done by Prometheus (PromQL) side.
        """
        try:
            query_url = f"{self.url}/api/v1/query"
            params: Dict[str, object] = {"query": promql}
            if ts is not None:
                params["time"] = ts

            response = requests.get(query_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                return None

            result = (data.get("data") or {}).get("result") or []
            if not result:
                return None

            # result[0]["value"] is [timestamp, "value"]
            value = result[0].get("value")
            if not value or len(value) < 2:
                return None

            v = float(value[1])
            if v != v:  # NaN
                return None
            return v
        except Exception as e:
            logging.error(f"Prometheus instant query failed: promql={promql}, err={e}")
            return None

    def get_averages(self, start_time, end_time):
        """
        Get average values for all configured metrics during the window.
        """
        window = self._window_seconds_to_prom_duration(float(end_time) - float(start_time))

        # Use Prometheus to compute averages: avg_over_time over the time window ending at end_time.
        # Use range vector syntax for better compatibility with Prometheus scrape intervals.
        stats: dict = {}
        for key, metric_name in self.metrics_map.items():
            promql = f"avg_over_time({metric_name}[{window}])"
            v = self.query_instant(promql, ts=end_time)
            stats[key] = float(v) if v is not None else 0.0

        # Derived metric: occupancy_when_active
        # Many PROF metrics are effectively time-window averages; when SM_ACTIVE is low,
        # raw SM_OCCUPANCY will also look low even if "occupancy during active time" is high.
        # This derived value estimates occupancy conditioned on being active.
        sm_active = float(stats.get("sm_active", 0.0) or 0.0)
        sm_occ = float(stats.get("sm_occupancy", 0.0) or 0.0)

        # Heuristic for units:
        # - DCGM often reports percentages in [0, 100]
        # - If values appear in [0, 1], treat them as ratios
        if sm_active > 1.5:
            denom = max(sm_active / 100.0, 1e-6)
            stats["sm_occupancy_when_active"] = sm_occ / denom
        else:
            denom = max(sm_active, 1e-6)
            stats["sm_occupancy_when_active"] = sm_occ / denom

        return stats


def _load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _clamp_ratio(x):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def _get_stress_levels(cfg):
    levels = cfg.get("STRESS_LEVELS") or []
    if not levels:
        # 默认 stress levels
        return [
            {"sm_active": 0.0, "sm_occ": 0.0, "dram": 0.0},
            {"sm_active": 0.3, "sm_occ": 0.3, "dram": 0.3},
            {"sm_active": 0.6, "sm_occ": 0.6, "dram": 0.6},
            {"sm_active": 0.9, "sm_occ": 0.9, "dram": 0.9},
        ]
    fixed = []
    for lv in levels:
        fixed.append({
            "sm_active": _clamp_ratio(lv.get("sm_active", 0.0)),
            "sm_occ": _clamp_ratio(lv.get("sm_occ", 0.0)),
            "dram": _clamp_ratio(lv.get("dram", 0.0)),
        })
    return fixed


def _get_runner_params(cfg):
    runner = cfg.get("RUNNER") or {}
    warmup = int(runner.get("WARMUP", 10))
    loops = int(runner.get("LOOPS", 50))
    settle = float(runner.get("SETTLE_SEC", 1.0))
    metric_window = float(runner.get("METRIC_WINDOW_SEC", 3.0))
    between_kernels = float(runner.get("BETWEEN_KERNEL_SEC", 2.0))
    return warmup, loops, settle, metric_window, between_kernels


def _list_kernel_files(kernel_dir):
    files = []
    for name in os.listdir(kernel_dir):
        if name.endswith(".onnx"):
            files.append(os.path.join(kernel_dir, name))
    return sorted(files)


def _parse_kernel_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _parse_op_type(kernel_id):
    # 约定: kernel_type 在 _bs 前
    if "_bs" in kernel_id:
        return kernel_id.split("_bs")[0]
    return kernel_id


def _get_input_shape_str(model_path):
    try:
        model = onnx.load(model_path)
        if not model.graph.input:
            return ""
        inp = model.graph.input[0]
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value else 0)
        return "x".join(str(d) for d in dims)
    except Exception:
        return ""


def _start_stressor(device_id, level):
    if level["sm_active"] == 0 and level["sm_occ"] == 0 and level["dram"] == 0:
        return None
    stressor_path = os.path.join(PROJECT_ROOT, "benchmark", "stressor.py")
    cmd = [
        sys.executable, stressor_path,
        "--device", str(device_id),
        "--sm-active", str(level["sm_active"]),
        "--sm-occ", str(level["sm_occ"]),
        "--dram", str(level["dram"]),
        "--duration", "0"
    ]
    return subprocess.Popen(cmd)


def _stop_stressor(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


def collect_data(config_path, kernel_dir, output_csv, device_id):
    cfg = _load_config(config_path)
    stress_levels = _get_stress_levels(cfg)
    warmup, loops, settle, metric_window, between_kernels = _get_runner_params(cfg)

    collector = PrometheusCollector(config_path=config_path)
    kernel_files = _list_kernel_files(kernel_dir)
    if not kernel_files:
        logger.error("未找到 kernel ONNX: %s", kernel_dir)
        return None

    logger.info("找到 %d 个 kernel 文件", len(kernel_files))
    logger.info("Stress levels: %s", json.dumps(stress_levels, ensure_ascii=False))

    # CSV header
    metrics_keys = list(collector.metrics_map.keys()) + ["sm_occupancy_when_active"]
    columns = ["Kernel_ID", "OpType", "Input_Shape"] + \
              ["DCGM_{}".format(k) for k in metrics_keys] + \
              ["Latency_ms"]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for level in stress_levels:
            logger.info("设置压力: %s", level)
            proc = _start_stressor(device_id, level)
            time.sleep(settle)

            for kernel_path in kernel_files:
                kernel_id = _parse_kernel_id(kernel_path)
                op_type = _parse_op_type(kernel_id)
                input_shape = _get_input_shape_str(kernel_path)

                # 等待背景负载稳定
                if between_kernels > 0:
                    time.sleep(between_kernels)

                # 先采集背景负载指标（必须早于 kernel 执行）
                pre_end = time.time()
                pre_start = pre_end - max(metric_window, 1)
                stats = collector.get_averages(pre_start, pre_end)

                avg_ms, start_dt, end_dt = run_kernel(
                    kernel_path, warmup=warmup, loops=loops, use_warmup=True
                )

                row = {
                    "Kernel_ID": kernel_id,
                    "OpType": op_type,
                    "Input_Shape": input_shape,
                    "Latency_ms": avg_ms,
                }
                for k in metrics_keys:
                    row["DCGM_{}".format(k)] = stats.get(k, 0.0)

                writer.writerow(row)

            _stop_stressor(proc)

    logger.info("数据采集完成，CSV 输出: %s", output_csv)
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Data Collector Orchestrator")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_ROOT, "benchmark", "config.yaml"),
                        help="配置文件路径")
    parser.add_argument("--kernel-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "kernel_model", "kernel_onnx"),
                        help="kernel ONNX 目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备 ID")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(PROJECT_ROOT, "benchmark", "data_{}.csv".format(ts))

    collect_data(args.config, args.kernel_dir, args.output, args.device)


if __name__ == "__main__":
    main()
