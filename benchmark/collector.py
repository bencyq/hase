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
    def __init__(self, config_path="nn_meter/hase/config.yaml", device_id=0):
        self.url = "http://localhost:9090"
        self.device_id = device_id
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
        # Filter by device_id using gpu label
        stats: dict = {}
        for key, metric_name in self.metrics_map.items():
            promql = f'avg_over_time({metric_name}{{gpu="{self.device_id}"}}[{window}])'
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
    overall_start_time = datetime.now()
    logger.info("")
    logger.info("=" * 80)
    logger.info("数据采集开始")
    logger.info("  开始时间: %s", overall_start_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  配置文件: %s", config_path)
    logger.info("  Kernel 目录: %s", kernel_dir)
    logger.info("  输出 CSV: %s", output_csv)
    logger.info("  GPU 设备: %d", device_id)
    
    cfg = _load_config(config_path)
    stress_levels = _get_stress_levels(cfg)
    warmup, loops, settle, metric_window, between_kernels = _get_runner_params(cfg)

    collector = PrometheusCollector(config_path=config_path, device_id=device_id)
    kernel_files = _list_kernel_files(kernel_dir)
    if not kernel_files:
        logger.error("未找到 kernel ONNX: %s", kernel_dir)
        return None

    total_levels = len(stress_levels)
    total_kernels = len(kernel_files)
    total_tasks = total_levels * total_kernels
    
    logger.info("=" * 80)
    logger.info("数据采集配置:")
    logger.info("  - Stress levels: %d 个", total_levels)
    logger.info("  - Kernel 文件: %d 个", total_kernels)
    logger.info("  - 总任务数: %d (levels × kernels)", total_tasks)
    logger.info("  - Warmup: %d, Loops: %d", warmup, loops)
    logger.info("  - SETTLE_SEC: %.1f, METRIC_WINDOW_SEC: %.1f, BETWEEN_KERNEL_SEC: %.1f",
                settle, metric_window, between_kernels)
    logger.info("=" * 80)

    # CSV header
    metrics_keys = list(collector.metrics_map.keys()) + ["sm_occupancy_when_active"]
    columns = ["Kernel_ID", "OpType", "Input_Shape"] + \
              ["DCGM_{}".format(k) for k in metrics_keys] + \
              ["Latency_ms"]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    task_counter = 0
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for level_idx, level in enumerate(stress_levels, 1):
            level_start_time = datetime.now()
            logger.info("")
            logger.info("─" * 80)
            logger.info("[Level %d/%d] 开始设置压力: sm_active=%.2f, sm_occ=%.2f, dram=%.2f",
                       level_idx, total_levels, level["sm_active"], level["sm_occ"], level["dram"])
            logger.info("  时间: %s", level_start_time.strftime("%Y-%m-%d %H:%M:%S"))
            
            proc = _start_stressor(device_id, level)
            if proc:
                logger.info("  Stressor 进程已启动 (PID: %d)", proc.pid)
            else:
                logger.info("  无背景负载 (所有参数为 0)")
            
            logger.info("  等待稳定期: %.1f 秒...", settle)
            time.sleep(settle)

            # 每个 stress level 仅采集一次背景负载指标
            logger.info("  采集 DCGM 背景负载指标 (窗口: %.1f 秒)...", metric_window)
            pre_end = time.time()
            pre_start = pre_end - max(metric_window, 1)
            level_stats = collector.get_averages(pre_start, pre_end)
            logger.info("  DCGM 指标: sm_active=%.3f, sm_occupancy=%.3f, dram_active=%.3f",
                       level_stats.get("sm_active", 0.0),
                       level_stats.get("sm_occupancy", 0.0),
                       level_stats.get("dram_active", 0.0))

            for kernel_idx, kernel_path in enumerate(kernel_files, 1):
                task_counter += 1
                kernel_id = _parse_kernel_id(kernel_path)
                op_type = _parse_op_type(kernel_id)
                input_shape = _get_input_shape_str(kernel_path)
                
                kernel_start_time = datetime.now()
                
                # 计算进度百分比
                progress_percent = (task_counter / total_tasks) * 100
                progress_bar_length = 30
                filled = int(progress_bar_length * task_counter / total_tasks)
                bar = '█' * filled + '░' * (progress_bar_length - filled)
                
                # 输出进度条到 stdout
                sys.stdout.write(f'\r进度: [{bar}] {progress_percent:.1f}% [{task_counter}/{total_tasks}] - {kernel_id}')
                sys.stdout.flush()

                # 等待背景负载稳定
                if between_kernels > 0:
                    # logger.info("    等待背景负载稳定: %.1f 秒...", between_kernels)
                    time.sleep(between_kernels)

                # logger.info("    执行推理 (Warmup=%d, Loops=%d)...", warmup, loops)
                avg_ms, start_dt, end_dt = run_kernel(
                    kernel_path, warmup=warmup, loops=loops, use_warmup=True
                )
                
                # kernel_end_time = datetime.now()
                # elapsed = (kernel_end_time - kernel_start_time).total_seconds()
                # logger.info("    推理完成: 平均耗时=%.3f ms, 执行时间=%.1f 秒",
                #            avg_ms, elapsed)
                # logger.info("    结束时间: %s", kernel_end_time.strftime("%Y-%m-%d %H:%M:%S"))

                row = {
                    "Kernel_ID": kernel_id,
                    "OpType": op_type,
                    "Input_Shape": input_shape,
                    "Latency_ms": avg_ms,
                }
                for k in metrics_keys:
                    row["DCGM_{}".format(k)] = level_stats.get(k, 0.0)

                writer.writerow(row)

            level_end_time = datetime.now()
            level_elapsed = (level_end_time - level_start_time).total_seconds()
            logger.info("")
            logger.info("[Level %d/%d] 完成，耗时: %.1f 秒", level_idx, total_levels, level_elapsed)
            logger.info("  停止 Stressor...")
            _stop_stressor(proc)
            logger.info("  Stressor 已停止")

    overall_end_time = datetime.now()
    overall_elapsed = (overall_end_time - overall_start_time).total_seconds()
    logger.info("")
    logger.info("=" * 80)
    logger.info("数据采集完成")
    logger.info("  结束时间: %s", overall_end_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  总耗时: %.1f 秒 (%.1f 分钟)", overall_elapsed, overall_elapsed / 60)
    logger.info("  完成任务: %d/%d", task_counter, total_tasks)
    logger.info("  CSV 输出: %s", output_csv)
    logger.info("=" * 80)
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
