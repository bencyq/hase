# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import csv
import time
import json
import logging
import argparse
import subprocess
import multiprocessing as mp
import queue as pyqueue
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


def _get_gpu_name(device_id):
    """
    获取指定 GPU 设备的型号名称。
    使用 nvidia-smi 命令查询。
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader,nounits",
            f"--id={device_id}"
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        gpu_name = result.stdout.strip()
        if gpu_name:
            return gpu_name
    except Exception as e:
        logger.warning(f"无法获取 GPU {device_id} 型号: {e}")
    return "Unknown"


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


def _detect_visible_devices():
    """返回当前可见 GPU 设备 id 列表（如 [0,1,2]）"""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass
    return [0]


def _split_levels(stress_levels, device_ids):
    """按轮询把 stress_levels 分配到多卡，保证每个 level 只跑一次"""
    buckets = {d: [] for d in device_ids}
    for idx, level in enumerate(stress_levels):
        d = device_ids[idx % len(device_ids)]
        buckets[d].append(level)
    return buckets


def _render_progress_line(progress_by_device):
    """渲染单行汇总进度（由主进程调用）"""
    total_done = 0
    total_all = 0
    parts = []
    for dev in sorted(progress_by_device.keys()):
        cur = int(progress_by_device[dev].get("current", 0))
        tot = int(progress_by_device[dev].get("total", 0))
        total_done += cur
        total_all += tot
        parts.append("GPU{} {}/{}".format(dev, cur, tot))

    if total_all <= 0:
        percent = 0.0
    else:
        percent = (float(total_done) / float(total_all)) * 100.0
    bar_len = 24
    filled = int(bar_len * percent / 100.0)
    bar = "#" * filled + "-" * (bar_len - filled)
    return "[{}] {:6.2f}% ({}/{}) | {}".format(
        bar, percent, total_done, total_all, " | ".join(parts)
    )


def _collect_levels_on_device(config_path, kernel_dir, output_csv, device_id, stress_levels,
                              warmup, loops, settle, metric_window, between_kernels,
                              progress_queue=None):
    collector = PrometheusCollector(config_path=config_path, device_id=device_id)
    kernel_files = _list_kernel_files(kernel_dir)
    if not kernel_files:
        logger.error("[GPU %d] 未找到 kernel ONNX: %s", device_id, kernel_dir)
        return None

    total_levels = len(stress_levels)
    total_kernels = len(kernel_files)
    total_tasks = total_levels * total_kernels
    gpu_name = _get_gpu_name(device_id)
    if progress_queue is not None:
        progress_queue.put({
            "type": "init",
            "device_id": device_id,
            "current": 0,
            "total": total_tasks,
        })

    metrics_keys = list(collector.metrics_map.keys()) + ["sm_occupancy_when_active"]
    columns = ["GPU"] + \
              ["Kernel_ID", "OpType", "Input_Shape"] + \
              ["DCGM_{}".format(k) for k in metrics_keys] + \
              ["Latency_ms"]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    task_counter = 0

    logger.debug("[GPU %d] 开始执行: levels=%d, kernels=%d, tasks=%d, 型号=%s",
                 device_id, total_levels, total_kernels, total_tasks, gpu_name)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for level_idx, level in enumerate(stress_levels, 1):
            logger.debug("[GPU %d][Level %d/%d] 压力: sm_active=%.2f sm_occ=%.2f dram=%.2f",
                         device_id, level_idx, total_levels,
                         level["sm_active"], level["sm_occ"], level["dram"])
            proc = _start_stressor(device_id, level)
            time.sleep(settle)

            # 每个 stress level 仅采集一次背景指标
            pre_end = time.time()
            pre_start = pre_end - max(metric_window, 1)
            level_stats = collector.get_averages(pre_start, pre_end)

            for kernel_idx, kernel_path in enumerate(kernel_files, 1):
                task_counter += 1
                kernel_id = _parse_kernel_id(kernel_path)
                op_type = _parse_op_type(kernel_id)
                input_shape = _get_input_shape_str(kernel_path)

                logger.debug("[GPU %d][Level %d/%d][Kernel %d/%d][Task %d/%d] %s",
                             device_id, level_idx, total_levels,
                             kernel_idx, total_kernels, task_counter, total_tasks, kernel_id)

                if between_kernels > 0:
                    time.sleep(between_kernels)

                avg_ms, start_dt, end_dt = run_kernel(
                    kernel_path, warmup=warmup, loops=loops, use_warmup=True
                )

                row = {
                    "GPU": gpu_name,
                    "Kernel_ID": kernel_id,
                    "OpType": op_type,
                    "Input_Shape": input_shape,
                    "Latency_ms": avg_ms,
                }
                for k in metrics_keys:
                    row["DCGM_{}".format(k)] = level_stats.get(k, 0.0)
                writer.writerow(row)
                if progress_queue is not None:
                    progress_queue.put({
                        "type": "progress",
                        "device_id": device_id,
                        "current": task_counter,
                        "total": total_tasks,
                        "kernel_id": kernel_id,
                    })

            _stop_stressor(proc)

    if progress_queue is not None:
        progress_queue.put({
            "type": "done",
            "device_id": device_id,
            "current": task_counter,
            "total": total_tasks,
        })
    logger.debug("[GPU %d] 完成并输出: %s", device_id, output_csv)
    return output_csv


def _merge_csv_files(part_files, output_csv):
    if not part_files:
        return None
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as out_f:
        writer = None
        for path in part_files:
            with open(path, "r", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    return output_csv



def collect_data(config_path, kernel_dir, output_csv, device_ids=None):
    overall_start_time = datetime.now()
    cfg = _load_config(config_path)
    stress_levels = _get_stress_levels(cfg)
    warmup, loops, settle, metric_window, between_kernels = _get_runner_params(cfg)

    if device_ids is None:
        device_ids = _detect_visible_devices()
    elif isinstance(device_ids, int):
        device_ids = [device_ids]
    else:
        device_ids = [int(d) for d in device_ids]
    device_ids = sorted(set(device_ids))
    if not device_ids:
        device_ids = [0]

    logger.info("=" * 80)
    logger.info("数据采集开始: %s", overall_start_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("设备列表: %s", device_ids)
    logger.info("stress levels: %d", len(stress_levels))

    # 单卡/多卡统一走 worker + queue，主进程汇总进度
    level_buckets = _split_levels(stress_levels, device_ids)
    part_files = []
    processes = []
    progress_queue = mp.Queue()
    progress_by_device = {}
    finished_workers = 0
    refresh_interval = 0.2
    last_print_ts = 0.0
    last_snapshot = None
    last_line_len = 0

    for dev in device_ids:
        dev_levels = level_buckets.get(dev) or []
        if not dev_levels:
            continue
        part_csv = "{}.gpu{}.part.csv".format(
            output_csv[:-4] if output_csv.endswith(".csv") else output_csv, dev
        )
        part_files.append(part_csv)
        p = mp.Process(
            target=_collect_levels_on_device,
            args=(config_path, kernel_dir, part_csv, dev, dev_levels,
                  warmup, loops, settle, metric_window, between_kernels, progress_queue)
        )
        p.start()
        processes.append(p)
        logger.info("启动 GPU %d worker, 分配 levels=%d, 输出=%s", dev, len(dev_levels), part_csv)

    # 主进程消费进度并节流刷新 stdout
    while finished_workers < len(processes):
        try:
            msg = progress_queue.get(timeout=0.3)
        except pyqueue.Empty:
            msg = None

        if msg:
            dev = msg.get("device_id")
            cur = int(msg.get("current", 0))
            tot = int(msg.get("total", 0))
            progress_by_device.setdefault(dev, {"current": 0, "total": 0})
            progress_by_device[dev]["current"] = cur
            if tot > 0:
                progress_by_device[dev]["total"] = tot
            if msg.get("type") == "done":
                finished_workers += 1

        now = time.time()
        if progress_by_device:
            snapshot = tuple(
                (dev, progress_by_device[dev].get("current", 0), progress_by_device[dev].get("total", 0))
                for dev in sorted(progress_by_device.keys())
            )
            if snapshot != last_snapshot and (now - last_print_ts >= refresh_interval):
                line = "进度: " + _render_progress_line(progress_by_device)
                pad = max(0, last_line_len - len(line))
                sys.stdout.write("\r" + line + (" " * pad))
                sys.stdout.flush()
                last_print_ts = now
                last_snapshot = snapshot
                last_line_len = len(line)

        # 若 worker 异常退出导致没有 done 消息，避免主循环卡住
        if all(not p.is_alive() for p in processes):
            break

    if progress_by_device:
        line = "进度: " + _render_progress_line(progress_by_device)
        pad = max(0, last_line_len - len(line))
        sys.stdout.write("\r" + line + (" " * pad) + "\n")
        sys.stdout.flush()

    failed = False
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed = True
            logger.error("worker 进程失败, pid=%d, exitcode=%s", p.pid, p.exitcode)

    if failed:
        logger.error("多卡采集失败，未合并 CSV")
        return None

    _merge_csv_files(part_files, output_csv)
    for pf in part_files:
        try:
            os.remove(pf)
        except Exception:
            pass

    overall_end_time = datetime.now()
    logger.info("数据采集完成: %s, 总耗时 %.1f 秒",
                output_csv, (overall_end_time - overall_start_time).total_seconds())
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
    parser.add_argument("--device", type=int, default=None, help="单卡模式 GPU 设备 ID")
    parser.add_argument("--devices", type=str, default=None,
                        help="多卡列表，如 '0,1,2'；不传则自动使用所有可见 GPU")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(PROJECT_ROOT, "benchmark", "data_{}.csv".format(ts))

    if args.devices:
        device_ids = [int(x.strip()) for x in args.devices.split(",") if x.strip() != ""]
    elif args.device is not None:
        device_ids = [args.device]
    else:
        device_ids = None

    collect_data(args.config, args.kernel_dir, args.output, device_ids)


if __name__ == "__main__":
    main()
