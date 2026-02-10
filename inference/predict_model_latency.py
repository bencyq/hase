# -*- coding: utf-8 -*-
"""
Task 7.1: Predictor Integration

输入:
- model_zoo/models/<model>.onnx
- inference/config.yaml

输出:
- 预测总耗时 (ms)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import requests
import yaml
from joblib import load

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from graph_model.dag_critical_path import (  # noqa: E402
    build_dag_from_kernel_json,
    calculate_latency,
    save_dag,
)
from utils.logger import get_logger  # noqa: E402

logger = get_logger("predict_model_latency")

MODEL_NAME_PATTERN = re.compile(
    r"^(?P<model>[a-zA-Z0-9_]+)_bs(?P<batch>\d+)_(?P<input>\d+)x(?P=input)$"
)


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def _parse_model_filename(model_path: str) -> Tuple[str, int, int, str]:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    m = MODEL_NAME_PATTERN.match(stem)
    if not m:
        raise ValueError(
            "模型文件名需满足 <model>_bs<batch>_<size>x<size>.onnx，当前: {}".format(stem)
        )
    model_name = m.group("model")
    batch = int(m.group("batch"))
    input_size = int(m.group("input"))
    return model_name, batch, input_size, stem


def _query_prometheus_scalar(url: str, query: str, timeout_sec: float) -> Optional[float]:
    api = url.rstrip("/") + "/api/v1/query"
    resp = requests.get(api, params={"query": query}, timeout=timeout_sec)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "success":
        return None
    result = (payload.get("data") or {}).get("result") or []
    if not result:
        return None
    value = result[0].get("value")
    if not value or len(value) < 2:
        return None
    return float(value[1])


def _get_realtime_dcgm(cfg: Dict) -> Dict[str, float]:
    prom = cfg.get("PROMETHEUS", {}) or {}
    prom_url = str(prom.get("URL", "http://localhost:9090"))
    gpu_id = int(prom.get("GPU_ID", 0))
    window_sec = max(1, int(prom.get("QUERY_WINDOW_SEC", 10)))
    timeout_sec = float(prom.get("TIMEOUT_SEC", 3))
    metrics = prom.get("METRICS", {}) or {}

    sm_active_metric = metrics.get("SM_ACTIVE", "DCGM_FI_PROF_SM_ACTIVE")
    sm_occ_metric = metrics.get("SM_OCCUPANCY", "DCGM_FI_PROF_SM_OCCUPANCY")
    dram_metric = metrics.get("DRAM_ACTIVE", "DCGM_FI_PROF_DRAM_ACTIVE")

    def avg_query(metric_name: str) -> str:
        return 'avg_over_time({}{{gpu="{}"}}[{}s])'.format(metric_name, gpu_id, window_sec)

    sm_active = _query_prometheus_scalar(prom_url, avg_query(sm_active_metric), timeout_sec)
    sm_occ = _query_prometheus_scalar(prom_url, avg_query(sm_occ_metric), timeout_sec)
    dram_active = _query_prometheus_scalar(prom_url, avg_query(dram_metric), timeout_sec)

    sm_active = float(sm_active) if sm_active is not None else 0.0
    sm_occ = float(sm_occ) if sm_occ is not None else 0.0
    dram_active = float(dram_active) if dram_active is not None else 0.0

    if sm_active > 1.5:
        denom = max(sm_active / 100.0, 1e-6)
    else:
        denom = max(sm_active, 1e-6)
    sm_occ_when_active = sm_occ / denom

    return {
        "DCGM_sm_active": sm_active,
        "DCGM_sm_occupancy": sm_occ,
        "DCGM_dram_active": dram_active,
        "DCGM_sm_occupancy_when_active": sm_occ_when_active,
    }


def _get_gpu_name_fallback() -> str:
    try:
        cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        if not out:
            return "Unknown"
        return out.splitlines()[0].strip()
    except Exception:
        return "Unknown"


def _load_hwpk_map(perf_json_path: str) -> Dict[str, Dict[str, float]]:
    with open(perf_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload if isinstance(payload, list) else [payload]
    out: Dict[str, Dict[str, float]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        gpu_name = item.get("gpu_name")
        lat = item.get("latency_ms", {})
        if not gpu_name or not isinstance(lat, dict):
            continue
        out[str(gpu_name)] = {
            "hwpk_compute_matmul_fp32": float(lat.get("compute_matmul_fp32", 0.0)),
            "hwpk_compute_conv2d_fp32": float(lat.get("compute_conv2d_fp32", 0.0)),
            "hwpk_memory_elementwise_add_fp32": float(
                lat.get("memory_elementwise_add_fp32", 0.0)
            ),
            "hwpk_memory_clone_fp32": float(lat.get("memory_clone_fp32", 0.0)),
        }
    if not out:
        raise ValueError("性能特征文件缺少有效 gpu_name/latency_ms 映射: {}".format(perf_json_path))
    return out


def _ensure_dag_for_model(model_name: str, dag_dir: str, kernel_record_dir: str) -> str:
    dag_path = os.path.join(dag_dir, "{}_dag.json".format(model_name))
    if os.path.isfile(dag_path):
        return dag_path

    kernel_json = os.path.join(kernel_record_dir, "{}.json".format(model_name))
    if not os.path.isfile(kernel_json):
        raise ValueError("缺少 DAG 且未找到对应 kernel record: {}".format(kernel_json))

    dag = build_dag_from_kernel_json(kernel_json)
    save_dag(dag, dag_path)
    logger.info("自动生成 DAG: %s", dag_path)
    return dag_path


def _load_kernel_json(model_name: str, kernel_record_dir: str) -> Dict:
    path = os.path.join(kernel_record_dir, "{}.json".format(model_name))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_shape_entry(
    kernel_item: Dict, source_model_key: str, batch: int, input_size: int
) -> Dict:
    shapes = kernel_item.get("shapes", []) or []
    for s in shapes:
        if str(s.get("source_model", "")) == source_model_key:
            return s
    for s in shapes:
        if int(s.get("batch_size", -1)) == batch and int(s.get("input_size", -1)) == input_size:
            return s
    if shapes:
        return shapes[0]
    raise ValueError("kernel {} 缺少 shapes 信息".format(kernel_item.get("node_name", "unknown")))


def _parse_nchw(shape: List[int]) -> Tuple[int, int, int, int]:
    vals = [int(x) for x in shape]
    if len(vals) >= 4:
        return vals[0], vals[1], vals[2], vals[3]
    if len(vals) == 3:
        return vals[0], vals[1], vals[2], 1
    if len(vals) == 2:
        return vals[0], vals[1], 1, 1
    if len(vals) == 1:
        return vals[0], 1, 1, 1
    return 1, 1, 1, 1


def _build_runtime_kernels(
    dag: Dict, kernel_json: Dict, source_model_key: str, batch: int, input_size: int
) -> List[Dict]:
    by_node = {k.get("node_name"): k for k in (kernel_json.get("kernels", []) or [])}
    rows = []
    for node_name in dag.get("topological_order", []):
        kernel_item = by_node.get(node_name)
        if not kernel_item:
            raise ValueError("DAG 节点在 kernel_json 中不存在: {}".format(node_name))
        shape_entry = _pick_shape_entry(kernel_item, source_model_key, batch, input_size)
        activation_shape = shape_entry.get("activation_input_shape", [])
        b, c, h, w = _parse_nchw(activation_shape)
        attrs = kernel_item.get("attributes", {}) or {}
        kernel_shape = attrs.get("kernel_shape", [1, 1]) or [1, 1]
        if len(kernel_shape) == 1:
            kernel_shape = [kernel_shape[0], 1]

        rows.append(
            {
                "node_name": node_name,
                "OpType": str(kernel_item.get("kernel_type", "Unknown")),
                "kernel_group": str(kernel_item.get("kernel_type", "Unknown")),
                "batch": float(b),
                "channel": float(c),
                "height": float(h),
                "width": float(w),
                "kernel_h": float(kernel_shape[0]),
                "kernel_w": float(kernel_shape[1]),
            }
        )
    return rows


def _encode_and_scale(feature_row: Dict, preprocess_meta: Dict, feature_cols: List[str]) -> List[float]:
    category_maps = preprocess_meta.get("category_maps", {}) or {}
    scaler = preprocess_meta.get("scaler", {}) or {}
    means = scaler.get("mean", {}) or {}
    stds = scaler.get("std", {}) or {}

    out = []
    for col in feature_cols:
        raw = feature_row.get(col)
        if col in category_maps:
            mapping = category_maps[col]
            raw_key = str(raw)
            if raw_key not in mapping:
                raise ValueError("特征 {} 出现未见类别: {}".format(col, raw_key))
            value = float(mapping[raw_key])
        else:
            value = float(raw)

        mean = float(means.get(col, 0.0))
        std = float(stds.get(col, 1.0))
        if abs(std) < 1e-12:
            std = 1.0
        out.append((value - mean) / std)
    return out


def predict_model_latency(
    model_path: str,
    config_path: str,
    dag_dir: str,
    kernel_record_dir: str,
    regressor_path: str,
    preprocess_meta_path: str,
    perf_json_path: str,
    sequential_stream0: bool = True,
) -> float:
    cfg = _load_yaml(config_path)
    model_name, batch, input_size, source_model_key = _parse_model_filename(model_path)
    dag_path = _ensure_dag_for_model(model_name, dag_dir=dag_dir, kernel_record_dir=kernel_record_dir)
    with open(dag_path, "r", encoding="utf-8") as f:
        dag = json.load(f)
    kernel_json = _load_kernel_json(model_name, kernel_record_dir=kernel_record_dir)

    runtime_kernels = _build_runtime_kernels(
        dag=dag,
        kernel_json=kernel_json,
        source_model_key=source_model_key,
        batch=batch,
        input_size=input_size,
    )

    with open(preprocess_meta_path, "r", encoding="utf-8") as f:
        preprocess_meta = json.load(f)

    feature_cols = preprocess_meta.get("feature_cols", [])
    if not feature_cols:
        raise ValueError("preprocess_meta 缺少 feature_cols: {}".format(preprocess_meta_path))

    dcgm = _get_realtime_dcgm(cfg)
    hwpk_map = _load_hwpk_map(perf_json_path)

    runtime_gpu = _get_gpu_name_fallback()
    if runtime_gpu not in hwpk_map:
        if len(hwpk_map) == 1:
            runtime_gpu = list(hwpk_map.keys())[0]
            logger.warning("当前 GPU 未命中性能映射，回退为唯一可用 GPU: %s", runtime_gpu)
        else:
            raise ValueError(
                "当前 GPU '{}' 在 performance_kernel_times.json 中无映射。可用: {}".format(
                    runtime_gpu, ", ".join(sorted(hwpk_map.keys()))
                )
            )
    hwpk = hwpk_map[runtime_gpu]

    regressor = load(regressor_path)

    node_latency_ms: Dict[str, float] = {}
    for row in runtime_kernels:
        feature_row = {}
        feature_row.update(dcgm)
        feature_row.update(hwpk)
        feature_row.update(row)
        feature_row["GPU Type"] = runtime_gpu

        x = _encode_and_scale(feature_row, preprocess_meta, feature_cols)
        pred_ms = float(regressor.predict([x])[0])
        node_latency_ms[row["node_name"]] = pred_ms

    total_ms = calculate_latency(dag, node_latency_ms, sequential_stream0=sequential_stream0)
    return float(total_ms)


def main():
    parser = argparse.ArgumentParser(description="Task7.1 predictor integration")
    parser.add_argument("--model", type=str, required=True, help="模型路径，如 model_zoo/models/resnet18_bs16_224x224.onnx")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "inference", "config.yaml"),
        help="inference 配置文件路径",
    )
    parser.add_argument(
        "--dag-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "graph_model", "model_DAG"),
        help="DAG 文件目录",
    )
    parser.add_argument(
        "--kernel-record-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "ort_analysis", "ort_kernel_record"),
        help="kernel record JSON 目录",
    )
    parser.add_argument(
        "--regressor",
        type=str,
        default=os.path.join(PROJECT_ROOT, "training", "models", "randomforest.joblib"),
        help="训练好的回归模型路径",
    )
    parser.add_argument(
        "--preprocess-meta",
        type=str,
        default=os.path.join(PROJECT_ROOT, "training", "models", "preprocess_meta.json"),
        help="预处理元信息路径",
    )
    parser.add_argument(
        "--perf-json",
        type=str,
        default=os.path.join(PROJECT_ROOT, "training", "performance_kernel_times.json"),
        help="硬件性能特征 JSON 路径",
    )
    parser.add_argument(
        "--non-sequential",
        action="store_true",
        help="使用 DAG 最长路径（默认顺序累加）",
    )
    args = parser.parse_args()

    model_path = _resolve_path(PROJECT_ROOT, args.model)
    config_path = _resolve_path(PROJECT_ROOT, args.config)
    dag_dir = _resolve_path(PROJECT_ROOT, args.dag_dir)
    kernel_record_dir = _resolve_path(PROJECT_ROOT, args.kernel_record_dir)
    regressor_path = _resolve_path(PROJECT_ROOT, args.regressor)
    preprocess_meta_path = _resolve_path(PROJECT_ROOT, args.preprocess_meta)
    perf_json_path = _resolve_path(PROJECT_ROOT, args.perf_json)

    if not os.path.isfile(model_path):
        raise ValueError("模型文件不存在: {}".format(model_path))
    if not os.path.isfile(config_path):
        raise ValueError("配置文件不存在: {}".format(config_path))
    if not os.path.isdir(kernel_record_dir):
        raise ValueError("kernel record 目录不存在: {}".format(kernel_record_dir))
    if not os.path.isfile(regressor_path):
        raise ValueError("回归模型不存在: {}".format(regressor_path))
    if not os.path.isfile(preprocess_meta_path):
        raise ValueError("预处理元信息不存在: {}".format(preprocess_meta_path))
    if not os.path.isfile(perf_json_path):
        raise ValueError("性能特征文件不存在: {}".format(perf_json_path))
    os.makedirs(dag_dir, exist_ok=True)

    total_ms = predict_model_latency(
        model_path=model_path,
        config_path=config_path,
        dag_dir=dag_dir,
        kernel_record_dir=kernel_record_dir,
        regressor_path=regressor_path,
        preprocess_meta_path=preprocess_meta_path,
        perf_json_path=perf_json_path,
        sequential_stream0=not args.non_sequential,
    )
    print("Predicted_Total_Latency_ms: {:.6f}".format(total_ms))


if __name__ == "__main__":
    main()
