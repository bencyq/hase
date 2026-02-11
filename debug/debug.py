# -*- coding: utf-8 -*-
"""
Task 8.1 Debug:
分解模型到内核，按执行顺序依次运行并输出各内核时间。
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import onnx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from benchmark.ort_kernel_runner import run_kernel  # noqa: E402
from graph_model.dag_critical_path import build_dag_from_kernel_json  # noqa: E402
from kernel_model.kernel_builder import make_kernel_filename, build_kernel_model  # noqa: E402
from ort_analysis.fusion_detector import process_model_zoo  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("debug_task")

MODEL_PATTERN = re.compile(
    r"^(?P<model>[a-zA-Z0-9_]+)_bs(?P<batch>\d+)_(?P<input>\d+)x(?P=input)$"
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _parse_model_filename(model_path: str) -> Tuple[str, int, int, str]:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    m = MODEL_PATTERN.match(stem)
    if not m:
        raise ValueError(
            "模型文件名需满足 <model>_bs<batch>_<size>x<size>.onnx，当前: {}".format(stem)
        )
    return m.group("model"), int(m.group("batch")), int(m.group("input")), stem


def _pick_shape_entry(kernel_item: Dict, source_model_key: str, batch: int, input_size: int) -> Dict:
    shapes = kernel_item.get("shapes", []) or []
    for s in shapes:
        if str(s.get("source_model", "")) == source_model_key:
            return s
    for s in shapes:
        if int(s.get("batch_size", -1)) == batch and int(s.get("input_size", -1)) == input_size:
            return s
    if shapes:
        return shapes[0]
    raise ValueError("kernel {} 缺少可用 shapes".format(kernel_item.get("node_name", "unknown")))


def _ensure_kernel_record(model_path: str, model_name: str, record_dir: str) -> str:
    record_json = os.path.join(record_dir, "{}.json".format(model_name))
    if os.path.isfile(record_json):
        return record_json

    model_dir = os.path.dirname(model_path)
    out = process_model_zoo(model_dir=model_dir, model_prefix=model_name)
    if not out or not os.path.isfile(out):
        raise RuntimeError("生成 kernel record 失败: {}".format(model_name))
    return out


def _ensure_needed_kernel_onnx(
    record: Dict,
    topo: List[str],
    source_model_key: str,
    batch: int,
    input_size: int,
    kernel_onnx_dir: str,
) -> None:
    by_node = {k.get("node_name"): k for k in record.get("kernels", [])}
    for node_name in topo:
        kernel_item = by_node.get(node_name)
        if not kernel_item:
            continue
        shape_info = _pick_shape_entry(kernel_item, source_model_key, batch, input_size)
        kernel_id = make_kernel_filename(kernel_item, shape_info)
        kernel_path = os.path.join(kernel_onnx_dir, "{}.onnx".format(kernel_id))
        if os.path.isfile(kernel_path):
            continue
        model = build_kernel_model(kernel_item, shape_info)
        onnx.save(model, kernel_path)


def _get_input_shape_str(onnx_path: str) -> str:
    model = onnx.load(onnx_path)
    if not model.graph.input:
        return ""
    dims = []
    for d in model.graph.input[0].type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(str(int(d.dim_value)))
        else:
            dims.append("0")
    return "x".join(dims)


def run_debug_for_model(
    model_path: str,
    record_dir: str,
    kernel_onnx_dir: str,
    warmup: int,
    loops: int,
    output_json: str = "",
) -> List[Dict]:
    model_name, batch, input_size, source_model_key = _parse_model_filename(model_path)

    record_json = _ensure_kernel_record(model_path, model_name, record_dir)
    with open(record_json, "r", encoding="utf-8") as f:
        record = json.load(f)

    dag = build_dag_from_kernel_json(record_json)
    topo = dag.get("topological_order", [])
    _ensure_needed_kernel_onnx(
        record=record,
        topo=topo,
        source_model_key=source_model_key,
        batch=batch,
        input_size=input_size,
        kernel_onnx_dir=kernel_onnx_dir,
    )
    by_node = {k.get("node_name"): k for k in record.get("kernels", [])}

    results = []
    print("Model: {}".format(model_path))
    print("Kernel Count: {}".format(len(topo)))
    print("Execution Order:")

    total_ms = 0.0
    for idx, node_name in enumerate(topo, start=1):
        k = by_node.get(node_name)
        if not k:
            raise ValueError("节点 {} 在 kernel record 中不存在".format(node_name))
        shape_info = _pick_shape_entry(k, source_model_key, batch, input_size)
        kernel_id = make_kernel_filename(k, shape_info)
        kernel_path = os.path.join(kernel_onnx_dir, "{}.onnx".format(kernel_id))
        if not os.path.isfile(kernel_path):
            raise ValueError("缺少内核模型文件: {}".format(kernel_path))

        avg_ms, _, _ = run_kernel(kernel_path, warmup=warmup, loops=loops, use_warmup=True)
        total_ms += float(avg_ms)

        item = {
            "index": idx,
            "node_name": node_name,
            "kernel_type": k.get("kernel_type"),
            "kernel_id": kernel_id,
            "kernel_onnx": kernel_path,
            "input_shape": _get_input_shape_str(kernel_path),
            "latency_ms": float(avg_ms),
        }
        results.append(item)

        print(
            "[{:02d}] {} | {} | {:.6f} ms".format(
                idx, node_name, item["kernel_type"], item["latency_ms"]
            )
        )

    model_avg_ms, _, _ = run_kernel(model_path, warmup=warmup, loops=loops, use_warmup=True)
    model_avg_ms = float(model_avg_ms)
    diff_ms = total_ms - model_avg_ms
    ratio = (total_ms / model_avg_ms) if model_avg_ms > 1e-9 else 0.0

    print("Total Kernel Latency(ms): {:.6f}".format(total_ms))
    print("Direct Model Latency(ms): {:.6f}".format(model_avg_ms))
    print("KernelSum - Model(ms): {:.6f}".format(diff_ms))
    print("KernelSum / Model: {:.6f}".format(ratio))

    if output_json:
        payload = {
            "model_path": model_path,
            "record_json": record_json,
            "kernel_onnx_dir": kernel_onnx_dir,
            "warmup": warmup,
            "loops": loops,
            "total_kernel_latency_ms": total_ms,
            "direct_model_latency_ms": model_avg_ms,
            "kernel_minus_model_ms": diff_ms,
            "kernel_over_model_ratio": ratio,
            "kernels": results,
        }
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("Saved: {}".format(output_json))

    return results


def main():
    parser = argparse.ArgumentParser(description="Task8.1 debug: split and run kernels in order")
    parser.add_argument("--model", type=str, required=True, help="模型路径（如 model_zoo/models/resnet18_bs16_224x224.onnx）")
    parser.add_argument(
        "--record-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "ort_analysis", "ort_kernel_record"),
        help="kernel record 目录",
    )
    parser.add_argument(
        "--kernel-onnx-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "kernel_model", "kernel_onnx"),
        help="单 kernel onnx 目录",
    )
    parser.add_argument("--warmup", type=int, default=10, help="每个内核预热次数")
    parser.add_argument("--loops", type=int, default=20, help="每个内核循环次数")
    parser.add_argument("--output-json", type=str, default="", help="可选：保存详细输出 json")
    args = parser.parse_args()

    model_path = _resolve_path(args.model)
    record_dir = _resolve_path(args.record_dir)
    kernel_onnx_dir = _resolve_path(args.kernel_onnx_dir)
    output_json = _resolve_path(args.output_json) if args.output_json else ""

    if not os.path.isfile(model_path):
        raise ValueError("模型文件不存在: {}".format(model_path))
    os.makedirs(record_dir, exist_ok=True)
    os.makedirs(kernel_onnx_dir, exist_ok=True)

    run_debug_for_model(
        model_path=model_path,
        record_dir=record_dir,
        kernel_onnx_dir=kernel_onnx_dir,
        warmup=args.warmup,
        loops=args.loops,
        output_json=output_json,
    )


if __name__ == "__main__":
    main()
