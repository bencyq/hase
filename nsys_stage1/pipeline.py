# -*- coding: utf-8 -*-
"""
NSYS Stage 1 单模型闭环流水线。
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import onnx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kernel_model.kernel_builder import build_kernel_model, validate_kernel
from nsys_stage1.nsys_sqlite_parser import export_sqlite, parse_pure_gpu_time, profile_with_nsys
from ort_analysis.fusion_detector import detect_kernel_type
from ort_analysis.ort_graph_parser import (
    _build_initial_shape_map,
    _get_attr,
    _infer_node_output_shape,
    get_optimized_model,
)
from utils.logger import get_logger

logger = get_logger("nsys_stage1.pipeline")

SUPPORTED_KERNEL_TYPES = {
    "Conv",
    "Conv_Relu",
    "Conv_Add_Relu",
    "MaxPool",
    "GlobalAveragePool",
    "Flatten",
    "Gemm",
}
ATTR_KEYS = {
    "Conv": ["kernel_shape", "strides", "pads", "dilations", "group"],
    "FusedConv": ["kernel_shape", "strides", "pads", "dilations", "group", "activation"],
    "MaxPool": ["kernel_shape", "strides", "pads", "ceil_mode"],
    "Gemm": ["transA", "transB"],
    "Flatten": ["axis"],
}


def _parse_model_name(model_path):
    return os.path.splitext(os.path.basename(model_path))[0]


def _normalize_value(value):
    if isinstance(value, tuple):
        return [_normalize_value(v) for v in value]
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(value[k]) for k in sorted(value)}
    if value is None:
        return None
    return value


def _extract_attributes(node):
    attrs = {}
    for key in ATTR_KEYS.get(node.op_type, []):
        value = _get_attr(node, key)
        if value is not None:
            attrs[key] = _normalize_value(value)
    return attrs


def _extract_node_shapes(node, shape_map, init_names):
    act_shape = None
    weight_shape = None
    bias_shape = None
    residual_shape = None

    for idx, inp_name in enumerate(node.input):
        shape = _normalize_value(shape_map.get(inp_name))
        if inp_name in init_names:
            if shape and len(shape) >= 2 and weight_shape is None:
                weight_shape = shape
            elif shape and len(shape) == 1 and bias_shape is None:
                bias_shape = shape
        else:
            if act_shape is None:
                act_shape = shape
            elif node.op_type == "FusedConv" and idx >= 3:
                residual_shape = shape

    output_shape = None
    if node.output:
        output_shape = _normalize_value(shape_map.get(node.output[0]))

    return act_shape, weight_shape, bias_shape, residual_shape, output_shape


def _signature_payload(instance):
    return {
        "kernel_type": instance["kernel_type"],
        "attributes": instance["attributes"],
        "activation_input_shape": instance["activation_input_shape"],
        "weight_shape": instance["weight_shape"],
        "bias_shape": instance["bias_shape"],
        "residual_shape": instance["residual_shape"],
        "output_shape": instance["output_shape"],
    }


def _signature_id(payload):
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def analyze_model(model_path):
    model_name = _parse_model_name(model_path)
    optimized_model = get_optimized_model(model_path)
    shape_map = _build_initial_shape_map(optimized_model)
    for node in optimized_model.graph.node:
        _infer_node_output_shape(node, shape_map)

    init_names = {init.name for init in optimized_model.graph.initializer}
    instances = []
    signatures = {}

    for node in optimized_model.graph.node:
        kernel_type = detect_kernel_type(node)
        if kernel_type not in SUPPORTED_KERNEL_TYPES:
            raise RuntimeError("检测到未支持的 kernel_type: {} ({})".format(kernel_type, node.name))

        attrs = _extract_attributes(node)
        act_shape, weight_shape, bias_shape, residual_shape, output_shape = _extract_node_shapes(
            node, shape_map, init_names
        )
        instance = {
            "source_model": model_name,
            "node_name": node.name,
            "kernel_type": kernel_type,
            "activation_input_shape": act_shape,
            "weight_shape": weight_shape,
            "bias_shape": bias_shape,
            "residual_shape": residual_shape,
            "output_shape": output_shape,
            "attributes": attrs,
        }
        signature_payload = _signature_payload(instance)
        signature_id = _signature_id(signature_payload)
        instance["signature_id"] = signature_id
        instances.append(instance)

        if signature_id not in signatures:
            signature_record = dict(signature_payload)
            signature_record["signature_id"] = signature_id
            signature_record["instance_node_names"] = [node.name]
            signatures[signature_id] = signature_record
        else:
            signatures[signature_id]["instance_node_names"].append(node.name)

    return instances, list(signatures.values())


def _signature_to_builder_inputs(signature):
    kernel_info = {
        "kernel_type": signature["kernel_type"],
        "attributes": signature["attributes"],
    }
    shape_info = {
        "activation_input_shape": signature["activation_input_shape"],
        "weight_shape": signature["weight_shape"],
        "bias_shape": signature["bias_shape"],
        "residual_shape": signature["residual_shape"],
        "output_shape": signature["output_shape"],
    }
    return kernel_info, shape_info


def build_signature_models(signatures, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    built = []
    for signature in signatures:
        kernel_info, shape_info = _signature_to_builder_inputs(signature)
        model = build_kernel_model(kernel_info, shape_info)
        onnx_path = os.path.join(output_dir, "{}.onnx".format(signature["signature_id"]))
        onnx.save(model, onnx_path)
        validate_kernel(onnx_path)
        built.append(
            {
                "signature_id": signature["signature_id"],
                "kernel_type": signature["kernel_type"],
                "onnx_path": onnx_path,
            }
        )
    return built


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def collect_signature_times(
    signature_models,
    result_dir,
    runner_path,
    python_bin,
    loops,
    warmup,
    range_name,
    nsys_bin,
):
    raw_dir = os.path.join(result_dir, "nsys_raw")
    os.makedirs(raw_dir, exist_ok=True)
    records = []
    for item in signature_models:
        prefix = os.path.join(raw_dir, item["signature_id"])
        report_path = profile_with_nsys(
            python_bin=python_bin,
            runner_path=runner_path,
            model_path=item["onnx_path"],
            output_prefix=prefix,
            range_name=range_name,
            warmup=warmup,
            loops=loops,
            nsys_bin=nsys_bin,
        )
        sqlite_path = export_sqlite(report_path, prefix, nsys_bin=nsys_bin)
        parsed = parse_pure_gpu_time(sqlite_path, range_name, loops)
        record = {
            "signature_id": item["signature_id"],
            "kernel_type": item["kernel_type"],
            "loops": loops,
            "nsys_rep_path": report_path,
            "sqlite_path": sqlite_path,
        }
        record.update(parsed)
        records.append(record)
    return records


def aggregate_kernel_types(model_name, instances, kernel_times):
    time_by_signature = {item["signature_id"]: item for item in kernel_times}
    summary = {}
    aggregate_total_ms = 0.0

    for instance in instances:
        time_record = time_by_signature[instance["signature_id"]]
        pure_gpu_ms = time_record["pure_gpu_ms"]
        aggregate_total_ms += pure_gpu_ms
        kernel_type = instance["kernel_type"]
        if kernel_type not in summary:
            summary[kernel_type] = {
                "model_name": model_name,
                "kernel_type": kernel_type,
                "instance_count": 0,
                "signature_count": 0,
                "total_pure_gpu_ms": 0.0,
                "avg_pure_gpu_ms": 0.0,
                "signature_breakdown": {},
            }

        entry = summary[kernel_type]
        entry["instance_count"] += 1
        entry["total_pure_gpu_ms"] += pure_gpu_ms
        sig = instance["signature_id"]
        if sig not in entry["signature_breakdown"]:
            entry["signature_breakdown"][sig] = {
                "signature_id": sig,
                "instance_count": 0,
                "pure_gpu_ms": pure_gpu_ms,
                "node_names": [],
            }
        entry["signature_breakdown"][sig]["instance_count"] += 1
        entry["signature_breakdown"][sig]["node_names"].append(instance["node_name"])

    result = []
    for kernel_type in sorted(summary):
        entry = summary[kernel_type]
        entry["signature_count"] = len(entry["signature_breakdown"])
        entry["avg_pure_gpu_ms"] = (
            entry["total_pure_gpu_ms"] / entry["instance_count"] if entry["instance_count"] else 0.0
        )
        entry["signature_breakdown"] = list(entry["signature_breakdown"].values())
        result.append(entry)

    return result, aggregate_total_ms


def profile_full_model(
    model_path,
    result_dir,
    runner_path,
    python_bin,
    loops,
    warmup,
    range_name,
    nsys_bin,
):
    prefix = os.path.join(result_dir, "nsys_raw", "full_model")
    report_path = profile_with_nsys(
        python_bin=python_bin,
        runner_path=runner_path,
        model_path=model_path,
        output_prefix=prefix,
        range_name=range_name,
        warmup=warmup,
        loops=loops,
        nsys_bin=nsys_bin,
    )
    sqlite_path = export_sqlite(report_path, prefix, nsys_bin=nsys_bin)
    parsed = parse_pure_gpu_time(sqlite_path, range_name, loops)
    record = {
        "loops": loops,
        "nsys_rep_path": report_path,
        "sqlite_path": sqlite_path,
    }
    record.update(parsed)
    return record


def validate_result(aggregate_total_ms, full_model_record, threshold):
    full_model_gpu_ms = full_model_record["pure_gpu_ms"]
    if full_model_gpu_ms <= 0:
        raise RuntimeError("完整模型 pure_gpu_ms 非法: {}".format(full_model_gpu_ms))
    diff_ratio = abs(aggregate_total_ms - full_model_gpu_ms) / full_model_gpu_ms
    return {
        "aggregated_model_gpu_ms": aggregate_total_ms,
        "full_model_gpu_ms": full_model_gpu_ms,
        "diff_ratio": diff_ratio,
        "threshold": threshold,
        "passed": diff_ratio <= threshold,
    }


def run_pipeline(args):
    model_name = _parse_model_name(args.model)
    result_dir = os.path.join(PROJECT_ROOT, "benchmark", "results", "nsys_stage1", model_name)
    kernel_dir = os.path.join(PROJECT_ROOT, "kernel_model", "kernel_onnx", "nsys_stage1", model_name)

    if args.clean and os.path.isdir(result_dir):
        shutil.rmtree(result_dir)
    if args.clean and os.path.isdir(kernel_dir):
        shutil.rmtree(kernel_dir)

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(kernel_dir, exist_ok=True)

    instances, signatures = analyze_model(args.model)
    _write_json(os.path.join(result_dir, "instances.json"), instances)
    _write_json(os.path.join(result_dir, "signatures.json"), signatures)

    signature_models = build_signature_models(signatures, kernel_dir)
    _write_json(os.path.join(result_dir, "signature_models.json"), signature_models)

    runner_path = os.path.join(PROJECT_ROOT, "nsys_stage1", "run_with_nvtx.py")
    kernel_times = collect_signature_times(
        signature_models=signature_models,
        result_dir=result_dir,
        runner_path=runner_path,
        python_bin=args.python_bin,
        loops=args.loops,
        warmup=args.warmup,
        range_name=args.range_name,
        nsys_bin=args.nsys_bin,
    )
    _write_json(os.path.join(result_dir, "kernel_times.json"), kernel_times)

    summary, aggregate_total_ms = aggregate_kernel_types(model_name, instances, kernel_times)
    _write_json(os.path.join(result_dir, "kernel_type_summary.json"), summary)

    full_model_record = profile_full_model(
        model_path=args.model,
        result_dir=result_dir,
        runner_path=runner_path,
        python_bin=args.python_bin,
        loops=args.loops,
        warmup=args.warmup,
        range_name=args.range_name,
        nsys_bin=args.nsys_bin,
    )
    _write_json(os.path.join(result_dir, "full_model_time.json"), full_model_record)

    validation = validate_result(aggregate_total_ms, full_model_record, args.threshold)
    _write_json(os.path.join(result_dir, "validation.json"), validation)
    logger.info("stage1 完成: %s", result_dir)
    logger.info("校验结果: %s", validation)
    if not validation["passed"]:
        raise RuntimeError("阶段 1 校验失败: diff_ratio={:.6f}".format(validation["diff_ratio"]))


def main():
    parser = argparse.ArgumentParser(description="NSYS Stage 1 single-model pipeline.")
    parser.add_argument("--model", required=True, help="目标模型路径")
    parser.add_argument("--loops", type=int, default=50, help="正式循环次数")
    parser.add_argument("--warmup", type=int, default=20, help="warmup 次数")
    parser.add_argument("--range-name", default="measure", help="NVTX range 名称")
    parser.add_argument("--threshold", type=float, default=0.15, help="校验阈值")
    parser.add_argument("--python-bin", default=sys.executable, help="采集时使用的 Python")
    parser.add_argument("--nsys-bin", default=os.environ.get("NSYS_BIN", "nsys"), help="nsys 可执行文件")
    parser.add_argument("--clean", action="store_true", help="运行前清空旧结果")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
