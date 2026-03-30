# -*- coding: utf-8 -*-
"""
ORT 完整模型原位插桩实验（阶段 A 基线）。

当前脚本完成以下工作：
1. 生成 ORT 优化图 metadata，并映射到 kernel_type / signature_id
2. 在完整模型上执行 warmup + loops，导出 ORT raw profile
3. 解析 Node 事件链：fence_before -> kernel_time -> fence_after
4. 输出阶段 A baseline、gap 分布、聚合摘要与 validation

说明：
- 这是 docs/ort_instrumented_fullmodel_experiment.md 定义的阶段 A 落地版本
- 若当前 ORT build 没有 device Kernel 事件，本脚本仍可完成阶段 A
- 若阶段 A 校验失败，本脚本会额外输出 gap 分布定位结果
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ort_analysis.fusion_detector import detect_kernel_type
from ort_analysis.ort_graph_parser import (
    _build_initial_shape_map,
    _get_attr,
    _infer_node_output_shape,
    get_optimized_model,
)
from utils.logger import get_logger

logger = get_logger("ort_fullmodel_experiment")

ATTR_KEYS = {
    "Conv": ["kernel_shape", "strides", "pads", "dilations", "group"],
    "FusedConv": ["kernel_shape", "strides", "pads", "dilations", "group", "activation"],
    "MaxPool": ["kernel_shape", "strides", "pads", "ceil_mode"],
    "AveragePool": ["kernel_shape", "strides", "pads", "ceil_mode"],
    "Gemm": ["transA", "transB"],
    "Flatten": ["axis"],
}
EVENT_SUFFIXES = {
    "_fence_before": "fence_before",
    "_kernel_time": "kernel_time",
    "_fence_after": "fence_after",
}
DEFAULT_PROVIDER_OPTIONS = {
    "cudnn_conv_algo_search": "HEURISTIC",
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



def build_optimized_node_metadata(model_path):
    model_name = _parse_model_name(model_path)
    optimized_model = get_optimized_model(model_path)
    shape_map = _build_initial_shape_map(optimized_model)
    for node in optimized_model.graph.node:
        _infer_node_output_shape(node, shape_map)

    init_names = {init.name for init in optimized_model.graph.initializer}
    records = []

    for node_index, node in enumerate(optimized_model.graph.node):
        kernel_type = detect_kernel_type(node)
        attrs = _extract_attributes(node)
        act_shape, weight_shape, bias_shape, residual_shape, output_shape = _extract_node_shapes(
            node, shape_map, init_names
        )
        record = {
            "source_model": model_name,
            "node_name": node.name,
            "node_index": node_index,
            "op_type": node.op_type,
            "kernel_type": kernel_type,
            "attributes": attrs,
            "activation_input_shape": act_shape,
            "weight_shape": weight_shape,
            "bias_shape": bias_shape,
            "residual_shape": residual_shape,
            "output_shape": output_shape,
        }
        record["signature_id"] = _signature_id(_signature_payload(record))
        records.append(record)

    return records



def _to_numpy_dtype(ort_type):
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



def _normalize_shape(shape):
    dims = []
    for dim in shape:
        if dim in (None, 0, "None"):
            dims.append(1)
        else:
            dims.append(int(dim))
    return dims



def _prepare_inputs(session):
    feed = {}
    for inp in session.get_inputs():
        shape = _normalize_shape(inp.shape)
        dtype = _to_numpy_dtype(inp.type)
        if np.issubdtype(dtype, np.bool_):
            data = np.random.randint(0, 2, size=shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            data = np.random.randint(0, 8, size=shape).astype(dtype)
        else:
            data = np.random.randn(*shape).astype(dtype)
        feed[inp.name] = data
    return feed



def _cuda_synchronize():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass



def _build_provider_list(device_id, provider_options):
    cuda_options = dict(provider_options)
    cuda_options["device_id"] = str(device_id)
    return [
        ("CUDAExecutionProvider", cuda_options),
        "CPUExecutionProvider",
    ]



def _build_session_options(enable_profiling=False, profile_file_prefix=""):
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if enable_profiling:
        options.enable_profiling = True
        if profile_file_prefix:
            options.profile_file_prefix = profile_file_prefix
    return options



def measure_model_e2e_ms(model_path, providers, warmup, loops):
    session = ort.InferenceSession(
        model_path,
        sess_options=_build_session_options(enable_profiling=False),
        providers=providers,
    )
    feed = _prepare_inputs(session)

    for _ in range(max(warmup, 0)):
        session.run(None, feed)

    _cuda_synchronize()
    t0 = time.perf_counter()
    for _ in range(max(loops, 1)):
        session.run(None, feed)
    _cuda_synchronize()
    t1 = time.perf_counter()

    return ((t1 - t0) / float(max(loops, 1))) * 1000.0



def collect_raw_profile(model_path, output_dir, providers, warmup, loops):
    profile_prefix = os.path.join(output_dir, "raw_profile")
    raw_profile_path = os.path.join(output_dir, "raw_profile.json")
    if os.path.exists(raw_profile_path):
        os.remove(raw_profile_path)

    session = ort.InferenceSession(
        model_path,
        sess_options=_build_session_options(
            enable_profiling=True,
            profile_file_prefix=profile_prefix,
        ),
        providers=providers,
    )
    feed = _prepare_inputs(session)

    for _ in range(max(warmup, 0)):
        session.run(None, feed)

    for _ in range(max(loops, 1)):
        session.run(None, feed)

    generated_path = session.end_profiling()
    time.sleep(0.05)
    shutil.move(generated_path, raw_profile_path)
    return raw_profile_path



def _load_profile_events(raw_profile_path):
    with open(raw_profile_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    if not isinstance(events, list):
        raise RuntimeError("ORT raw profile 格式非法: {}".format(raw_profile_path))
    return events



def _event_ts_us(event):
    return float(event.get("ts", 0.0) or 0.0)



def _event_dur_us(event):
    return float(event.get("dur", 0.0) or 0.0)



def _event_end_us(event):
    return _event_ts_us(event) + _event_dur_us(event)



def _split_node_event_name(name):
    for suffix, kind in EVENT_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], kind
    return name, None



def _profile_overview(events):
    cat_counter = Counter()
    session_name_counter = Counter()
    node_event_suffix_counter = Counter()

    for event in events:
        cat = event.get("cat", "")
        cat_counter[cat] += 1
        if cat == "Session":
            session_name_counter[event.get("name", "")] += 1
        elif cat == "Node":
            _, kind = _split_node_event_name(str(event.get("name", "")))
            if kind:
                node_event_suffix_counter[kind] += 1

    return {
        "category_counts": dict(cat_counter),
        "session_event_counts": dict(session_name_counter),
        "node_suffix_counts": dict(node_event_suffix_counter),
        "has_device_kernel_events": bool(cat_counter.get("Kernel", 0)),
        "has_gpu_copy_events": bool(cat_counter.get("Memcpy", 0) or cat_counter.get("Memset", 0)),
    }



def _collect_session_windows(events, event_name, loops, required=True):
    matched = []
    for event in events:
        if event.get("cat") != "Session":
            continue
        if event.get("name") != event_name:
            continue
        matched.append(event)

    if len(matched) < loops:
        if required:
            raise RuntimeError(
                "profile 中 {} 事件不足: got={} loops={}".format(event_name, len(matched), loops)
            )
        return [], len(matched)

    selected = matched[-loops:]
    windows = []
    for iteration_id, event in enumerate(selected):
        windows.append(
            {
                "iteration_id": iteration_id,
                "iteration_global_index": len(matched) - loops + iteration_id,
                "event_name": event_name,
                "session_start_us": _event_ts_us(event),
                "session_end_us": _event_end_us(event),
                "session_ms": _event_dur_us(event) / 1000.0,
            }
        )
    return windows, len(matched)



def _parse_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None



def _attach_event_args(record, event):
    args = event.get("args") or {}
    if record.get("provider") is None and args.get("provider") is not None:
        record["provider"] = args.get("provider")
    if record.get("node_index") is None:
        node_index = _parse_int(args.get("node_index"))
        if node_index is not None:
            record["node_index"] = node_index
    if record.get("op_name") is None and args.get("op_name") is not None:
        record["op_name"] = args.get("op_name")



def _finalize_exec_record(record):
    record["fence_before_us"] = record.get("fence_before_us")
    record["kernel_time_us"] = record.get("kernel_time_us")
    record["fence_after_us"] = record.get("fence_after_us")
    record["has_complete_chain"] = (
        record.get("fence_before_us") is not None
        and record.get("kernel_time_us") is not None
        and record.get("fence_after_us") is not None
    )
    if record["has_complete_chain"]:
        record["kernel_time_ms"] = record["kernel_time_us"] / 1000.0
        record["gpu_kernel_ms"] = record["kernel_time_us"] / 1000.0
        record["gpu_kernel_source"] = "ort_node_kernel_time"
        record["gpu_copy_ms"] = 0.0
        record["node_span_ms"] = max(
            0.0,
            (record["fence_after_us"] - record["fence_before_us"]) / 1000.0,
        )
    else:
        record["kernel_time_ms"] = None
        record["gpu_kernel_ms"] = None
        record["gpu_kernel_source"] = "missing_chain"
        record["gpu_copy_ms"] = 0.0
        record["node_span_ms"] = None
    return record



def _parse_iteration_node_execs(events, iteration_info):
    iteration_start = iteration_info["model_run_start_us"]
    iteration_end = iteration_info["model_run_end_us"]
    iteration_id = iteration_info["iteration_id"]

    selected_events = []
    for event in events:
        if event.get("cat") != "Node":
            continue
        event_ts = _event_ts_us(event)
        if event_ts < iteration_start or event_ts > iteration_end:
            continue
        selected_events.append(event)

    selected_events.sort(key=lambda item: (_event_ts_us(item), item.get("name", "")))
    open_records = {}
    ordered = []

    for event in selected_events:
        node_name, kind = _split_node_event_name(str(event.get("name", "")))
        if kind is None:
            continue

        if kind == "fence_before":
            if node_name in open_records:
                ordered.append(_finalize_exec_record(open_records.pop(node_name)))
            open_records[node_name] = {
                "iteration_id": iteration_id,
                "iteration_global_index": iteration_info["iteration_global_index"],
                "node_name": node_name,
                "node_index": None,
                "provider": None,
                "op_name": None,
                "dispatch_gap_ms": 0.0,
                "fence_before_us": _event_ts_us(event),
                "kernel_time_us": None,
                "fence_after_us": None,
                "model_run_ms": iteration_info["model_run_ms"],
                "sequential_executor_ms": iteration_info.get("sequential_executor_ms"),
            }
            _attach_event_args(open_records[node_name], event)
            continue

        record = open_records.get(node_name)
        if record is None:
            record = {
                "iteration_id": iteration_id,
                "iteration_global_index": iteration_info["iteration_global_index"],
                "node_name": node_name,
                "node_index": None,
                "provider": None,
                "op_name": None,
                "dispatch_gap_ms": 0.0,
                "fence_before_us": None,
                "kernel_time_us": None,
                "fence_after_us": None,
                "model_run_ms": iteration_info["model_run_ms"],
                "sequential_executor_ms": iteration_info.get("sequential_executor_ms"),
            }
            open_records[node_name] = record

        if kind == "kernel_time":
            record["kernel_time_us"] = _event_dur_us(event)
        elif kind == "fence_after":
            record["fence_after_us"] = _event_ts_us(event)

        _attach_event_args(record, event)

        if kind == "fence_after":
            ordered.append(_finalize_exec_record(open_records.pop(node_name)))

    for node_name in sorted(open_records):
        ordered.append(_finalize_exec_record(open_records[node_name]))

    ordered.sort(
        key=lambda item: (
            item["fence_before_us"] if item["fence_before_us"] is not None else float("inf"),
            item["node_index"] if item["node_index"] is not None else float("inf"),
            item["node_name"],
        )
    )

    prev_fence_after_us = None
    for record in ordered:
        if prev_fence_after_us is None or record.get("fence_before_us") is None:
            record["dispatch_gap_ms"] = 0.0
        else:
            record["dispatch_gap_ms"] = max(
                0.0,
                (record["fence_before_us"] - prev_fence_after_us) / 1000.0,
            )
        if record.get("fence_after_us") is not None:
            prev_fence_after_us = record["fence_after_us"]

    return ordered



def build_iteration_summary(node_execs, iteration_windows):
    grouped = defaultdict(list)
    for item in node_execs:
        grouped[item["iteration_id"]].append(item)

    summaries = []
    for window in iteration_windows:
        iteration_id = window["iteration_id"]
        records = sorted(
            grouped.get(iteration_id, []),
            key=lambda item: item.get("node_index") if item.get("node_index") is not None else 10 ** 9,
        )
        kernel_sum_ms = sum(float(item.get("kernel_time_ms") or 0.0) for item in records)
        node_span_sum_ms = sum(float(item.get("node_span_ms") or 0.0) for item in records)
        dispatch_gap_sum_ms = sum(float(item.get("dispatch_gap_ms") or 0.0) for item in records)

        first_node_start_us = None
        last_node_end_us = None
        if records:
            starts = [item["fence_before_us"] for item in records if item.get("fence_before_us") is not None]
            ends = [item["fence_after_us"] for item in records if item.get("fence_after_us") is not None]
            if starts:
                first_node_start_us = min(starts)
            if ends:
                last_node_end_us = max(ends)

        model_run_head_gap_ms = None
        model_run_tail_gap_ms = None
        sequential_head_gap_ms = None
        sequential_tail_gap_ms = None
        if first_node_start_us is not None:
            model_run_head_gap_ms = max(
                0.0, (first_node_start_us - window["model_run_start_us"]) / 1000.0
            )
        if last_node_end_us is not None:
            model_run_tail_gap_ms = max(
                0.0, (window["model_run_end_us"] - last_node_end_us) / 1000.0
            )
        if first_node_start_us is not None and window.get("sequential_executor_start_us") is not None:
            sequential_head_gap_ms = max(
                0.0,
                (first_node_start_us - window["sequential_executor_start_us"]) / 1000.0,
            )
        if last_node_end_us is not None and window.get("sequential_executor_end_us") is not None:
            sequential_tail_gap_ms = max(
                0.0,
                (window["sequential_executor_end_us"] - last_node_end_us) / 1000.0,
            )

        sequential_executor_ms = window.get("sequential_executor_ms")
        run_boundary_overhead_ms = None
        executor_unattributed_ms = None
        if sequential_executor_ms is not None:
            run_boundary_overhead_ms = max(0.0, window["model_run_ms"] - sequential_executor_ms)
            executor_unattributed_ms = max(
                0.0,
                sequential_executor_ms - node_span_sum_ms - dispatch_gap_sum_ms,
            )

        summaries.append(
            {
                "iteration_id": iteration_id,
                "iteration_global_index": window["iteration_global_index"],
                "node_exec_count": len(records),
                "model_run_ms": window["model_run_ms"],
                "sequential_executor_ms": sequential_executor_ms,
                "kernel_sum_ms": kernel_sum_ms,
                "node_span_sum_ms": node_span_sum_ms,
                "dispatch_gap_sum_ms": dispatch_gap_sum_ms,
                "in_node_overhead_ms": max(0.0, node_span_sum_ms - kernel_sum_ms),
                "run_boundary_overhead_ms": run_boundary_overhead_ms,
                "executor_unattributed_ms": executor_unattributed_ms,
                "model_run_head_gap_ms": model_run_head_gap_ms,
                "model_run_tail_gap_ms": model_run_tail_gap_ms,
                "sequential_head_gap_ms": sequential_head_gap_ms,
                "sequential_tail_gap_ms": sequential_tail_gap_ms,
            }
        )

    return summaries



def parse_node_exec_baseline(raw_profile_path, loops):
    events = _load_profile_events(raw_profile_path)
    overview = _profile_overview(events)
    model_run_windows, total_model_runs = _collect_session_windows(
        events, "model_run", loops, required=True
    )
    sequential_windows, total_sequential_runs = _collect_session_windows(
        events, "SequentialExecutor::Execute", loops, required=False
    )

    for item in model_run_windows:
        item["model_run_start_us"] = item["session_start_us"]
        item["model_run_end_us"] = item["session_end_us"]
        item["model_run_ms"] = item["session_ms"]

    for idx, item in enumerate(model_run_windows):
        if idx < len(sequential_windows):
            seq = sequential_windows[idx]
            item["sequential_executor_ms"] = seq["session_ms"]
            item["sequential_executor_start_us"] = seq["session_start_us"]
            item["sequential_executor_end_us"] = seq["session_end_us"]
        else:
            item["sequential_executor_ms"] = None
            item["sequential_executor_start_us"] = None
            item["sequential_executor_end_us"] = None

    node_execs = []
    for iteration_info in model_run_windows:
        node_execs.extend(_parse_iteration_node_execs(events, iteration_info))

    iteration_summary = build_iteration_summary(node_execs, model_run_windows)
    profile_stats = {
        "selected_iterations": len(model_run_windows),
        "total_profiled_model_runs": total_model_runs,
        "total_profiled_sequential_executor": total_sequential_runs,
        "warmup_iterations_ignored": total_model_runs - len(model_run_windows),
        "avg_profiled_model_run_ms": (
            sum(item["model_run_ms"] for item in model_run_windows) / float(len(model_run_windows))
            if model_run_windows
            else 0.0
        ),
        "avg_profiled_sequential_executor_ms": (
            sum(item["sequential_executor_ms"] for item in model_run_windows if item.get("sequential_executor_ms") is not None)
            / float(sum(1 for item in model_run_windows if item.get("sequential_executor_ms") is not None))
            if any(item.get("sequential_executor_ms") is not None for item in model_run_windows)
            else None
        ),
    }

    return node_execs, iteration_summary, overview, profile_stats



def join_node_execs_with_metadata(node_execs, metadata):
    by_key = {(item["node_name"], item["node_index"]): item for item in metadata}
    joined = []
    failures = []

    for record in node_execs:
        key = (record.get("node_name"), record.get("node_index"))
        meta = by_key.get(key)
        merged = dict(record)
        if meta is None:
            merged["kernel_type"] = None
            merged["signature_id"] = None
            merged["join_ok"] = False
            failures.append(
                {
                    "node_name": record.get("node_name"),
                    "node_index": record.get("node_index"),
                    "provider": record.get("provider"),
                }
            )
        else:
            merged["kernel_type"] = meta.get("kernel_type")
            merged["signature_id"] = meta.get("signature_id")
            merged["join_ok"] = True
        joined.append(merged)

    join_report = {
        "total_node_exec": len(node_execs),
        "joined_node_exec": sum(1 for item in joined if item["join_ok"]),
        "failed_node_exec": len(failures),
        "join_success_ratio": (
            sum(1 for item in joined if item["join_ok"]) / float(len(joined)) if joined else 0.0
        ),
        "failed_examples": failures[:20],
    }
    return joined, join_report



def _safe_mean(total, count):
    return total / float(count) if count else 0.0



def _build_summary_entry(model_name, group_key, group_value):
    return {
        "model_name": model_name,
        group_key: group_value,
        "exec_count": 0,
        "unique_node_count": 0,
        "total_kernel_time_ms": 0.0,
        "avg_kernel_time_ms": 0.0,
        "total_gpu_kernel_ms": 0.0,
        "avg_gpu_kernel_ms": 0.0,
        "total_gpu_copy_ms": 0.0,
        "avg_gpu_copy_ms": 0.0,
        "total_node_span_ms": 0.0,
        "avg_node_span_ms": 0.0,
        "total_dispatch_gap_ms": 0.0,
        "avg_dispatch_gap_ms": 0.0,
        "node_names": [],
        "signature_ids": [],
    }



def aggregate_exec_summary(model_name, node_execs, group_key):
    groups = {}
    node_names_by_group = defaultdict(set)
    signature_ids_by_group = defaultdict(set)

    for item in node_execs:
        group_value = item.get(group_key) or "__unjoined__"
        if group_value not in groups:
            groups[group_value] = _build_summary_entry(model_name, group_key, group_value)
        entry = groups[group_value]
        entry["exec_count"] += 1
        entry["total_kernel_time_ms"] += float(item.get("kernel_time_ms") or 0.0)
        entry["total_gpu_kernel_ms"] += float(item.get("gpu_kernel_ms") or 0.0)
        entry["total_gpu_copy_ms"] += float(item.get("gpu_copy_ms") or 0.0)
        entry["total_node_span_ms"] += float(item.get("node_span_ms") or 0.0)
        entry["total_dispatch_gap_ms"] += float(item.get("dispatch_gap_ms") or 0.0)
        if item.get("node_name"):
            node_names_by_group[group_value].add(item["node_name"])
        if item.get("signature_id"):
            signature_ids_by_group[group_value].add(item["signature_id"])

    result = []
    for group_value, entry in groups.items():
        entry["avg_kernel_time_ms"] = _safe_mean(entry["total_kernel_time_ms"], entry["exec_count"])
        entry["avg_gpu_kernel_ms"] = _safe_mean(entry["total_gpu_kernel_ms"], entry["exec_count"])
        entry["avg_gpu_copy_ms"] = _safe_mean(entry["total_gpu_copy_ms"], entry["exec_count"])
        entry["avg_node_span_ms"] = _safe_mean(entry["total_node_span_ms"], entry["exec_count"])
        entry["avg_dispatch_gap_ms"] = _safe_mean(
            entry["total_dispatch_gap_ms"], entry["exec_count"]
        )
        entry["node_names"] = sorted(node_names_by_group[group_value])
        entry["signature_ids"] = sorted(signature_ids_by_group[group_value])
        entry["unique_node_count"] = len(entry["node_names"])
        result.append(entry)

    result.sort(key=lambda item: item["total_node_span_ms"], reverse=True)
    return result



def build_gap_breakdown(iteration_summary):
    def _avg(key):
        values = [item[key] for item in iteration_summary if item.get(key) is not None]
        return sum(values) / float(len(values)) if values else None

    gap_breakdown = {
        "avg_in_node_overhead_ms": _avg("in_node_overhead_ms"),
        "avg_dispatch_gap_ms": _avg("dispatch_gap_sum_ms"),
        "avg_executor_unattributed_ms": _avg("executor_unattributed_ms"),
        "avg_run_boundary_overhead_ms": _avg("run_boundary_overhead_ms"),
        "avg_model_run_head_gap_ms": _avg("model_run_head_gap_ms"),
        "avg_model_run_tail_gap_ms": _avg("model_run_tail_gap_ms"),
        "avg_sequential_head_gap_ms": _avg("sequential_head_gap_ms"),
        "avg_sequential_tail_gap_ms": _avg("sequential_tail_gap_ms"),
    }

    candidates = {
        key: value
        for key, value in gap_breakdown.items()
        if value is not None and key.startswith("avg_")
    }
    if candidates:
        dominant_key = max(candidates, key=lambda item: candidates[item])
        gap_breakdown["dominant_gap_component"] = dominant_key
        gap_breakdown["dominant_gap_value_ms"] = candidates[dominant_key]
    else:
        gap_breakdown["dominant_gap_component"] = None
        gap_breakdown["dominant_gap_value_ms"] = None
    return gap_breakdown



def build_validation(
    metadata,
    node_execs,
    iteration_summary,
    profile_overview,
    profile_stats,
    join_report,
    model_e2e_ms,
    threshold,
):
    selected_iterations = max(int(profile_stats.get("selected_iterations", 0)), 1)
    complete_chain_count = sum(1 for item in node_execs if item.get("has_complete_chain"))
    node_span_ge_kernel_count = sum(
        1
        for item in node_execs
        if item.get("node_span_ms") is not None
        and item.get("kernel_time_ms") is not None
        and item["node_span_ms"] + 1e-9 >= item["kernel_time_ms"]
    )
    dispatch_gap_non_negative_count = sum(
        1 for item in node_execs if float(item.get("dispatch_gap_ms") or 0.0) >= -1e-9
    )

    total_kernel_time_ms = sum(float(item.get("kernel_time_ms") or 0.0) for item in node_execs)
    total_node_span_ms = sum(float(item.get("node_span_ms") or 0.0) for item in node_execs)
    total_dispatch_gap_ms = sum(float(item.get("dispatch_gap_ms") or 0.0) for item in node_execs)
    total_gpu_copy_ms = sum(float(item.get("gpu_copy_ms") or 0.0) for item in node_execs)

    avg_kernel_time_ms = total_kernel_time_ms / float(selected_iterations)
    avg_node_span_ms = total_node_span_ms / float(selected_iterations)
    avg_dispatch_gap_ms = total_dispatch_gap_ms / float(selected_iterations)
    avg_gpu_copy_ms = total_gpu_copy_ms / float(selected_iterations)

    expected_exec_count = len(metadata) * selected_iterations
    observed_exec_count = len(node_execs)

    def _relative_diff(observed, target):
        if target is None or target <= 1e-9:
            return None
        return abs(observed - target) / float(target)

    avg_profiled_model_run_ms = float(profile_stats.get("avg_profiled_model_run_ms", 0.0) or 0.0)
    avg_profiled_sequential_executor_ms = profile_stats.get("avg_profiled_sequential_executor_ms")
    kernel_vs_model_diff = _relative_diff(avg_kernel_time_ms, model_e2e_ms)
    node_span_vs_model_diff = _relative_diff(avg_node_span_ms, model_e2e_ms)
    node_span_vs_profiled_diff = _relative_diff(avg_node_span_ms, avg_profiled_model_run_ms)
    node_span_vs_seq_diff = _relative_diff(avg_node_span_ms, avg_profiled_sequential_executor_ms)
    gap_breakdown = build_gap_breakdown(iteration_summary)

    validation = {
        "stage": "A",
        "optimized_node_count": len(metadata),
        "selected_iterations": selected_iterations,
        "expected_node_exec_count": expected_exec_count,
        "observed_node_exec_count": observed_exec_count,
        "profile_overview": profile_overview,
        "profile_stats": profile_stats,
        "join_report": join_report,
        "gap_breakdown": gap_breakdown,
        "model_e2e_ms": model_e2e_ms,
        "avg_profiled_model_run_ms": avg_profiled_model_run_ms,
        "avg_profiled_sequential_executor_ms": avg_profiled_sequential_executor_ms,
        "avg_kernel_time_ms": avg_kernel_time_ms,
        "avg_node_span_ms": avg_node_span_ms,
        "avg_dispatch_gap_ms": avg_dispatch_gap_ms,
        "avg_gpu_copy_ms": avg_gpu_copy_ms,
        "kernel_vs_model_diff_ratio": kernel_vs_model_diff,
        "node_span_vs_model_diff_ratio": node_span_vs_model_diff,
        "node_span_vs_profiled_model_run_diff_ratio": node_span_vs_profiled_diff,
        "node_span_vs_profiled_sequential_executor_diff_ratio": node_span_vs_seq_diff,
        "complete_chain_count": complete_chain_count,
        "node_span_ge_kernel_count": node_span_ge_kernel_count,
        "dispatch_gap_non_negative_count": dispatch_gap_non_negative_count,
        "all_complete_chain": complete_chain_count == observed_exec_count,
        "all_node_span_ge_kernel": node_span_ge_kernel_count == observed_exec_count,
        "all_dispatch_gap_non_negative": dispatch_gap_non_negative_count == observed_exec_count,
        "observed_exec_matches_expected": observed_exec_count == expected_exec_count,
        "acceptance": {
            "complete_chain": complete_chain_count == observed_exec_count,
            "node_span_ge_kernel": node_span_ge_kernel_count == observed_exec_count,
            "dispatch_gap_non_negative": dispatch_gap_non_negative_count == observed_exec_count,
            "node_span_vs_model_within_threshold": (
                node_span_vs_model_diff is not None and node_span_vs_model_diff <= threshold
            ),
        },
    }
    validation["passed"] = all(validation["acceptance"].values())
    return validation



def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def _parse_provider_option(items):
    options = dict(DEFAULT_PROVIDER_OPTIONS)
    for item in items or []:
        if "=" not in item:
            raise ValueError("provider option 需为 key=value，当前: {}".format(item))
        key, value = item.split("=", 1)
        options[key.strip()] = value.strip()
    return options



def run_stage_a(args):
    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        raise ValueError("模型文件不存在: {}".format(model_path))

    model_name = _parse_model_name(model_path)
    output_dir = os.path.abspath(
        args.output_dir
        or os.path.join(PROJECT_ROOT, "benchmark", "results", "ort_stageA", model_name)
    )
    os.makedirs(output_dir, exist_ok=True)

    provider_options = _parse_provider_option(args.provider_option)
    providers = _build_provider_list(args.device_id, provider_options)

    logger.info("阶段 A 开始: model=%s output=%s", model_name, output_dir)
    logger.info(
        "执行参数: warmup=%d loops=%d device_id=%d provider_options=%s",
        args.warmup,
        args.loops,
        args.device_id,
        provider_options,
    )

    metadata = build_optimized_node_metadata(model_path)
    _write_json(os.path.join(output_dir, "optimized_node_metadata.json"), metadata)

    model_e2e_ms = measure_model_e2e_ms(
        model_path=model_path,
        providers=providers,
        warmup=args.warmup,
        loops=args.loops,
    )
    raw_profile_path = collect_raw_profile(
        model_path=model_path,
        output_dir=output_dir,
        providers=providers,
        warmup=args.warmup,
        loops=args.loops,
    )

    node_execs, iteration_summary, profile_overview, profile_stats = parse_node_exec_baseline(
        raw_profile_path=raw_profile_path,
        loops=args.loops,
    )
    joined_node_execs, join_report = join_node_execs_with_metadata(node_execs, metadata)

    kernel_type_summary = aggregate_exec_summary(model_name, joined_node_execs, "kernel_type")
    signature_summary = aggregate_exec_summary(model_name, joined_node_execs, "signature_id")
    validation = build_validation(
        metadata=metadata,
        node_execs=joined_node_execs,
        iteration_summary=iteration_summary,
        profile_overview=profile_overview,
        profile_stats=profile_stats,
        join_report=join_report,
        model_e2e_ms=model_e2e_ms,
        threshold=args.validation_threshold,
    )
    experiment_metadata = {
        "stage": "A",
        "model_name": model_name,
        "model_path": model_path,
        "output_dir": output_dir,
        "warmup": args.warmup,
        "loops": args.loops,
        "device_id": args.device_id,
        "provider_options": provider_options,
        "providers": providers,
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }

    _write_json(os.path.join(output_dir, "node_exec_baseline.json"), joined_node_execs)
    _write_json(os.path.join(output_dir, "iteration_summary.json"), iteration_summary)
    _write_json(os.path.join(output_dir, "kernel_type_summary.json"), kernel_type_summary)
    _write_json(os.path.join(output_dir, "signature_summary.json"), signature_summary)
    _write_json(os.path.join(output_dir, "join_report.json"), join_report)
    _write_json(os.path.join(output_dir, "experiment_metadata.json"), experiment_metadata)
    _write_json(os.path.join(output_dir, "validation.json"), validation)

    logger.info("阶段 A 完成: %s", output_dir)
    logger.info(
        "校验结果: passed=%s avg_kernel=%.6f avg_span=%.6f avg_seq=%.6f model_e2e=%.6f diff=%.6f",
        validation["passed"],
        validation["avg_kernel_time_ms"],
        validation["avg_node_span_ms"],
        validation["avg_profiled_sequential_executor_ms"]
        if validation["avg_profiled_sequential_executor_ms"] is not None
        else -1.0,
        validation["model_e2e_ms"],
        validation["node_span_vs_model_diff_ratio"]
        if validation["node_span_vs_model_diff_ratio"] is not None
        else -1.0,
    )



def main():
    parser = argparse.ArgumentParser(description="ORT full-model instrumented experiment stage A")
    parser.add_argument("--model", required=True, help="目标 ONNX 模型路径")
    parser.add_argument("--output-dir", default="", help="输出目录，默认 benchmark/results/ort_stageA/<model>")
    parser.add_argument("--warmup", type=int, default=20, help="warmup 次数")
    parser.add_argument("--loops", type=int, default=50, help="正式循环次数")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        help="附加 CUDA provider 选项，格式 key=value，可重复",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.15,
        help="sum(node_span_ms) 与 model_e2e_ms 相对误差阈值",
    )
    args = parser.parse_args()

    run_stage_a(args)


if __name__ == "__main__":
    main()
