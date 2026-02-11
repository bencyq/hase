# -*- coding: utf-8 -*-
"""
Task 7.2: Evaluation

调用 Task 7.1 预测器，对比实际模型延迟与预测延迟。
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from inference.predict_model_latency import predict_model_latency  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("evaluator")


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(root, path))


def _collect_model_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".onnx"):
            return [input_path]
        raise ValueError("输入文件不是 .onnx: {}".format(input_path))

    if not os.path.isdir(input_path):
        raise ValueError("输入路径不存在: {}".format(input_path))

    files = [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if f.lower().endswith(".onnx")
    ]
    if not files:
        raise ValueError("目录下未找到 .onnx 文件: {}".format(input_path))
    return files


def _parse_batch_input_from_name(model_path: str) -> Tuple[Optional[int], Optional[int]]:
    stem = os.path.splitext(os.path.basename(model_path))[0]
    # 兼容 <model>_bs16_224x224
    if "_bs" not in stem or "_" not in stem:
        return None, None
    try:
        right = stem.split("_bs", 1)[1]
        batch = int(right.split("_", 1)[0])
        size = int(right.rsplit("_", 1)[1].split("x", 1)[0])
        return batch, size
    except Exception:
        return None, None


def _ort_type_to_numpy(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int16)": np.int16,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(ort_type, np.float32)


def _infer_dim_value(
    dim, dim_idx: int, rank: int, batch_hint: Optional[int], input_hint: Optional[int]
) -> int:
    if isinstance(dim, int) and dim > 0:
        return int(dim)

    dim_name = str(dim).lower() if dim is not None else ""
    if "batch" in dim_name or dim_name == "n":
        return int(batch_hint or 1)
    if "channel" in dim_name or dim_name == "c":
        return 3 if rank >= 4 else 1
    if "height" in dim_name or dim_name == "h":
        return int(input_hint or 224)
    if "width" in dim_name or dim_name == "w":
        return int(input_hint or 224)

    if rank >= 4:
        if dim_idx == 0:
            return int(batch_hint or 1)
        if dim_idx == 1:
            return 3
        return int(input_hint or 224)
    return 1


def _build_dummy_inputs(session: ort.InferenceSession, model_path: str) -> Dict[str, np.ndarray]:
    batch_hint, input_hint = _parse_batch_input_from_name(model_path)
    feeds: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        rank = len(inp.shape)
        shape = [
            _infer_dim_value(d, idx, rank, batch_hint=batch_hint, input_hint=input_hint)
            for idx, d in enumerate(inp.shape)
        ]
        np_dtype = _ort_type_to_numpy(inp.type)

        if np.issubdtype(np_dtype, np.floating):
            value = np.random.rand(*shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            value = (np.random.rand(*shape) > 0.5).astype(np.bool_)
        else:
            value = np.random.randint(0, 10, size=shape, dtype=np_dtype)
        feeds[inp.name] = value
    return feeds


def _measure_actual_latency_ms(
    model_path: str, warmup: int, loops: int, use_cuda: bool, device_id: int
) -> float:
    if use_cuda:
        providers = [("CUDAExecutionProvider", {"device_id": str(device_id)}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)
    feeds = _build_dummy_inputs(session, model_path)
    outputs = [o.name for o in session.get_outputs()]

    for _ in range(max(0, warmup)):
        session.run(outputs, feeds)

    begin = time.perf_counter()
    for _ in range(max(1, loops)):
        session.run(outputs, feeds)
    elapsed = time.perf_counter() - begin
    return float(elapsed * 1000.0 / float(max(1, loops)))


def evaluate_one_model(model_path: str, args) -> Dict:
    pred_ms = predict_model_latency(
        model_path=model_path,
        config_path=args.config,
        dag_dir=args.dag_dir,
        kernel_record_dir=args.kernel_record_dir,
        regressor_path=args.regressor,
        preprocess_meta_path=args.preprocess_meta,
        perf_json_path=args.perf_json,
        sequential_stream0=not args.non_sequential,
    )
    actual_ms = _measure_actual_latency_ms(
        model_path=model_path,
        warmup=args.warmup,
        loops=args.loops,
        use_cuda=not args.cpu_only,
        device_id=args.device_id,
    )
    diff_ms = pred_ms - actual_ms
    ape = abs(diff_ms) / max(actual_ms, 1e-9) * 100.0
    return {
        "model": model_path,
        "predicted_ms": float(pred_ms),
        "actual_ms": float(actual_ms),
        "diff_ms": float(diff_ms),
        "ape_percent": float(ape),
    }


def main():
    parser = argparse.ArgumentParser(description="Task7.2 evaluate predicted vs actual model latency")
    parser.add_argument("--input", type=str, required=True, help="模型文件(.onnx)或目录")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "inference", "config.yaml"),
        help="Prometheus 配置文件路径",
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
    parser.add_argument("--non-sequential", action="store_true", help="预测阶段使用 DAG 最长路径")
    parser.add_argument("--warmup", type=int, default=10, help="实测模型预热次数")
    parser.add_argument("--loops", type=int, default=20, help="实测模型循环次数")
    parser.add_argument("--cpu-only", action="store_true", help="实测阶段仅使用 CPUExecutionProvider")
    parser.add_argument("--device-id", type=int, default=0, help="实测 CUDA device id")
    parser.add_argument("--output-json", type=str, default="", help="可选：保存评估结果 JSON")
    args = parser.parse_args()

    input_path = _resolve_path(PROJECT_ROOT, args.input)
    args.config = _resolve_path(PROJECT_ROOT, args.config)
    args.dag_dir = _resolve_path(PROJECT_ROOT, args.dag_dir)
    args.kernel_record_dir = _resolve_path(PROJECT_ROOT, args.kernel_record_dir)
    args.regressor = _resolve_path(PROJECT_ROOT, args.regressor)
    args.preprocess_meta = _resolve_path(PROJECT_ROOT, args.preprocess_meta)
    args.perf_json = _resolve_path(PROJECT_ROOT, args.perf_json)
    output_json = _resolve_path(PROJECT_ROOT, args.output_json) if args.output_json else ""

    model_files = _collect_model_files(input_path)
    results = []
    for model_path in model_files:
        try:
            item = evaluate_one_model(model_path, args)
            results.append(item)
            print(
                "{} | pred={:.6f} ms | actual={:.6f} ms | diff={:.6f} ms | APE={:.4f}%".format(
                    os.path.basename(model_path),
                    item["predicted_ms"],
                    item["actual_ms"],
                    item["diff_ms"],
                    item["ape_percent"],
                )
            )
        except Exception as e:
            logger.error("评估失败: %s, err=%s", model_path, e)
            results.append(
                {
                    "model": model_path,
                    "error": str(e),
                }
            )
            print("{} | ERROR: {}".format(os.path.basename(model_path), e))

    valid = [x for x in results if "error" not in x]
    if valid:
        mean_ape = sum(x["ape_percent"] for x in valid) / float(len(valid))
        mean_diff = sum(abs(x["diff_ms"]) for x in valid) / float(len(valid))
        print("SUMMARY | count={} | mean_abs_diff_ms={:.6f} | mean_APE={:.4f}%".format(len(valid), mean_diff, mean_ape))
    else:
        print("SUMMARY | no valid result")

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("结果已保存: {}".format(output_json))


if __name__ == "__main__":
    main()
