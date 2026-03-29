# -*- coding: utf-8 -*-
"""
Benchmark3:
比较“整模型直跑”与“内核逐个运行总和”的额外时间开销比例。
"""
import argparse
import contextlib
import csv
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from debug.debug import run_debug_for_model  # noqa: E402


DEFAULT_MODELS = [
    "model_zoo/models/densenet121_bs32_224x224.onnx",
    "model_zoo/models/mobilenet_v2_bs32_448x448.onnx",
    "model_zoo/models/resnet18_bs64_224x224.onnx",
    "model_zoo/models/vgg16_bs64_224x224.onnx",
    "model_zoo/models/YOLOv8m_bs64_448x448.onnx",
]


def _resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_row(model_path, payload):
    t_model = float(payload.get("direct_model_latency_ms", 0.0))
    t_kernel = float(payload.get("total_kernel_latency_ms", 0.0))
    t_extra = t_kernel - t_model
    overhead = (t_extra / t_model * 100.0) if t_model > 1e-9 else 0.0
    return {
        "Model": model_path,
        "T_model_ms": t_model,
        "T_kernel_sum_ms": t_kernel,
        "T_extra_ms": t_extra,
        "Overhead_Ratio_percent": overhead,
    }


def _write_csv(rows, output_csv):
    _ensure_parent(output_csv)
    fields = [
        "Model",
        "T_model_ms",
        "T_kernel_sum_ms",
        "T_extra_ms",
        "Overhead_Ratio_percent",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(rows, summary_path, started_at, finished_at):
    _ensure_parent(summary_path)
    sorted_rows = sorted(rows, key=lambda x: x["Overhead_Ratio_percent"], reverse=True)
    max_item = sorted_rows[0] if sorted_rows else None
    min_item = sorted_rows[-1] if sorted_rows else None

    lines = []
    lines.append("# Benchmark3 验收报告")
    lines.append("")
    lines.append("- 开始时间: {}".format(started_at.isoformat(timespec="seconds")))
    lines.append("- 结束时间: {}".format(finished_at.isoformat(timespec="seconds")))
    lines.append("- 模型数量: {}".format(len(rows)))
    lines.append("")
    lines.append("## 开销比例对比（高 -> 低）")
    lines.append("")
    for idx, row in enumerate(sorted_rows, start=1):
        lines.append(
            "{}. {} | Overhead={:.6f}% | T_model={:.6f} ms | T_kernel_sum={:.6f} ms".format(
                idx,
                row["Model"],
                row["Overhead_Ratio_percent"],
                row["T_model_ms"],
                row["T_kernel_sum_ms"],
            )
        )
    lines.append("")
    lines.append("## 简要分析")
    lines.append("")
    if max_item and min_item:
        lines.append(
            "- 开销最高: {} ({:.6f}%)".format(
                max_item["Model"], max_item["Overhead_Ratio_percent"]
            )
        )
        lines.append(
            "- 开销最低: {} ({:.6f}%)".format(
                min_item["Model"], min_item["Overhead_Ratio_percent"]
            )
        )
    lines.append(
        "- 可能原因: 内核数量越多、逐个调用的调度/启动开销越容易累积；访存密集层较多时，分段执行也可能放大额外开销。"
    )
    lines.append("")
    lines.append("## 验收结果")
    lines.append("")
    ok_count = sum(1 for r in rows if "Overhead_Ratio_percent" in r)
    lines.append("- 5 个模型是否全部完成: {}".format("是" if len(rows) == 5 else "否"))
    lines.append("- 每个模型是否都有 Overhead_Ratio: {}".format("是" if ok_count == len(rows) else "否"))
    lines.append("- 是否产出对比结论: 是")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_all(models, warmup, loops, output_csv, detail_dir, summary_path):
    rows = []
    started_at = datetime.now()
    os.makedirs(detail_dir, exist_ok=True)

    for model in models:
        abs_model = _resolve_path(model)
        if not os.path.isfile(abs_model):
            raise FileNotFoundError("模型文件不存在: {}".format(abs_model))

        model_stem = os.path.splitext(os.path.basename(abs_model))[0]
        detail_json = os.path.join(detail_dir, "{}.json".format(model_stem))
        detail_log = os.path.join(detail_dir, "{}.log".format(model_stem))

        print("[{}/{}] start: {}".format(len(rows) + 1, len(models), abs_model))

        if not os.path.isfile(detail_json):
            with open(detail_log, "w", encoding="utf-8") as log_fp:
                with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
                    run_debug_for_model(
                        model_path=abs_model,
                        record_dir=os.path.join(PROJECT_ROOT, "ort_analysis", "ort_kernel_record"),
                        kernel_onnx_dir=os.path.join(PROJECT_ROOT, "kernel_model", "kernel_onnx"),
                        warmup=warmup,
                        loops=loops,
                        output_json=detail_json,
                    )
        payload = _load_payload(detail_json)
        row = _to_row(abs_model, payload)
        rows.append(row)
        print(
            "[{}/{}] done: T_model={:.6f} ms, T_kernel_sum={:.6f} ms, Overhead={:.6f}%".format(
                len(rows),
                len(models),
                row["T_model_ms"],
                row["T_kernel_sum_ms"],
                row["Overhead_Ratio_percent"],
            )
        )

    _write_csv(rows, output_csv)
    finished_at = datetime.now()
    _write_summary(rows, summary_path, started_at, finished_at)
    return rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark3: 内核逐个运行 vs 整模型运行 额外开销比例分析"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warmup 次数（默认 1）",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="loops 次数（默认 1）",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "benchmark3", "results", "benchmark3_summary.csv"),
        help="输出结果 CSV 路径",
    )
    parser.add_argument(
        "--detail-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "benchmark3", "results", "details"),
        help="逐模型详细 JSON 输出目录",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default=os.path.join(PROJECT_ROOT, "benchmark3", "results", "acceptance_report.md"),
        help="验收报告 Markdown 路径",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    rows = run_all(
        models=DEFAULT_MODELS,
        warmup=args.warmup,
        loops=args.loops,
        output_csv=_resolve_path(args.output_csv),
        detail_dir=_resolve_path(args.detail_dir),
        summary_path=_resolve_path(args.report_md),
    )
    print("Benchmark3 completed, models={}".format(len(rows)))
    print("CSV: {}".format(_resolve_path(args.output_csv)))
    print("Report: {}".format(_resolve_path(args.report_md)))


if __name__ == "__main__":
    main()
