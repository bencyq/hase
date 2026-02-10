# -*- coding: utf-8 -*-
"""
Task 5.2: Dataset Preparation

功能：
1. 支持输入单个 CSV 或 CSV 目录，合并为统一数据集。
2. 清理空值行（报告并删除）。
3. 解析 Kernel_ID / Input_Shape，提取结构化特征。
4. 加载 Task 5.1 的硬件性能特征算子时间并并入特征。
5. 对数值列做标准化（保留 Latency_ms 原值，同时新增 Latency_ms_z）。
6. 划分并导出 train.csv / test.csv。
"""
import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


KERNEL_ID_PATTERN = re.compile(
    r"^(?P<fusion_rule>.+?)_bs(?P<batch>\d+)_(?P<channel>\d+)c(?P<height>\d+)x(?P<width>\d+)_k(?P<kernel_h>\d+)x(?P<kernel_w>\d+)$"
)
INPUT_SHAPE_PATTERN = re.compile(
    r"^(?P<batch>\d+)x(?P<channel>\d+)x(?P<height>\d+)x(?P<width>\d+)$"
)


def collect_csv_files(input_path: str) -> List[str]:
    """收集输入路径下的 CSV 文件列表。"""
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".csv"):
            return [input_path]
        raise ValueError("输入是文件，但不是 CSV: {}".format(input_path))

    if not os.path.isdir(input_path):
        raise ValueError("输入路径不存在: {}".format(input_path))

    files = [
        os.path.join(input_path, f)
        for f in sorted(os.listdir(input_path))
        if f.lower().endswith(".csv")
    ]
    if not files:
        raise ValueError("目录下未找到 CSV 文件: {}".format(input_path))
    return files


def read_and_merge_csv(csv_files: List[str]) -> pd.DataFrame:
    """读取并合并多个 CSV。"""
    parts = []
    for path in csv_files:
        df = pd.read_csv(path)
        df["Source_File"] = os.path.basename(path)
        parts.append(df)
    merged = pd.concat(parts, ignore_index=True)
    return merged


def report_and_drop_empty_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """报告并删除包含空值的行。"""
    cleaned = df.replace(r"^\s*$", np.nan, regex=True)
    empty_mask = cleaned.isna().any(axis=1)
    empty_count = int(empty_mask.sum())

    if empty_count > 0:
        cols = ["Source_File", "Kernel_ID", "OpType", "Input_Shape"]
        existing_cols = [c for c in cols if c in cleaned.columns]
        print("发现空值行: {} 行，已删除。示例:".format(empty_count))
        print(cleaned.loc[empty_mask, existing_cols].head(10).to_string(index=False))

    return cleaned.loc[~empty_mask].copy(), empty_count


def parse_kernel_id(df: pd.DataFrame) -> pd.DataFrame:
    """解析 Kernel_ID，并与 Input_Shape 互补填充结构化特征。"""
    required = ["Kernel_ID", "OpType", "Input_Shape"]
    for col in required:
        if col not in df.columns:
            raise ValueError("缺少必要列: {}".format(col))

    parsed = df["Kernel_ID"].astype(str).str.extract(KERNEL_ID_PATTERN)
    shape_parsed = df["Input_Shape"].astype(str).str.extract(INPUT_SHAPE_PATTERN)

    # 兼容非4维形状（如 128x1000），按 N,C,H,W 规则补齐缺失维度为 1
    generic_dims = (
        df["Input_Shape"]
        .astype(str)
        .str.split("x")
        .apply(lambda x: [int(v) for v in x if v.isdigit()])
    )
    generic_batch = generic_dims.apply(lambda d: d[0] if len(d) >= 1 else np.nan)
    generic_channel = generic_dims.apply(lambda d: d[1] if len(d) >= 2 else 1)
    generic_height = generic_dims.apply(lambda d: d[2] if len(d) >= 3 else 1)
    generic_width = generic_dims.apply(lambda d: d[3] if len(d) >= 4 else 1)

    # 统一融合规则命名：优先 Kernel_ID 解析，失败则回退 OpType
    df["kernel_fusion_rule"] = parsed["fusion_rule"].fillna(df["OpType"].astype(str))

    # 从 Kernel_ID 或 Input_Shape 提取 batch/channel/height/width
    for key in ["batch", "channel", "height", "width"]:
        from_kernel_id = pd.to_numeric(parsed[key], errors="coerce")
        from_shape = pd.to_numeric(shape_parsed[key], errors="coerce")
        if key == "batch":
            fallback = generic_batch
        elif key == "channel":
            fallback = generic_channel
        elif key == "height":
            fallback = generic_height
        else:
            fallback = generic_width
        df[key] = from_kernel_id.fillna(from_shape).fillna(fallback)

    df["kernel_h"] = pd.to_numeric(parsed["kernel_h"], errors="coerce").fillna(1)
    df["kernel_w"] = pd.to_numeric(parsed["kernel_w"], errors="coerce").fillna(1)

    # 同一算子组关联特征：让模型可学习同融合规则下不同尺寸间关系
    df["kernel_group"] = df["kernel_fusion_rule"]

    before = len(df)
    numeric_needed = ["batch", "channel", "height", "width", "kernel_h", "kernel_w"]
    df = df.dropna(subset=numeric_needed).copy()
    dropped = before - len(df)
    if dropped > 0:
        print("Kernel_ID/Input_Shape 无法解析的行: {} 行，已删除。".format(dropped))

    return df


def add_performance_kernel_features(df: pd.DataFrame, perf_json_path: str) -> pd.DataFrame:
    """加载 Task5.1 的硬件性能特征算子时间并加入数据集。"""
    if not os.path.isfile(perf_json_path):
        raise ValueError("性能特征文件不存在: {}".format(perf_json_path))

    with open(perf_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    latency_map: Dict[str, float] = payload.get("latency_ms", {})
    if not latency_map:
        raise ValueError("性能特征文件缺少 latency_ms 内容: {}".format(perf_json_path))

    for name, value in latency_map.items():
        df["hwpk_{}".format(name)] = float(value)
    return df


def build_numeric_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """做类别编码、数值化和标准化。"""
    if "Latency_ms" not in df.columns:
        raise ValueError("缺少必要列: Latency_ms")

    numeric_candidates = [
        c for c in df.columns if c.startswith("DCGM_") or c in ["Latency_ms"]
    ]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Latency_ms"]).copy()
    dropped = before - len(df)
    if dropped > 0:
        print("Latency_ms 非法或空值行: {} 行，已删除。".format(dropped))

    # 对类别特征进行 One-Hot，保留同类算子的共享列以建立关联
    cat_cols = [c for c in ["GPU", "OpType", "kernel_fusion_rule", "kernel_group"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(float)

    # 标准化：对所有数值列做 z-score；Latency_ms 保留原值并附加 Latency_ms_z
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std(ddof=0).replace(0, 1.0)
    z = (df[numeric_cols] - means) / stds

    for col in numeric_cols:
        if col == "Latency_ms":
            df["Latency_ms_raw"] = df["Latency_ms"]
            df["Latency_ms_z"] = z[col]
        else:
            df[col] = z[col]

    return df


def split_train_test(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按比例划分 train / test。"""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须在 (0, 1) 范围内。")
    if len(df) < 2:
        raise ValueError("有效数据不足，无法划分 train/test。")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_n = max(1, int(len(shuffled) * test_size))
    test_df = shuffled.iloc[:test_n].copy()
    train_df = shuffled.iloc[test_n:].copy()
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Task5.2 dataset preparation")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark/csv",
        help="CSV 文件或目录路径（目录会读取所有 CSV）",
    )
    parser.add_argument(
        "--performance-kernel-json",
        type=str,
        default="training/performance_kernel_times.json",
        help="Task5.1 输出 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/dataset",
        help="输出目录（merged/train/test）",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    csv_files = collect_csv_files(args.input)
    print("读取 CSV 文件数: {}".format(len(csv_files)))

    merged = read_and_merge_csv(csv_files)
    cleaned, dropped_empty = report_and_drop_empty_rows(merged)
    print("空值删除后数据量: {}（删除 {} 行）".format(len(cleaned), dropped_empty))

    parsed = parse_kernel_id(cleaned)
    featured = add_performance_kernel_features(parsed, args.performance_kernel_json)
    final_df = build_numeric_dataset(featured)

    train_df, test_df = split_train_test(final_df, args.test_size, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    merged_path = os.path.join(args.output_dir, "merged.csv")
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    final_df.to_csv(merged_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("已输出: {}".format(merged_path))
    print("已输出: {}".format(train_path))
    print("已输出: {}".format(test_path))
    print("Train/Test 行数: {}/{}".format(len(train_df), len(test_df)))
    print("处理后 DataFrame 头部:")
    print(final_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
