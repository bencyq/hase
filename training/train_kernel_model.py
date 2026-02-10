# -*- coding: utf-8 -*-
"""
Task 5.2 + Task 5.3: Dataset Preparation & Regressor Training

功能：
1. 支持输入单个 CSV 或 CSV 目录，合并为统一数据集。
2. 清理空值行（报告并删除）。
3. 解析 Kernel_ID / Input_Shape，提取结构化特征。
4. 根据 GPU 型号加载 Task 5.1 的硬件性能特征算子时间并并入特征。
5. 保留 Latency_ms 原值，对其他特征做数值化与标准化。
6. 划分并导出 train.csv / test.csv。
7. 训练多个回归模型（XGBoost、RandomForest、LightGBM），评估 MAPE 并保存模型。
"""
import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error


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
    """按 GPU 型号加载 Task5.1 性能特征并加入数据集。"""
    if not os.path.isfile(perf_json_path):
        raise ValueError("性能特征文件不存在: {}".format(perf_json_path))
    if "GPU" not in df.columns:
        raise ValueError("缺少必要列: GPU")

    with open(perf_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records = payload if isinstance(payload, list) else [payload]
    gpu_to_latency: Dict[str, Dict[str, float]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        gpu_name = item.get("gpu_name")
        latency_map = item.get("latency_ms", {})
        if not gpu_name or not isinstance(latency_map, dict):
            continue
        gpu_to_latency[str(gpu_name)] = {k: float(v) for k, v in latency_map.items()}

    required_ops = [
        "compute_matmul_fp32",
        "compute_conv2d_fp32",
        "memory_elementwise_add_fp32",
        "memory_clone_fp32",
    ]
    if not gpu_to_latency:
        raise ValueError("性能特征文件缺少可用的 gpu_name/latency_ms 内容: {}".format(perf_json_path))

    missing_gpu = sorted(set(df["GPU"].astype(str)) - set(gpu_to_latency.keys()))
    if missing_gpu:
        raise ValueError(
            "以下 GPU 在性能特征文件中不存在映射: {}".format(", ".join(missing_gpu))
        )

    for op_name in required_ops:
        col = "hwpk_{}".format(op_name)
        df[col] = df["GPU"].astype(str).map(
            lambda g: gpu_to_latency[g].get(op_name, np.nan)
        )

    if df[[f"hwpk_{x}" for x in required_ops]].isna().any().any():
        raise ValueError("性能特征映射后存在空值，请检查 performance_kernel_times.json 中的算子字段。")
    return df


def build_numeric_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """做类别编码、数值化和标准化（保留 Latency_ms 原值），并返回预处理元信息。"""
    if "Latency_ms" not in df.columns:
        raise ValueError("缺少必要列: Latency_ms")
    if "GPU" not in df.columns:
        raise ValueError("缺少必要列: GPU")
    if "OpType" not in df.columns:
        raise ValueError("缺少必要列: OpType")
    if "kernel_group" not in df.columns:
        raise ValueError("缺少必要列: kernel_group")

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

    # 单列类别编码：避免 One-Hot 展开为大量列，同时保存映射供推理复用
    df["GPU Type"] = df["GPU"].astype(str)
    code_cols = ["GPU Type", "OpType", "kernel_group"]
    category_maps: Dict[str, Dict[str, int]] = {}
    for col in code_cols:
        values = sorted(df[col].astype(str).unique().tolist())
        mapping = {name: idx for idx, name in enumerate(values)}
        category_maps[col] = mapping
        df[col] = df[col].astype(str).map(mapping).astype(float)

    # 标准化：仅对特征列做 z-score，不改动目标列 Latency_ms
    feature_numeric_cols = [
        c
        for c in df.columns
        if c != "Latency_ms" and pd.api.types.is_numeric_dtype(df[c])
    ]
    means = df[feature_numeric_cols].mean()
    stds = df[feature_numeric_cols].std(ddof=0).replace(0, 1.0)
    df[feature_numeric_cols] = (df[feature_numeric_cols] - means) / stds

    preprocess_meta = {
        "target_col": "Latency_ms",
        "category_maps": category_maps,
        "scaler": {
            "mean": {k: float(v) for k, v in means.to_dict().items()},
            "std": {k: float(v) for k, v in stds.to_dict().items()},
        },
    }
    return df, preprocess_meta


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


def build_regressors(seed: int):
    """
    构建多个回归模型。
    XGBoost / LightGBM 如缺失依赖则跳过并提示。
    """
    models = {}
    unavailable = {}

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    try:
        from xgboost import XGBRegressor  # type: ignore

        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception as e:
        unavailable["XGBoost"] = str(e).split("\n")[0]

    try:
        from lightgbm import LGBMRegressor  # type: ignore

        models["LightGBM"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
    except Exception as e:
        unavailable["LightGBM"] = str(e).split("\n")[0]

    return models, unavailable


def build_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """从处理后的数据构造训练/测试特征与标签。"""
    target_col = "Latency_ms"
    drop_cols = {"Kernel_ID", "Input_Shape", "Source_File", "GPU", "Latency_ms"}

    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]

    x_train = train_df[feature_cols]
    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    x_test = test_df[feature_cols]
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")
    return x_train, y_train, x_test, y_test, feature_cols, target_col


def train_and_save_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_dir: str,
    seed: int,
    preprocess_meta: Optional[Dict] = None,
):
    """训练模型，评估 MAPE，并保存模型与评估结果。"""
    os.makedirs(model_dir, exist_ok=True)

    x_train, y_train, x_test, y_test, feature_cols, target_col = build_xy(train_df, test_df)
    models, unavailable = build_regressors(seed)

    if unavailable:
        for name, reason in unavailable.items():
            print("模型 {} 跳过：{}".format(name, reason))

    if not models:
        raise RuntimeError("没有可用的回归模型，请安装依赖后重试。")

    metrics = {"target_column": target_col, "feature_count": len(feature_cols), "mape": {}}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mape = float(mean_absolute_percentage_error(y_test, pred) * 100.0)
        metrics["mape"][model_name] = mape

        model_path = os.path.join(model_dir, "{}.joblib".format(model_name.lower()))
        dump(model, model_path)
        print("{} MAPE: {:.4f}%".format(model_name, mape))
        print("模型已保存: {}".format(model_path))

    ranking = sorted(metrics["mape"].items(), key=lambda x: x[1])
    print("测试集 MAPE 对比（越小越好）:")
    for idx, (name, score) in enumerate(ranking, start=1):
        print("{}. {}: {:.4f}%".format(idx, name, score))

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("评估结果已保存: {}".format(metrics_path))

    if preprocess_meta is not None:
        preprocess_meta = dict(preprocess_meta)
        preprocess_meta["feature_cols"] = feature_cols
        preprocess_meta_path = os.path.join(model_dir, "preprocess_meta.json")
        with open(preprocess_meta_path, "w", encoding="utf-8") as f:
            json.dump(preprocess_meta, f, ensure_ascii=False, indent=2)
        print("预处理元信息已保存: {}".format(preprocess_meta_path))


def preprocess_and_split(args):
    """执行 Task5.2：数据预处理并导出 merged/train/test。"""
    csv_files = collect_csv_files(args.input)
    print("读取 CSV 文件数: {}".format(len(csv_files)))

    merged = read_and_merge_csv(csv_files)
    cleaned, dropped_empty = report_and_drop_empty_rows(merged)
    print("空值删除后数据量: {}（删除 {} 行）".format(len(cleaned), dropped_empty))

    parsed = parse_kernel_id(cleaned)
    featured = add_performance_kernel_features(parsed, args.performance_kernel_json)
    final_df, preprocess_meta = build_numeric_dataset(featured)

    # 输出前去掉不需要的原始标识列
    drop_for_output = [
        c
        for c in ["Kernel_ID", "Input_Shape", "Source_File", "GPU", "kernel_fusion_rule"]
        if c in final_df.columns
    ]
    if drop_for_output:
        final_df = final_df.drop(columns=drop_for_output)

    preprocess_meta["feature_cols"] = [
        c for c in final_df.columns if c != "Latency_ms" and pd.api.types.is_numeric_dtype(final_df[c])
    ]

    train_df, test_df = split_train_test(final_df, args.test_size, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    merged_path = os.path.join(args.output_dir, "merged.csv")
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    final_df.to_csv(merged_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    preprocess_meta_path = os.path.join(args.output_dir, "preprocess_meta.json")
    with open(preprocess_meta_path, "w", encoding="utf-8") as f:
        json.dump(preprocess_meta, f, ensure_ascii=False, indent=2)

    print("已输出: {}".format(merged_path))
    print("已输出: {}".format(train_path))
    print("已输出: {}".format(test_path))
    print("已输出: {}".format(preprocess_meta_path))
    print("Train/Test 行数: {}/{}".format(len(train_df), len(test_df)))
    print("处理后 DataFrame 头部:")
    print(final_df.head().to_string(index=False))
    return train_df, test_df, preprocess_meta


def load_train_test_from_output(output_dir: str):
    """读取已存在的 train/test 数据集，用于仅训练模式。"""
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        raise ValueError(
            "仅训练模式需要已有 train/test 文件: {}, {}".format(train_path, test_path)
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("读取已有数据集: train={}, test={}".format(len(train_df), len(test_df)))
    return train_df, test_df


def load_preprocess_meta_from_output(output_dir: str) -> Optional[Dict]:
    """从输出目录读取预处理元信息（如果存在）。"""
    meta_path = os.path.join(output_dir, "preprocess_meta.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Task5.2+5.3 dataset preparation and training")
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
        help="Task5.1 输出的 JSON 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training/dataset",
        help="输出目录（merged/train/test）",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["preprocess", "train", "all"],
        help="执行模式：preprocess=只预处理，train=只训练，all=全流程",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="training/models",
        help="模型输出目录（Task5.3）",
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess_and_split(args)
        print("仅预处理完成（--mode preprocess）。")
        return

    if args.mode == "train":
        train_df, test_df = load_train_test_from_output(args.output_dir)
        preprocess_meta = load_preprocess_meta_from_output(args.output_dir)
    else:
        train_df, test_df, preprocess_meta = preprocess_and_split(args)

    train_and_save_models(train_df, test_df, args.model_dir, args.seed, preprocess_meta)


if __name__ == "__main__":
    main()
