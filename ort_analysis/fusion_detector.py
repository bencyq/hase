# -*- coding: utf-8 -*-
"""
Fusion Detector & Kernel Recorder
识别 ORT 优化后模型中的算子融合策略，提取每个 Kernel 的元数据，
以统一命名保存到 ort_kernel_record/。
"""
import os
import sys
import json
import glob
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from ort_analysis.ort_graph_parser import (
    get_optimized_model, _build_initial_shape_map,
    _infer_node_output_shape, _get_attr,
)

logger = get_logger("fusion_detector")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(SCRIPT_DIR, "ort_kernel_record")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_zoo", "models")


# ──────────────────────────────────────────────
# 1. 融合检测 & 统一命名
# ──────────────────────────────────────────────

def detect_kernel_type(node):
    """
    检测节点的融合内核类型，返回统一命名。
    例: FusedConv(activation=Relu, 4 inputs) → "Conv_Add_Relu"
    """
    op = node.op_type

    if op == "FusedConv":
        activation = _get_attr(node, "activation", "")
        has_residual = len(node.input) >= 4  # X, W, B, Z(residual)
        parts = ["Conv"]
        if has_residual:
            parts.append("Add")
        if activation:
            parts.append(activation)
        return "_".join(parts)

    # 其他算子直接使用 op_type
    return op


def _fusion_description(kernel_type):
    """生成可读的融合描述"""
    descs = {
        "Conv_Relu": "Conv + Bias + Relu",
        "Conv_Add_Relu": "Conv + Bias + Add(residual) + Relu",
        "Conv": "Conv + Bias",
        "MaxPool": "MaxPool",
        "GlobalAveragePool": "GlobalAveragePool",
        "Flatten": "Flatten",
        "Gemm": "Gemm (fully connected)",
    }
    return descs.get(kernel_type, kernel_type)


# ──────────────────────────────────────────────
# 2. 属性 & FLOPs
# ──────────────────────────────────────────────

_ATTR_KEYS = {
    "Conv":      ["kernel_shape", "strides", "pads", "dilations", "group"],
    "FusedConv": ["kernel_shape", "strides", "pads", "dilations", "group", "activation"],
    "MaxPool":   ["kernel_shape", "strides", "pads", "ceil_mode"],
    "Gemm":      ["transA", "transB"],
    "Flatten":   ["axis"],
}


def _extract_attributes(node):
    """提取节点的关键属性"""
    attrs = {}
    for key in _ATTR_KEYS.get(node.op_type, []):
        val = _get_attr(node, key)
        if val is not None:
            attrs[key] = val
    return attrs


def _estimate_flops(kernel_type, attrs, act_shape, weight_shape, out_shape):
    """
    估算 FLOPs。
    act_shape:    主 activation 输入 shape (e.g. [N,C,H,W])
    weight_shape: 权重 shape (e.g. [C_out,C_in,kH,kW])
    out_shape:    输出 shape
    """
    if kernel_type in ("Conv", "Conv_Relu", "Conv_Add_Relu"):
        if out_shape and weight_shape and len(out_shape) == 4:
            n, c_out, h_out, w_out = out_shape
            c_in, kh, kw = weight_shape[1], weight_shape[2], weight_shape[3]
            group = attrs.get("group", 1)
            flops = 2 * n * c_out * h_out * w_out * (c_in // group) * kh * kw
            flops += n * c_out * h_out * w_out          # bias
            if "Add" in kernel_type:
                flops += n * c_out * h_out * w_out      # residual add
            if "Relu" in kernel_type:
                flops += n * c_out * h_out * w_out      # relu
            return flops

    elif kernel_type == "MaxPool":
        if out_shape and len(out_shape) == 4:
            n, c, h_out, w_out = out_shape
            ks = attrs.get("kernel_shape", [3, 3])
            return n * c * h_out * w_out * ks[0] * ks[1]

    elif kernel_type == "GlobalAveragePool":
        if act_shape and len(act_shape) == 4:
            n, c, h_in, w_in = act_shape
            return n * c * h_in * w_in

    elif kernel_type == "Gemm":
        if act_shape and weight_shape and len(act_shape) >= 2 and len(weight_shape) >= 2:
            m, k = act_shape[0], act_shape[1]
            n_out = weight_shape[0] if attrs.get("transB", 0) else weight_shape[1]
            return 2 * m * k * n_out

    return 0


# ──────────────────────────────────────────────
# 3. 单模型解析
# ──────────────────────────────────────────────

def _parse_model_filename(filename):
    """从文件名解析 batch_size 和 input_size"""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    bs, input_size = None, None
    for p in parts:
        if p.startswith("bs"):
            try:
                bs = int(p[2:])
            except ValueError:
                pass
        if "x" in p:
            segs = p.split("x")
            if len(segs) == 2 and segs[0].isdigit():
                input_size = int(segs[0])
    return bs, input_size


def process_single_model(onnx_path):
    """
    处理单个 ONNX 模型，返回每个节点的内核信息列表。
    """
    model = get_optimized_model(onnx_path)

    # 构建 shape_map
    shape_map = _build_initial_shape_map(model)
    for node in model.graph.node:
        _infer_node_output_shape(node, shape_map)

    init_names = set(init.name for init in model.graph.initializer)
    bs, input_size = _parse_model_filename(onnx_path)
    source_name = os.path.splitext(os.path.basename(onnx_path))[0]

    kernels = []
    for node in model.graph.node:
        kernel_type = detect_kernel_type(node)
        attrs = _extract_attributes(node)

        # 分离 activation / weight / bias / residual
        act_shape = None
        weight_shape = None
        bias_shape = None
        residual_shape = None

        for idx, inp_name in enumerate(node.input):
            shape = shape_map.get(inp_name)
            if inp_name in init_names:
                if shape and len(shape) >= 2:
                    weight_shape = shape
                elif shape and len(shape) == 1:
                    bias_shape = shape
            else:
                if act_shape is None:
                    act_shape = shape          # 第一个非权重输入 = 主 activation
                elif node.op_type == "FusedConv" and idx >= 3:
                    residual_shape = shape    # 第四个输入 = residual

        out_shape = shape_map.get(node.output[0]) if node.output else None
        flops = _estimate_flops(kernel_type, attrs, act_shape, weight_shape, out_shape)

        # 构建 shape 记录
        shape_record = {
            "source_model": source_name,
            "batch_size": bs,
            "input_size": input_size,
            "activation_input_shape": act_shape,
            "output_shape": out_shape,
            "flops": flops,
        }
        if weight_shape:
            shape_record["weight_shape"] = weight_shape
        if bias_shape:
            shape_record["bias_shape"] = bias_shape
        if residual_shape:
            shape_record["residual_shape"] = residual_shape

        kernels.append({
            "kernel_type": kernel_type,
            "node_name": node.name,
            "attrs": attrs,
            "shape_record": shape_record,
        })

    return kernels


# ──────────────────────────────────────────────
# 4. 多模型聚合
# ──────────────────────────────────────────────

def _aggregate_kernels(all_model_kernels):
    """
    将多个模型变体的 kernel 信息按 node_name 聚合。
    同一 node_name 的不同 batch_size/input_size 变体放在 shapes 列表中。
    """
    kernels_by_node = {}

    for model_kernels in all_model_kernels:
        for k in model_kernels:
            node_name = k["node_name"]
            if node_name not in kernels_by_node:
                kernels_by_node[node_name] = {
                    "kernel_type": k["kernel_type"],
                    "node_name": node_name,
                    "fusion_desc": _fusion_description(k["kernel_type"]),
                    "attributes": k["attrs"],
                    "shapes": [],
                }
            kernels_by_node[node_name]["shapes"].append(k["shape_record"])

    # 保持第一个模型的节点顺序
    if all_model_kernels:
        ordered_names = [k["node_name"] for k in all_model_kernels[0]]
        return [kernels_by_node[n] for n in ordered_names if n in kernels_by_node]

    return list(kernels_by_node.values())


# ──────────────────────────────────────────────
# 5. 主入口
# ──────────────────────────────────────────────

def process_model_zoo(model_dir, model_prefix="resnet18"):
    """
    处理 model_zoo 中指定前缀的所有模型变体，
    聚合内核信息并保存 JSON 到 ort_kernel_record/。

    Returns:
        保存的 JSON 文件路径
    """
    pattern = os.path.join(model_dir, "{}*.onnx".format(model_prefix))
    model_files = sorted(glob.glob(pattern))

    if not model_files:
        logger.error("未找到匹配的模型文件: %s", pattern)
        return None

    logger.info("找到 %d 个模型文件", len(model_files))

    all_model_kernels = []
    for mf in model_files:
        logger.info("处理模型: %s", os.path.basename(mf))
        kernels = process_single_model(mf)
        all_model_kernels.append(kernels)

    aggregated = _aggregate_kernels(all_model_kernels)

    kernel_types = sorted(set(k["kernel_type"] for k in aggregated))
    logger.info("检测到 %d 个内核节点, %d 种内核类型: %s",
                len(aggregated), len(kernel_types), kernel_types)

    # 保存 JSON
    os.makedirs(RECORD_DIR, exist_ok=True)
    output = {
        "model_name": model_prefix,
        "total_kernels": len(aggregated),
        "kernel_types": kernel_types,
        "kernels": aggregated,
    }

    output_path = os.path.join(RECORD_DIR, "{}.json".format(model_prefix))
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("内核记录已保存: %s", output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Fusion Detector & Kernel Recorder")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
                        help="模型目录 (default: model_zoo/models)")
    parser.add_argument("--prefix", type=str, default="resnet18",
                        help="模型名前缀 (default: resnet18)")
    args = parser.parse_args()

    process_model_zoo(args.model_dir, args.prefix)


if __name__ == "__main__":
    main()
