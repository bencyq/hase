# -*- coding: utf-8 -*-
"""
ORT Graph Parser
加载 ONNX 模型，使用 ORT 获取优化后的图，
遍历所有节点提取 OpType、Input Shapes、Output Shapes。
"""
import os
import sys
import math
import argparse
import tempfile

import onnx
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger("ort_graph_parser")

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ──────────────────────────────────────────────
# 1. 获取 ORT 优化后模型
# ──────────────────────────────────────────────

def get_optimized_model(onnx_path):
    """
    使用 ORT InferenceSession (ORT_ENABLE_ALL) 优化模型并保存到临时文件，
    然后用 onnx 库加载返回 ModelProto。

    Args:
        onnx_path: 输入 ONNX 文件路径

    Returns:
        onnx.ModelProto
    """
    fd, tmp_path = tempfile.mkstemp(suffix="_opt.onnx")
    os.close(fd)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = tmp_path

    _ = ort.InferenceSession(onnx_path, sess_options, providers=PROVIDERS)
    logger.info("ORT 优化后模型已保存: %s", tmp_path)

    model = onnx.load(tmp_path)
    os.remove(tmp_path)
    return model


# ──────────────────────────────────────────────
# 2. Shape 辅助：从 graph inputs / initializers 获取已知 shape，
#    并通过 forward propagation 推导中间张量的 shape
# ──────────────────────────────────────────────

def _get_attr(node, name, default=None):
    """从 node.attribute 中按名称取值"""
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == 7:       # INTS
                return list(attr.ints)
            elif attr.type == 2:     # INT
                return attr.i
            elif attr.type == 3:     # STRING
                return attr.s.decode()
            elif attr.type == 1:     # FLOAT
                return attr.f
    return default


def _conv_output_hw(h_in, w_in, kernel_shape, strides, pads, dilations):
    """计算 Conv / MaxPool 的输出 H, W"""
    kh, kw = kernel_shape
    sh, sw = strides
    dh, dw = dilations if dilations else [1, 1]
    pad_top, pad_left, pad_bottom, pad_right = pads

    h_out = math.floor((h_in + pad_top + pad_bottom - dh * (kh - 1) - 1) / sh + 1)
    w_out = math.floor((w_in + pad_left + pad_right - dw * (kw - 1) - 1) / sw + 1)
    return h_out, w_out


def _build_initial_shape_map(model):
    """从 graph inputs、initializers、value_info 中收集已知 shape"""
    shape_map = {}

    # graph inputs
    for inp in model.graph.input:
        if inp.type.tensor_type.HasField("shape"):
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                dims.append(d.dim_param if d.dim_param else d.dim_value)
            shape_map[inp.name] = dims

    # graph outputs
    for out in model.graph.output:
        if out.type.tensor_type.HasField("shape"):
            dims = []
            for d in out.type.tensor_type.shape.dim:
                dims.append(d.dim_param if d.dim_param else d.dim_value)
            shape_map[out.name] = dims

    # value_info
    for vi in model.graph.value_info:
        if vi.type.tensor_type.HasField("shape"):
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_param if d.dim_param else d.dim_value)
            shape_map[vi.name] = dims

    # initializers (权重)
    for init in model.graph.initializer:
        shape_map[init.name] = list(init.dims)

    return shape_map


def _infer_node_output_shape(node, shape_map):
    """
    根据算子类型和已知输入 shape，推导节点输出 shape 并写入 shape_map。
    支持: Conv, FusedConv, MaxPool, GlobalAveragePool, Flatten, Gemm, Relu, Add, MatMul
    """
    op = node.op_type

    if op in ("Conv", "FusedConv"):
        # 输入: X, W, [B], [Z (residual for FusedConv)]
        x_shape = shape_map.get(node.input[0])
        w_shape = shape_map.get(node.input[1])
        if x_shape and w_shape and len(x_shape) == 4:
            n, _, h_in, w_in = x_shape
            out_c = w_shape[0]
            ks = _get_attr(node, "kernel_shape", [w_shape[2], w_shape[3]])
            strides = _get_attr(node, "strides", [1, 1])
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
            dilations = _get_attr(node, "dilations", [1, 1])
            h_out, w_out = _conv_output_hw(h_in, w_in, ks, strides, pads, dilations)
            out_shape = [n, out_c, h_out, w_out]
            for o in node.output:
                shape_map[o] = out_shape

    elif op == "MaxPool":
        x_shape = shape_map.get(node.input[0])
        if x_shape and len(x_shape) == 4:
            n, c, h_in, w_in = x_shape
            ks = _get_attr(node, "kernel_shape")
            strides = _get_attr(node, "strides", ks)
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
            h_out, w_out = _conv_output_hw(h_in, w_in, ks, strides, pads, [1, 1])
            shape_map[node.output[0]] = [n, c, h_out, w_out]

    elif op == "GlobalAveragePool":
        x_shape = shape_map.get(node.input[0])
        if x_shape and len(x_shape) == 4:
            shape_map[node.output[0]] = [x_shape[0], x_shape[1], 1, 1]

    elif op == "Flatten":
        x_shape = shape_map.get(node.input[0])
        axis = _get_attr(node, "axis", 1)
        if x_shape:
            left = 1
            for d in x_shape[:axis]:
                left *= d
            right = 1
            for d in x_shape[axis:]:
                right *= d
            shape_map[node.output[0]] = [left, right]

    elif op == "Gemm":
        # Y = alpha * A * B + beta * C
        a_shape = shape_map.get(node.input[0])
        b_shape = shape_map.get(node.input[1])
        trans_a = _get_attr(node, "transA", 0)
        trans_b = _get_attr(node, "transB", 0)
        if a_shape and b_shape:
            m = a_shape[0] if not trans_a else a_shape[1]
            k_b = b_shape[1] if not trans_b else b_shape[0]
            shape_map[node.output[0]] = [m, k_b]

    elif op in ("Relu", "BatchNormalization", "Dropout", "Identity"):
        x_shape = shape_map.get(node.input[0])
        if x_shape:
            for o in node.output:
                shape_map[o] = list(x_shape)

    elif op == "Add":
        a_shape = shape_map.get(node.input[0])
        b_shape = shape_map.get(node.input[1])
        # element-wise add 输出与较长的输入 shape 相同
        if a_shape:
            shape_map[node.output[0]] = list(a_shape)
        elif b_shape:
            shape_map[node.output[0]] = list(b_shape)

    elif op == "MatMul":
        a_shape = shape_map.get(node.input[0])
        b_shape = shape_map.get(node.input[1])
        if a_shape and b_shape:
            shape_map[node.output[0]] = list(a_shape[:-1]) + [b_shape[-1]]

    elif op == "Reshape":
        # 'shape' 输入一般是常量 initializer
        shape_val = shape_map.get(node.input[1]) if len(node.input) > 1 else None
        # 这里无法完整推断，留 None
        pass

    # 未知算子: shape_map 中不会有该输出，后续会显示 None


# ──────────────────────────────────────────────
# 3. 主解析流程
# ──────────────────────────────────────────────

def parse_graph(model):
    """
    解析模型中所有节点的 OpType、Input Shapes、Output Shapes。

    Args:
        model: onnx.ModelProto

    Returns:
        list[dict]: 节点信息列表
    """
    shape_map = _build_initial_shape_map(model)

    # Forward propagation: 按图中节点顺序依次推导
    for node in model.graph.node:
        _infer_node_output_shape(node, shape_map)

    nodes_info = []
    for node in model.graph.node:
        input_shapes = []
        for name in node.input:
            input_shapes.append({
                "name": name,
                "shape": shape_map.get(name),
            })

        output_shapes = []
        for name in node.output:
            output_shapes.append({
                "name": name,
                "shape": shape_map.get(name),
            })

        nodes_info.append({
            "op_type": node.op_type,
            "name": node.name,
            "inputs": input_shapes,
            "outputs": output_shapes,
        })

    return nodes_info


def print_graph_summary(nodes_info):
    """打印图节点摘要"""
    logger.info("共 %d 个节点", len(nodes_info))
    logger.info("-" * 90)
    for i, n in enumerate(nodes_info):
        # 只显示非 initializer 的输入 shape（即 activation）
        act_in = [inp for inp in n["inputs"]
                  if inp["shape"] is not None and len(inp["shape"]) >= 2]
        in_str = ", ".join(str(inp["shape"]) for inp in act_in) if act_in else "N/A"
        out_str = ", ".join(str(o["shape"]) for o in n["outputs"]
                           if o["shape"] is not None) or "N/A"
        logger.info("[%2d] %-25s  输入: %-30s  输出: %s", i, n["op_type"], in_str, out_str)


def parse_onnx_model(onnx_path):
    """
    解析给定 ONNX 模型的完整流程。

    Args:
        onnx_path: ONNX 模型路径

    Returns:
        list[dict]: 节点信息列表
    """
    logger.info("解析模型: %s", onnx_path)
    model = get_optimized_model(onnx_path)
    nodes_info = parse_graph(model)
    print_graph_summary(nodes_info)
    return nodes_info


# ──────────────────────────────────────────────
# 4. CLI 入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ORT Graph Parser")
    parser.add_argument("--model", type=str, required=True,
                        help="ONNX 模型文件路径")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        logger.error("模型文件不存在: %s", args.model)
        sys.exit(1)

    parse_onnx_model(args.model)


if __name__ == "__main__":
    main()
