# -*- coding: utf-8 -*-
"""
Single Kernel Extractor
读取 ort_kernel_record/ 下的 kernel JSON，
为每个内核构建独立可运行的 ONNX 模型，保存到 kernel_model/kernel_onnx/。
"""
import os
import sys
import json
import argparse

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger

logger = get_logger("kernel_builder")

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "kernel_onnx")
RECORD_DIR = os.path.join(PROJECT_ROOT, "ort_analysis", "ort_kernel_record")


# ──────────────────────────────────────────────
# 1. ONNX 模型构建器 (按 kernel_type)
# ──────────────────────────────────────────────

def _build_conv_model(attrs, act_shape, weight_shape, bias_shape, out_shape,
                      add_relu=False, add_residual=False, residual_shape=None):
    """构建 Conv [+ Add] [+ Relu] 的独立 ONNX 模型"""
    # 图输入: 只有 activation (和可选 residual)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    graph_inputs = [X]

    if add_residual and residual_shape:
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, residual_shape)
        graph_inputs.append(Z)

    # 初始化器: 权重 & 偏置 (随机数据)
    W_data = np.random.randn(*weight_shape).astype(np.float32) * 0.01
    B_data = np.zeros(bias_shape, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(W_data, name="W"),
        numpy_helper.from_array(B_data, name="B"),
    ]

    # 构建节点链
    nodes = []
    last_out = "conv_out" if (add_relu or add_residual) else "Y"

    nodes.append(helper.make_node(
        "Conv", inputs=["X", "W", "B"], outputs=[last_out],
        kernel_shape=attrs["kernel_shape"],
        strides=attrs.get("strides", [1, 1]),
        pads=attrs.get("pads", [0, 0, 0, 0]),
        dilations=attrs.get("dilations", [1, 1]),
        group=attrs.get("group", 1),
    ))

    if add_residual:
        next_out = "add_out" if add_relu else "Y"
        nodes.append(helper.make_node("Add", inputs=[last_out, "Z"], outputs=[next_out]))
        last_out = next_out

    if add_relu:
        nodes.append(helper.make_node("Relu", inputs=[last_out], outputs=["Y"]))

    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
    graph = helper.make_graph(nodes, "kernel", graph_inputs, [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_maxpool_model(attrs, act_shape, out_shape):
    """构建 MaxPool 独立 ONNX 模型"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
    node = helper.make_node(
        "MaxPool", inputs=["X"], outputs=["Y"],
        kernel_shape=attrs["kernel_shape"],
        strides=attrs.get("strides", attrs["kernel_shape"]),
        pads=attrs.get("pads", [0, 0, 0, 0]),
        ceil_mode=attrs.get("ceil_mode", 0),
    )
    graph = helper.make_graph([node], "kernel", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_gap_model(act_shape, out_shape):
    """构建 GlobalAveragePool 独立 ONNX 模型"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
    node = helper.make_node("GlobalAveragePool", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "kernel", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_flatten_model(attrs, act_shape, out_shape):
    """构建 Flatten 独立 ONNX 模型"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
    node = helper.make_node("Flatten", inputs=["X"], outputs=["Y"],
                            axis=attrs.get("axis", 1))
    graph = helper.make_graph([node], "kernel", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_gemm_model(attrs, act_shape, weight_shape, bias_shape, out_shape):
    """构建 Gemm 独立 ONNX 模型"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)

    W_data = np.random.randn(*weight_shape).astype(np.float32) * 0.01
    B_data = np.zeros(bias_shape, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(W_data, name="W"),
        numpy_helper.from_array(B_data, name="B"),
    ]

    node = helper.make_node(
        "Gemm", inputs=["X", "W", "B"], outputs=["Y"],
        transA=attrs.get("transA", 0),
        transB=attrs.get("transB", 0),
    )
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


# ──────────────────────────────────────────────
# 2. 构建入口 & 文件命名
# ──────────────────────────────────────────────

def build_kernel_model(kernel_info, shape_info):
    """
    根据 kernel 元数据和 shape 信息构建独立 ONNX 模型。

    Args:
        kernel_info: kernel JSON 中的一个 kernel dict
        shape_info:  kernel["shapes"] 中的一个 shape 变体

    Returns:
        onnx.ModelProto
    """
    kt = kernel_info["kernel_type"]
    attrs = kernel_info["attributes"]
    act = shape_info["activation_input_shape"]
    out = shape_info["output_shape"]
    ws = shape_info.get("weight_shape")
    bs = shape_info.get("bias_shape")
    rs = shape_info.get("residual_shape")

    if kt == "Conv_Relu":
        return _build_conv_model(attrs, act, ws, bs, out, add_relu=True)
    elif kt == "Conv_Add_Relu":
        return _build_conv_model(attrs, act, ws, bs, out,
                                 add_relu=True, add_residual=True, residual_shape=rs)
    elif kt == "Conv":
        return _build_conv_model(attrs, act, ws, bs, out)
    elif kt == "MaxPool":
        return _build_maxpool_model(attrs, act, out)
    elif kt == "GlobalAveragePool":
        return _build_gap_model(act, out)
    elif kt == "Flatten":
        return _build_flatten_model(attrs, act, out)
    elif kt == "Gemm":
        return _build_gemm_model(attrs, act, ws, bs, out)
    else:
        raise ValueError("不支持的 kernel_type: {}".format(kt))


def make_kernel_filename(kernel_info, shape_info):
    """
    生成内核 ONNX 文件名（不含 .onnx 后缀）。
    格式: {kernel_type}_bs{N}_{C}c{H}x{W}[_k{kH}x{kW}]
    """
    kt = kernel_info["kernel_type"]
    act = shape_info["activation_input_shape"]
    n = act[0]

    if len(act) == 4:
        _, c, h, w = act
        base = "{}_bs{}_{:d}c{:d}x{:d}".format(kt, n, c, h, w)
        # 对 Conv 类加上 kernel_shape 区分
        ks = kernel_info["attributes"].get("kernel_shape")
        if ks:
            base += "_k{}x{}".format(ks[0], ks[1])
    elif len(act) == 2:
        base = "{}_bs{}_{:d}".format(kt, n, act[1])
    else:
        dims = "x".join(str(d) for d in act[1:])
        base = "{}_bs{}_{}".format(kt, n, dims)

    return base


# ──────────────────────────────────────────────
# 3. 验证
# ──────────────────────────────────────────────

def validate_kernel(onnx_path):
    """
    使用 ONNX Runtime 加载并运行 kernel 模型进行验证。

    Returns:
        True 表示验证通过
    """
    session = ort.InferenceSession(onnx_path, providers=PROVIDERS)
    feed = {}
    for inp in session.get_inputs():
        feed[inp.name] = np.random.randn(*inp.shape).astype(np.float32)
    outputs = session.run(None, feed)
    return True


# ──────────────────────────────────────────────
# 4. 主流程
# ──────────────────────────────────────────────

def kernel_build(json_path, output_dir=OUTPUT_DIR):
    """
    读取 kernel JSON 文件，为每个 (kernel_type, shape) 组合
    生成独立的 ONNX 模型文件。

    Args:
        json_path: ort_kernel_record/ 下的 JSON 文件路径
        output_dir: 输出目录 (default: kernel_model/kernel_onnx/)

    Returns:
        生成的 ONNX 文件路径列表
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    model_name = data["model_name"]
    logger.info("加载 kernel 记录: %s (%d 个内核节点)",
                model_name, data["total_kernels"])

    generated = set()
    paths = []

    for kernel in data["kernels"]:
        for shape in kernel["shapes"]:
            fname = make_kernel_filename(kernel, shape)
            if fname in generated:
                continue  # 同样的计算模式已生成，跳过

            onnx_path = os.path.join(output_dir, fname + ".onnx")

            # 构建 & 保存
            model = build_kernel_model(kernel, shape)
            onnx.save(model, onnx_path)

            # 验证
            validate_kernel(onnx_path)

            generated.add(fname)
            paths.append(onnx_path)
            logger.info("生成并验证: %s", fname)

    logger.info("共生成 %d 个 kernel ONNX 文件，保存于: %s", len(paths), output_dir)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Single Kernel Extractor")
    parser.add_argument("--json", type=str, default=None,
                        help="kernel JSON 文件路径 (默认: ort_kernel_record/resnet18.json)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="输出目录 (默认: kernel_model/kernel_onnx/)")
    args = parser.parse_args()

    json_path = args.json
    if json_path is None:
        json_path = os.path.join(RECORD_DIR, "resnet18.json")

    if not os.path.isfile(json_path):
        logger.error("JSON 文件不存在: %s", json_path)
        sys.exit(1)

    kernel_build(json_path, args.output_dir)


if __name__ == "__main__":
    main()
