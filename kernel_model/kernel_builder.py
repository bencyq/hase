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
    if not bias_shape or len(bias_shape) != 1:
        bias_shape = [weight_shape[0]]
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


def _build_averagepool_model(attrs, act_shape, out_shape):
    """构建 AveragePool 独立 ONNX 模型"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)
    node = helper.make_node(
        "AveragePool", inputs=["X"], outputs=["Y"],
        kernel_shape=attrs.get("kernel_shape", [2, 2]),
        strides=attrs.get("strides", attrs.get("kernel_shape", [2, 2])),
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


def _build_unary_model(op_type, attrs, act_shape, out_shape):
    """构建一元算子模型，默认输出与输入同 shape"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    node_kwargs = {}
    if op_type == "Softmax":
        node_kwargs["axis"] = attrs.get("axis", -1)
    node = helper.make_node(op_type, inputs=["X"], outputs=["Y"], **node_kwargs)
    graph = helper.make_graph([node], "kernel", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_batchnorm_model(act_shape, out_shape):
    """构建 BatchNormalization 独立 ONNX 模型"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    c = act_shape[1] if len(act_shape) > 1 else 1
    scale = numpy_helper.from_array(np.ones((c,), dtype=np.float32), name="scale")
    bias = numpy_helper.from_array(np.zeros((c,), dtype=np.float32), name="bias")
    mean = numpy_helper.from_array(np.zeros((c,), dtype=np.float32), name="mean")
    var = numpy_helper.from_array(np.ones((c,), dtype=np.float32), name="var")
    node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "scale", "bias", "mean", "var"],
        outputs=["Y"],
        epsilon=1e-5,
        momentum=0.9,
    )
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=[scale, bias, mean, var])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_quickgelu_model(act_shape, out_shape):
    """用标准 ONNX 算子近似 QuickGelu: x * sigmoid(1.702*x)"""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    alpha = numpy_helper.from_array(np.array([1.702], dtype=np.float32), name="alpha")
    nodes = [
        helper.make_node("Mul", inputs=["X", "alpha"], outputs=["scaled"]),
        helper.make_node("Sigmoid", inputs=["scaled"], outputs=["sig"]),
        helper.make_node("Mul", inputs=["X", "sig"], outputs=["Y"]),
    ]
    graph = helper.make_graph(nodes, "kernel", [X], [Y], initializer=[alpha])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_binary_model(op_type, act_shape, out_shape, rhs_shape=None):
    """构建二元逐元素算子模型"""
    if rhs_shape is None:
        rhs_shape = act_shape

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    R_data = np.random.randn(*rhs_shape).astype(np.float32)
    R = numpy_helper.from_array(R_data, name="R")
    node = helper.make_node(op_type, inputs=["X", "R"], outputs=["Y"])
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=[R])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_concat_model(attrs, act_shape, out_shape):
    axis = attrs.get("axis", 1 if len(act_shape) > 1 else 0)
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, act_shape)
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    node = helper.make_node("Concat", inputs=["A", "B"], outputs=["Y"], axis=axis)
    graph = helper.make_graph([node], "kernel", [A, B], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_split_model(attrs, act_shape, out_shape):
    axis = attrs.get("axis", 0)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    node = helper.make_node("Split", inputs=["X"], outputs=["Y"], axis=axis)
    graph = helper.make_graph([node], "kernel", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_reshape_model(act_shape, out_shape):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    target = np.array((out_shape or act_shape), dtype=np.int64)
    shape_init = numpy_helper.from_array(target, name="shape")
    node = helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"])
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=[shape_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_transpose_model(attrs, act_shape, out_shape):
    perm = attrs.get("perm")
    if not perm and out_shape and len(out_shape) == len(act_shape):
        perm = []
        used = set()
        for d in out_shape:
            idx = next((i for i, v in enumerate(act_shape) if v == d and i not in used), None)
            if idx is None:
                perm = None
                break
            used.add(idx)
            perm.append(idx)
    if not perm:
        perm = list(range(len(act_shape) - 1, -1, -1))

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    node = helper.make_node("Transpose", inputs=["X"], outputs=["Y"], perm=perm)
    graph = helper.make_graph([node], "kernel", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_pad_model(act_shape, out_shape):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    rank = len(act_shape)
    if out_shape and len(out_shape) == rank:
        pads = []
        tail = []
        for i in range(rank):
            diff = max(0, out_shape[i] - act_shape[i])
            p0 = diff // 2
            p1 = diff - p0
            pads.append(p0)
            tail.append(p1)
        pads = np.array(pads + tail, dtype=np.int64)
    else:
        pads = np.zeros(rank * 2, dtype=np.int64)
    pads_init = numpy_helper.from_array(pads, name="pads")
    node = helper.make_node("Pad", inputs=["X", "pads"], outputs=["Y"], mode="constant")
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=[pads_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_slice_model(act_shape, out_shape):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)
    rank = len(act_shape)
    starts = np.zeros(rank, dtype=np.int64)
    ends = np.array(act_shape, dtype=np.int64)
    axes = np.arange(rank, dtype=np.int64)
    steps = np.ones(rank, dtype=np.int64)

    if out_shape and len(out_shape) == rank:
        for i in range(rank):
            ends[i] = min(act_shape[i], out_shape[i])

    inits = [
        numpy_helper.from_array(starts, name="starts"),
        numpy_helper.from_array(ends, name="ends"),
        numpy_helper.from_array(axes, name="axes"),
        numpy_helper.from_array(steps, name="steps"),
    ]
    node = helper.make_node("Slice", inputs=["X", "starts", "ends", "axes", "steps"], outputs=["Y"])
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    return model


def _build_resize_model(act_shape, out_shape):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, act_shape)
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape or act_shape)

    rank = len(act_shape)
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    if out_shape and len(out_shape) == rank:
        scales = [float(o) / float(a) if a else 1.0 for a, o in zip(act_shape, out_shape)]
    else:
        scales = [1.0] * rank
    scales_init = numpy_helper.from_array(np.array(scales, dtype=np.float32), name="scales")

    node = helper.make_node(
        "Resize",
        inputs=["X", "roi", "scales"],
        outputs=["Y"],
        mode="nearest",
    )
    graph = helper.make_graph([node], "kernel", [X], [Y], initializer=[roi, scales_init])
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
    act = shape_info.get("activation_input_shape")
    out = shape_info.get("output_shape")
    ws = shape_info.get("weight_shape")
    bs = shape_info.get("bias_shape")
    rs = shape_info.get("residual_shape")

    batch = shape_info.get("batch_size", 1)

    if act is None and out is not None:
        act = out
    if out is None and act is not None:
        out = act
    if act is None and out is None:
        if kt == "Gemm" and ws and len(ws) >= 2:
            trans_b = attrs.get("transB", 0)
            in_dim = ws[1] if trans_b else ws[0]
            out_dim = ws[0] if trans_b else ws[1]
            act = [batch, in_dim]
            out = [batch, out_dim]
        elif kt in ("Conv", "Conv_Relu", "Conv_Add_Relu") and ws and len(ws) == 4:
            act = [batch, ws[1], 1, 1]
            out = [batch, ws[0], 1, 1]
        elif kt == "Flatten":
            act = [batch, 1, 1, 1]
            out = [batch, 1]
        else:
            act = [batch, 1]
            out = [batch, 1]

    if kt == "Conv_Relu":
        return _build_conv_model(attrs, act, ws, bs, out, add_relu=True)
    elif kt == "Conv_Add_Relu":
        return _build_conv_model(attrs, act, ws, bs, out,
                                 add_relu=True, add_residual=True, residual_shape=rs)
    elif kt == "Conv":
        return _build_conv_model(attrs, act, ws, bs, out)
    elif kt == "MaxPool":
        return _build_maxpool_model(attrs, act, out)
    elif kt == "AveragePool":
        return _build_averagepool_model(attrs, act, out)
    elif kt == "GlobalAveragePool":
        return _build_gap_model(act, out)
    elif kt == "Flatten":
        return _build_flatten_model(attrs, act, out)
    elif kt == "Gemm":
        return _build_gemm_model(attrs, act, ws, bs, out)
    elif kt == "QuickGelu":
        return _build_quickgelu_model(act, out)
    elif kt == "BatchNormalization":
        return _build_batchnorm_model(act, out)
    elif kt in ("Relu", "Clip", "Sigmoid", "HardSigmoid", "Softmax"):
        return _build_unary_model(kt, attrs, act, out)
    elif kt in ("Add", "Mul", "Div", "Sub"):
        return _build_binary_model(kt, act, out, ws or rs)
    elif kt == "Concat":
        return _build_concat_model(attrs, act, out)
    elif kt == "Split":
        return _build_split_model(attrs, act, out)
    elif kt == "Reshape":
        return _build_reshape_model(act, out)
    elif kt == "Transpose":
        return _build_transpose_model(attrs, act, out)
    elif kt == "Pad":
        return _build_pad_model(act, out)
    elif kt == "Slice":
        return _build_slice_model(act, out)
    elif kt == "Resize":
        return _build_resize_model(act, out)
    else:
        raise ValueError("不支持的 kernel_type: {}".format(kt))


def make_kernel_filename(kernel_info, shape_info):
    """
    生成内核 ONNX 文件名（不含 .onnx 后缀）。
    格式: {kernel_type}_bs{N}_{C}c{H}x{W}[_k{kH}x{kW}]
    """
    kt = kernel_info["kernel_type"]
    act = shape_info.get("activation_input_shape") or shape_info.get("output_shape")
    if not act:
        n = shape_info.get("batch_size", "u")
        return "{}_bs{}_unknown".format(kt, n)
    n = act[0] if len(act) > 0 else shape_info.get("batch_size", "u")

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

def kernel_build(json_path, output_dir=OUTPUT_DIR, skip_existing=True):
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
    skipped_existing = 0
    skipped_invalid = 0
    paths = []

    for kernel in data["kernels"]:
        for shape in kernel["shapes"]:
            fname = make_kernel_filename(kernel, shape)
            if fname in generated:
                continue  # 同样的计算模式已生成，跳过

            onnx_path = os.path.join(output_dir, fname + ".onnx")
            if skip_existing and os.path.exists(onnx_path):
                skipped_existing += 1
                generated.add(fname)
                logger.info("已存在，跳过: %s", fname)
                continue

            # 构建 & 保存
            try:
                model = build_kernel_model(kernel, shape)
            except ValueError as e:
                skipped_invalid += 1
                logger.warning("构建失败，跳过: %s (%s)", fname, e)
                continue
            onnx.save(model, onnx_path)

            # 验证
            validate_kernel(onnx_path)

            generated.add(fname)
            paths.append(onnx_path)
            logger.info("生成并验证: %s", fname)

    logger.info("共生成 %d 个 kernel ONNX 文件，跳过已有 %d 个，跳过无效 %d 个，保存于: %s",
                len(paths), skipped_existing, skipped_invalid, output_dir)
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
