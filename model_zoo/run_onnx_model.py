#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import time

import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行 model_zoo 目录下的 ONNX 模型，并按输入签名自动生成随机输入。"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="模型名，不带 .onnx，例如 mobilenet_v2_bs32_224x224",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_DIR,
        help="模型文件路径，或模型目录路径；默认指向 model_zoo/models",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="运行轮数，测试时统一可传 20",
    )
    return parser.parse_args()


def resolve_model_path(model_name, model_path):
    if os.path.isdir(model_path):
        resolved = os.path.join(model_path, f"{model_name}.onnx")
    else:
        resolved = model_path
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"ONNX 模型不存在: {resolved}")
    return resolved


def parse_model_name_hints(model_name):
    batch_size = None
    height = None
    width = None

    batch_match = re.search(r"_bs(\d+)", model_name)
    if batch_match:
        batch_size = int(batch_match.group(1))

    spatial_match = re.search(r"_(\d+)x(\d+)(?:_|$)", model_name)
    if spatial_match:
        height = int(spatial_match.group(1))
        width = int(spatial_match.group(2))

    return {
        "batch_size": batch_size,
        "height": height,
        "width": width,
    }


def to_numpy_dtype(ort_type):
    ort_type = ort_type.lower()
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
    if "int16" in ort_type:
        return np.int16
    if "int8" in ort_type:
        return np.int8
    if "uint64" in ort_type:
        return np.uint64
    if "uint32" in ort_type:
        return np.uint32
    if "uint16" in ort_type:
        return np.uint16
    if "uint8" in ort_type:
        return np.uint8
    if "bool" in ort_type:
        return np.bool_
    return np.float32


def infer_dim(dim, axis, rank, hints):
    if isinstance(dim, int) and dim > 0:
        return dim

    dim_name = "" if dim is None else str(dim).lower()

    if axis == 0 and hints["batch_size"] is not None:
        return hints["batch_size"]

    if rank >= 4 and axis == rank - 2 and hints["height"] is not None:
        return hints["height"]

    if rank >= 4 and axis == rank - 1 and hints["width"] is not None:
        return hints["width"]

    if "batch" in dim_name and hints["batch_size"] is not None:
        return hints["batch_size"]
    if ("height" in dim_name or dim_name == "h") and hints["height"] is not None:
        return hints["height"]
    if ("width" in dim_name or dim_name == "w") and hints["width"] is not None:
        return hints["width"]
    if "seq" in dim_name or "length" in dim_name:
        return 16
    if "channel" in dim_name or dim_name == "c":
        return 3

    return 1


def normalize_shape(shape, hints):
    rank = len(shape)
    return [infer_dim(dim, axis, rank, hints) for axis, dim in enumerate(shape)]


def build_random_array(shape, dtype):
    if np.issubdtype(dtype, np.bool_):
        return np.random.randint(0, 2, size=shape).astype(dtype)
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 8, size=shape).astype(dtype)
    return np.random.randn(*shape).astype(dtype)


def build_feed(session, model_name):
    hints = parse_model_name_hints(model_name)
    feed = {}
    input_summaries = []

    for inp in session.get_inputs():
        shape = normalize_shape(inp.shape, hints)
        dtype = to_numpy_dtype(inp.type)
        feed[inp.name] = build_random_array(shape, dtype)
        input_summaries.append(
            {
                "name": inp.name,
                "ort_type": inp.type,
                "numpy_dtype": np.dtype(dtype).name,
                "shape": shape,
            }
        )

    return feed, input_summaries


def select_providers():
    available = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    # if "CPUExecutionProvider" in available:
    #     providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError("未检测到可用的 onnxruntime provider")
    return providers


def create_session(model_file, providers):
    last_error = None
    tried = []

    for provider in providers:
        tried.append(provider)
        try:
            return ort.InferenceSession(model_file, providers=[provider]), [provider]
        except Exception as exc:
            last_error = exc
            print(f"provider {provider} 初始化失败，尝试下一个 provider: {exc}")

    raise RuntimeError(
        "无法创建 onnxruntime session，已尝试 providers={}: {}".format(tried, last_error)
    )


def main():
    args = parse_args()
    model_file = resolve_model_path(args.model_name, args.model_path)
    providers = select_providers()
    session, active_providers = create_session(model_file, providers)
    feed, input_summaries = build_feed(session, args.model_name)

    print(f"model_name: {args.model_name}")
    print(f"model_file: {model_file}")
    print(f"providers: {active_providers}")
    print(f"epoch: {args.epoch}")
    print("inputs:")
    for item in input_summaries:
        print(
            "  - name={name}, ort_type={ort_type}, numpy_dtype={numpy_dtype}, shape={shape}".format(
                **item
            )
        )

    start = time.perf_counter()
    for _ in range(args.epoch):
        session.run(None, feed)
    elapsed = time.perf_counter() - start

    print(f"total_time_s: {elapsed:.6f}")
    print(f"avg_time_ms: {elapsed * 1000.0 / args.epoch:.3f}")


if __name__ == "__main__":
    main()
