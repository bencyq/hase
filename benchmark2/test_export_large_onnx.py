#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 for_graph 场景下导出大 batch / 大输入的 ORT 优化 ONNX 模型。

目标模型:
- resnet152
- densenet201
- vgg19
- yolov8m
"""

import os
import sys
from typing import Iterable, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as tv_models

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_zoo.torchvision_exporter import _export_onnx, _optimize_onnx
from utils.logger import get_logger

try:
    from ultralytics import YOLO
    YOLO_IMPORT_ERROR = None
except Exception as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc


logger = get_logger("for_graph_export_large_onnx")
OUTPUT_DIR = "benchmark2"
COMBINATIONS: Tuple[Tuple[int, int], ...] = (
    (64, 320),
    (128, 512),
)
MODEL_COMBINATIONS = {
    # 旧版 PyTorch ONNX 导出下，VGG19 的 adaptive_avg_pool2d 对 320/512 不稳定。
    # 使用 448 (14x14 -> 7x7) 可规避该限制，同时保持大输入。
    "vgg19": ((64, 448), (128, 448)),
}


def _validate_onnx_generic(onnx_path: str, batch_size: int, input_size: int) -> None:
    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)
    _ = session.run(None, {input_name: dummy})


def export_torchvision_models() -> None:
    # model_registry = {
    #     "resnet152": tv_models.resnet152,
    #     "densenet201": tv_models.densenet201,
    #     "vgg19": tv_models.vgg19,
    # }
    model_registry = {
        "vgg19": tv_models.vgg19,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for model_name, model_builder in model_registry.items():
        model = model_builder(weights=None).eval()
        combinations = MODEL_COMBINATIONS.get(model_name, COMBINATIONS)
        for bs, size in combinations:
            name = f"{model_name}_bs{bs}_{size}x{size}"
            raw_path = os.path.join(OUTPUT_DIR, name + "_raw.onnx")
            opt_path = os.path.join(OUTPUT_DIR, name + ".onnx")
            try:
                _export_onnx(model, bs, size, raw_path)
                _optimize_onnx(raw_path, opt_path)
                _validate_onnx_generic(opt_path, bs, size)
                logger.info("导出并验证成功: %s", opt_path)
            except Exception as exc:
                logger.warning("导出失败，跳过 %s: %s", name, exc)
            finally:
                if os.path.exists(raw_path):
                    os.remove(raw_path)


def export_yolov8m(combinations: Iterable[Tuple[int, int]]) -> None:
    if YOLO is None:
        logger.warning("ultralytics 导入失败，跳过 yolov8m 导出: %s", YOLO_IMPORT_ERROR)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yolo = YOLO("yolov8m.pt")
    for bs, size in combinations:
        name = f"yolov8m_bs{bs}_{size}x{size}"
        raw_path = os.path.join(OUTPUT_DIR, name + "_raw.onnx")
        opt_path = os.path.join(OUTPUT_DIR, name + ".onnx")
        try:
            exported = yolo.export(
                format="onnx",
                imgsz=size,
                batch=bs,
                opset=13,
                simplify=False,
                dynamic=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )
            if isinstance(exported, str) and os.path.exists(exported):
                os.replace(exported, raw_path)
            elif os.path.exists(raw_path):
                pass
            else:
                raise RuntimeError("YOLO 导出后未找到原始 ONNX 文件")

            _optimize_onnx(raw_path, opt_path)
            _validate_onnx_generic(opt_path, bs, size)
            logger.info("导出并验证成功: %s", opt_path)
        except Exception as exc:
            logger.warning("导出失败，跳过 %s: %s", name, exc)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)


def main() -> None:
    export_torchvision_models()
    export_yolov8m(COMBINATIONS)
    logger.info("for_graph 大模型导出任务完成")


if __name__ == "__main__":
    main()
