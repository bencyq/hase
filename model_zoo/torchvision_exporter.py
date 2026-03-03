# -*- coding: utf-8 -*-
"""
Torchvision 模型导出器
从 torchvision 加载模型，根据 config.yaml 中的不同
batch_size 和 input_size 组合，导出为 ONNX Runtime 优化后的 ONNX 格式。

统一接口: export_model(model_name) — 按模型名称导出所有配置组合。
"""
import os
import sys
import yaml
import torch
import torchvision.models as models
import onnxruntime as ort
import numpy as np
import shutil

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger

try:
    from ultralytics import YOLO
    YOLO_IMPORT_ERROR = None
except Exception as exc:
    YOLO = None
    YOLO_IMPORT_ERROR = exc

logger = get_logger("torchvision_exporter")

# 路径常量
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")

# ── 模型注册表 ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "resnet18":  models.resnet18,
    "resnet50":  models.resnet50,
    "resnet152": models.resnet152,
    "vgg11":     models.vgg11,
    "vgg16":     models.vgg16,
    "vgg19":     models.vgg19,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "mobilenet_v2": models.mobilenet_v2,
    # config.yaml 中使用 mobilenet_v3 作为统一名称，这里映射到 torchvision 的 large 版本
    "mobilenet_v3": models.mobilenet_v3_large,
    "alexnet":   models.alexnet,
}

YOLO_MODELS = {"yolov8n", "yolov8m", "yolov8x"}


def load_config(config_path=CONFIG_PATH):
    """加载 config.yaml 配置"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info("加载配置: models=%s, batch_sizes=%s, input_sizes=%s",
                cfg.get("models", []), cfg["batch_sizes"], cfg["input_sizes"])
    return cfg


def _export_onnx(model, batch_size, input_size, raw_onnx_path):
    """将 PyTorch 模型导出为 ONNX 格式。"""
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        raw_onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )
    logger.info("导出原始 ONNX: %s", raw_onnx_path)


def _optimize_onnx(raw_onnx_path, optimized_onnx_path):
    """使用 ONNX Runtime 对模型进行图优化并保存。"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = optimized_onnx_path

    _ = ort.InferenceSession(
        raw_onnx_path,
        sess_options,
        providers=["CUDAExecutionProvider"],
    )
    logger.info("ORT 优化完成: %s", optimized_onnx_path)


def _validate_onnx(optimized_onnx_path, batch_size, input_size):
    """使用 ONNX Runtime 验证优化后的模型可正常推理。"""
    session = ort.InferenceSession(
        optimized_onnx_path,
        providers=["CUDAExecutionProvider"],
    )
    dummy = np.random.randn(batch_size, 3, input_size, input_size).astype(np.float32)
    outputs = session.run(None, {"input": dummy})
    logger.info("验证通过: %s  输出形状=%s", optimized_onnx_path, outputs[0].shape)
    return True


def _resolve_yolo_weight(model_name):
    """优先使用本地权重文件，否则交给 ultralytics 按名称处理。"""
    local_path = os.path.join(PROJECT_ROOT, "{}.pt".format(model_name))
    if os.path.exists(local_path):
        return local_path
    return "{}.pt".format(model_name)


def _export_yolo_raw_onnx(model_name, batch_size, input_size, raw_onnx_path):
    """导出 YOLOv8 为 ONNX 原始文件。"""
    if YOLO is None:
        raise RuntimeError("ultralytics 不可用: {}".format(YOLO_IMPORT_ERROR))

    yolo_name = model_name.lower()
    weights = _resolve_yolo_weight(yolo_name)
    yolo = YOLO(weights)
    exported_path = yolo.export(
        format="onnx",
        imgsz=input_size,
        batch=batch_size,
        opset=13,
        simplify=False,
        dynamic=False,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    if not isinstance(exported_path, str) or not os.path.exists(exported_path):
        raise RuntimeError("YOLO 导出后未找到 ONNX 文件: {}".format(exported_path))
    shutil.move(exported_path, raw_onnx_path)
    logger.info("导出 YOLO 原始 ONNX: %s", raw_onnx_path)


def export_model(model_name, config_path=CONFIG_PATH):
    """
    统一导出接口：按模型名称导出所有 batch_size × input_size 组合。

    Args:
        model_name: 模型名称，需在 MODEL_REGISTRY 中注册
                    (resnet18 / resnet50 / vgg11 / vgg13 / alexnet ...)
        config_path: config.yaml 路径，默认使用同目录下的配置
    """
    normalized_name = model_name.lower()
    is_torchvision = model_name in MODEL_REGISTRY
    is_yolo = normalized_name in YOLO_MODELS
    if not is_torchvision and not is_yolo:
        raise ValueError(
            f"未注册的模型: {model_name}，"
            f"可用模型: {list(MODEL_REGISTRY.keys()) + sorted(YOLO_MODELS)}"
        )

    cfg = load_config(config_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = None
    if is_torchvision:
        model = MODEL_REGISTRY[model_name](weights=None)
        model.eval()
        logger.info("已加载 torchvision 模型: %s", model_name)
    else:
        logger.info("已加载 YOLO 导出配置: %s", normalized_name)

    for bs in cfg["batch_sizes"]:
        for size in cfg["input_sizes"]:
            name = "{}_bs{}_{}x{}".format(model_name, bs, size, size)
            raw_path = os.path.join(OUTPUT_DIR, name + "_raw.onnx")
            opt_path = os.path.join(OUTPUT_DIR, name + ".onnx")

            try:
                # 1. 导出原始 ONNX
                if is_torchvision:
                    _export_onnx(model, bs, size, raw_path)
                else:
                    _export_yolo_raw_onnx(normalized_name, bs, size, raw_path)

                # 2. ORT 优化
                _optimize_onnx(raw_path, opt_path)

                # 3. 验证
                _validate_onnx(opt_path, bs, size)

                logger.info("完成: %s\n", name)
            except Exception as e:
                logger.warning("跳过 %s: %s", name, e)
            finally:
                # 清理原始文件（无论成功与否）
                if os.path.exists(raw_path):
                    os.remove(raw_path)

    logger.info("模型 %s 所有组合导出完毕", model_name)


def main():
    cfg = load_config()
    model_names = cfg.get("models", [])

    if not model_names:
        logger.warning("config.yaml 中未配置 models 列表，跳过导出")
        return

    for name in model_names:
        export_model(name)

    logger.info("所有模型导出完毕，保存于: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
