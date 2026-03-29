# 仓库规范

## 项目结构与模块组织
本仓库按工作流阶段组织。`model_zoo/` 用于将源模型导出为 ONNX，`ort_analysis/` 用于解析优化后的 ORT 图和融合记录，`kernel_model/` 用于构建独立的 kernel ONNX 文件，`benchmark/` 用于收集延迟和 GPU 负载数据，`training/` 用于准备数据集并训练回归器，`graph_model/` 用于构建基于 DAG 的模型图，`inference/` 用于预测端到端延迟。`debug/` 用于排查脚本，`utils/` 用于共享辅助函数，生成产物应保存在已有数据目录中，例如 `training/models/`、`training/dataset/`、`benchmark/csv/` 和 `graph_model/model_DAG/`。

## 构建、测试与开发命令
请从仓库根目录运行脚本，以确保相对路径解析正确。

- `bash container.sh` 或其中的 `docker run ...` 命令：启动启用 CUDA 的开发容器。
- `python model_zoo/torchvision_exporter.py`：导出并验证 `model_zoo/config.yaml` 中定义的 ONNX 模型。
- `python ort_analysis/ort_graph_parser.py --help`：在生成 ORT 图元数据前查看解析器选项。
- `python benchmark/collector.py --help`：使用 `benchmark/config.yaml` 中的设置运行基准测试编排器。
- `python training/train_kernel_model.py --help`：构建数据集并训练延迟回归器。
- `python inference/predict_model_latency.py --help`：使用训练产物和实时 Prometheus 指标预测模型延迟。

## 编码风格与命名约定
使用 Python，采用 4 空格缩进；函数和模块使用 `snake_case`，配置常量使用具有描述性的 `UPPER_CASE` 名称。遵循现有 CLI 模式：可执行脚本保持基于 argparse，并将可复用逻辑放入函数中，而不是写在顶层代码里。优先编写小而聚焦的模块，并保持 JSON/YAML 文件名与模型名一致，例如 `resnet18.json` 或 `resnet18_bs64_224x224.onnx`。

## 测试规范
目前还没有统一的测试运行器；验证以脚本驱动为主。对于代码修改，请运行距离改动最近的可执行检查，例如使用 `python benchmark/stressor_test.py` 验证 stressor 行为，或使用 `python benchmark2/test_export_large_onnx.py` 验证导出。新增检查请命名为 `test_*.py`，并放在靠近其验证功能的位置，同时在文件头或 PR 描述中说明所需的 GPU、CUDA 或 Prometheus 前置条件。

## 提交与 Pull Request 规范
最近的提交通常使用简短、面向任务的摘要，且常为中文，例如 `完成Task 7.1...` 或 `添加了论文用的并行负载实验`。提交信息应保持简洁、使用祈使语气，并聚焦单一改动。对于 pull request，请说明受影响的工作流阶段、你运行过的命令、所需硬件或服务依赖；如果改动影响了生成的报告、CSV 或性能分析产物，请附上示例输出或截图。

## 配置与数据
将 `benchmark/config.yaml`、`inference/config.yaml` 和 Prometheus 端点视为环境相关配置。不要在新代码中硬编码密钥或仅适用于集群环境的地址。位于 `benchmark3/results/`、`debug/` 或 `training/models/` 下的大型生成文件，只有在它们是明确交付物时才应提交。
