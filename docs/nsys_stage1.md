# NSYS Stage 1 统一说明

日期：2026-03-29

## 1. 目标与边界

NSYS Stage 1 的目标是：在不修改 ONNX Runtime、不引入定制运行时的前提下，为单个模型 `model_zoo/models/resnet18_bs64_224x224.onnx` 建立一条可验证的闭环，输出模型内各 `kernel_type` 的纯 GPU 时间分布，并用一次完整模型 `nsys` 采集做总量校验。

本阶段唯一时间口径为 `pure_gpu_ms`，定义为：

- 目标 NVTX capture range 内所有 CUDA kernel 执行时长之和
- 再除以正式循环次数 `loops`
- 再换算为毫秒

该口径不包含：

- CPU dispatch
- launch gap
- 等待时间

本阶段明确不做：

- 完整模型运行期间每个融合实例的精确原位归因
- 端到端时间归因
- ORT 重编译
- 节点级 NVTX
- 批量模型处理
- 针对不同 `nsys` SQLite schema 的兼容分支

## 2. 核心结论

当前约束下，可行的最短路径不是直接从完整模型时间线反推出 ORT 融合类型时间，而是：

1. 解析优化后模型，得到融合实例
2. 为每个唯一融合签名生成独立 kernel ONNX
3. 用 `nsys + NVTX` 采集每个签名的纯 GPU 时间
4. 将签名时间回填到模型实例
5. 按 `kernel_type` 聚合
6. 用完整模型 `nsys` 做总量校验

原因很直接：

- `nsys` 默认看到的是 CUDA/cuDNN kernel 名，不是 ORT 融合节点名
- 一个融合节点可能对应多个底层 CUDA kernel
- 多个不同融合节点也可能落到相似的底层 CUDA kernel

因此，在不改 ORT 的前提下，不能把完整模型时间线稳定、自动、无歧义地映射回 `Conv_Add`、`Conv_Add_Relu` 这类融合类型。

## 3. 数据闭环

Stage 1 的数据流为：

```text
优化后 ONNX
  -> 融合实例表
  -> 唯一融合签名表
  -> 独立 kernel ONNX
  -> nsys profile / export sqlite
  -> 每个签名的 pure_gpu_ms
  -> 回填模型实例
  -> 按 kernel_type 聚合
  -> 与完整模型 pure_gpu_ms 对比校验
```

本阶段必须区分 3 类对象：

### 3.1 融合实例

一条实例记录对应优化图中的一个具体节点，最少字段：

- `source_model`
- `node_name`
- `kernel_type`
- `activation_input_shape`
- `weight_shape`
- `bias_shape`
- `residual_shape`
- `output_shape`
- `attributes`
- `signature_id`

### 3.2 唯一融合签名

唯一签名用于表达“可独立构建且执行语义等价”的 kernel，最少由以下字段组成：

- `kernel_type`
- `attributes`
- `activation_input_shape`
- `weight_shape`
- `bias_shape`
- `residual_shape`
- `output_shape`

`signature_id` 由上述字段的规范化 JSON 做稳定哈希生成。

### 3.3 纯 GPU 时间记录

一条时间记录对应一个唯一签名，最少字段：

- `signature_id`
- `kernel_type`
- `loops`
- `gpu_kernel_time_sum_ns`
- `pure_gpu_ms`
- `nsys_rep_path`
- `sqlite_path`
- `capture_range_name`
- `gpu_name`

## 4. 当前选定方案

采用一条新的 `nsys_stage1/` 专用流水线，不直接复用 `ort_analysis/ort_kernel_record/*.json` 作为中间格式。原因是现有记录按 `node_name` 聚合，不能表达 Stage 1 所需的“实例表 + 签名表 + 签名时间回填”闭环。

现有仓库中只复用合适的底层能力：

- `ort_analysis/ort_graph_parser.py`
  - ORT 优化图导出
  - shape 推导
- `ort_analysis/fusion_detector.py`
  - `kernel_type` 识别
- `kernel_model/kernel_builder.py`
  - 独立 kernel ONNX 构建
  - ONNX Runtime 可运行性验证

## 5. 已落地代码

当前已新增 3 个脚本：

- `nsys_stage1/pipeline.py`
- `nsys_stage1/run_with_nvtx.py`
- `nsys_stage1/nsys_sqlite_parser.py`

职责如下：

### 5.1 `nsys_stage1/run_with_nvtx.py`

- 用 ONNX Runtime 加载模型
- 自动构造输入
- warmup 放在 NVTX range 外
- 正式 loop 放在固定 NVTX range 内

### 5.2 `nsys_stage1/nsys_sqlite_parser.py`

- 调 `nsys profile`
- 调 `nsys export --type sqlite`
- 解析 SQLite
- 提取目标 NVTX range 内 GPU kernel 时间总和
- 计算 `pure_gpu_ms`

### 5.3 `nsys_stage1/pipeline.py`

- 读取 ORT 优化后模型
- 生成实例表和签名表
- 为每个 `signature_id` 构建独立 kernel ONNX
- 采集每个签名的 `pure_gpu_ms`
- 回填实例并按 `kernel_type` 聚合
- 对完整模型做一次同口径校验

当前已完成的静态检查：

```bash
python -m py_compile \
  nsys_stage1/run_with_nvtx.py \
  nsys_stage1/nsys_sqlite_parser.py \
  nsys_stage1/pipeline.py
```

## 6. 统一执行口径

当前固定参数为：

- `warmup = 20`
- `loops = 50`
- NVTX range 名称：`measure`
- 校验阈值：`0.15`

单 kernel 采集命令模板：

```bash
nsys profile \
  --trace cuda,nvtx \
  --capture-range nvtx \
  --nvtx-capture measure \
  --output <output_prefix> \
  python nsys_stage1/run_with_nvtx.py \
    --model <kernel.onnx> \
    --warmup 20 \
    --loops 50 \
    --range-name measure
```

单模型总入口：

```bash
python nsys_stage1/pipeline.py \
  --model model_zoo/models/resnet18_bs64_224x224.onnx \
  --threshold 0.15 \
  --clean
```

## 7. 预期输出

如果单模型流程跑通，应生成：

结果目录：

- `benchmark/results/nsys_stage1/resnet18_bs64_224x224/`

独立 kernel ONNX 目录：

- `kernel_model/kernel_onnx/nsys_stage1/resnet18_bs64_224x224/`

结果目录中至少包含：

- `instances.json`
- `signatures.json`
- `signature_models.json`
- `kernel_times.json`
- `kernel_type_summary.json`
- `full_model_time.json`
- `validation.json`

## 8. 校验规则

完整模型校验时，比较两项：

- `aggregated_model_gpu_ms`
  - 所有实例回填后的 `pure_gpu_ms` 求和
- `full_model_gpu_ms`
  - 完整模型 NVTX range 内 GPU kernel 总时间除以 `loops`

通过条件：

```text
abs(aggregated_model_gpu_ms - full_model_gpu_ms) / full_model_gpu_ms <= 0.15
```

超过 15% 视为 Stage 1 失败，需要停止继续堆补丁并重新讨论方案。

## 9. 当前真实进度

已完成：

- 单模型范围、时间口径和校验阈值已定
- 设计文档已形成
- Stage 1 专用代码骨架已落地
- 3 个脚本已通过 Python 语法检查
- 在容器 `7ae8bbf75f1c` 内安装了：
  - `nvtx`
  - `cuda-nsight-systems-11-6`

已确认容器能力：

- `onnx` 可导入
- `onnxruntime` 可导入
- `numpy` 可导入
- `nvtx` 可导入
- ORT 可见 provider：
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`

尚未完成：

- 单模型解析后的实例数和签名数确认
- 每个 `signature_id` 的独立 ONNX 实际生成与验证
- 每个签名的 `nsys` 采集与 SQLite 导出
- 全模型 `nsys` 采集
- 15% 阈值校验

## 10. 当前阻塞点

当前代码不是阻塞点，GPU 资源才是阻塞点。

在原实验机器上，宿主机和容器都看到两张卡处于：

- `Compute Mode = Exclusive_Process`
- GPU 利用率接近 100%

占用进程为：

- GPU 0: PID `47444`，`/root/miniconda3/envs/csj/bin/python`
- GPU 1: PID `47445`，`/root/miniconda3/envs/csj/bin/python`

由此在容器内调用 ORT 时会报：

```text
CUDA failure 46: all CUDA-capable devices are busy or unavailable
```

所以原机器上未能继续完成单模型实验。

## 11. 在别的机器上继续的最小步骤

### 11.1 环境要求

确保：

- NVIDIA GPU 可用
- `nvidia-smi` 正常
- GPU 没有被独占进程占满
- 已安装：
  - `onnx`
  - `onnxruntime-gpu`
  - `numpy`
  - `nvtx`
  - `nsys`

Ubuntu + CUDA apt 源安装 `nsys` 示例：

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-nsight-systems-11-6
```

Python 包：

```bash
pip install nvtx
```

### 11.2 检查命令

先检查 GPU：

```bash
nvidia-smi
```

再检查 ORT provider：

```bash
python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

再做脚本语法检查：

```bash
python -m py_compile \
  nsys_stage1/run_with_nvtx.py \
  nsys_stage1/nsys_sqlite_parser.py \
  nsys_stage1/pipeline.py
```

最后跑单模型流水线：

```bash
python nsys_stage1/pipeline.py \
  --model model_zoo/models/resnet18_bs64_224x224.onnx \
  --threshold 0.15 \
  --clean
```

## 12. 验收标准

单模型 `resnet18_bs64_224x224.onnx` 验收通过必须同时满足：

- 成功生成实例表和签名表
- 每个 `signature_id` 成功生成且仅生成一个独立 kernel ONNX
- 每个 `signature_id` 都有稳定的 `pure_gpu_ms` 记录
- 成功输出 `kernel_type` 聚合结果
- 成功输出完整模型校验结果
- 聚合总 GPU 时间与完整模型 GPU 总时间的偏差不超过 15%

## 13. 备注

此前分散在 `docs/superpowers/specs/` 下的 3 份文档：

- `nsys_stage1_todo.md`
- `2026-03-29-nsys-stage1-single-model-design.md`
- `nsys_stage1_progress_2026-03-29.md`

其有效信息已经在本文件中统一归纳。后续如果继续推进 Stage 1，优先维护本文件即可。
