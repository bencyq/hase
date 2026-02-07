# Hase Implementation Plan

## Phase 0: Infrastructure & Skeleton
*目标：建立项目基础结构，配置日志与通用工具。*

### Task 0.1: Project Skeleton & Configuration
- **Files**:  `utils/logger.py`, 
- **Description**:
    - 创建基本文件夹结构。
    - 实现 `logger.py`，配置控制台和文件日志。

---

## Phase 1: Model Zoo & ONNX Preparation
*目标：准备用于分析和测试的标准模型数据。*

### Task 1.1: Torchvision Exporter
- **Files**: `model_zoo/torchvision_exporter.py`
- **Description**:
    - 创建config.yaml，设定batchsize和输入张量的类型，张量输入则有三种类型：224x224，320x320，512x512，batchsize设置有16，64，128
    - 编写一个脚本，从 `torchvision` 加载 `ResNet18`。
    - 将模型导出为 onnx runtime 优化后的ONNX 格式，并根据config.yaml里的不同组合导出。只能使用CUDA runtime
    - 保存到 `model_zoo/models/`。
- **Acceptance Criteria**:
    - 运行脚本后，`model_zoo/models/` 下出现 `.onnx` 文件。
    - 使用 onnx runtime 验证模型文件有效。

---

## Phase 2: ORT Graph Analysis
*目标：解析 ONNX 模型，理解算子（Kernel）结构与融合策略。*

### Task 2.1: Basic Graph Parser
- **Files**: `ort_analysis/ort_graph_parser.py`
- **Description**:
    - 加载 ONNX 模型。
    - 使用 ONNX Runtime 的 `InferenceSession` 获取优化后的图（GraphOptimizationLevel.ORT_ENABLE_ALL）。
    - *注意*：这一步通常需要保存优化后的模型到磁盘才能解析。
    - 遍历图中的所有 Node，提取 OpType, Input Shapes, Output Shapes。
- **Acceptance Criteria**:
    - 运行脚本，打印出模型中所有节点的列表（例如：`Conv`, `Relu`, `Add`）及其维度。

### Task 2.2: Fusion Detector & Kernel Recorder
- **Files**: `ort_analysis/fusion_detector.py`, `ort_analysis/ort_kernel_record/`
- **Description**:
    - 识别哪些算子被 ORT 融合了（例如 `Conv+Bias+Activation`）。
    - 将每个独立的 Kernel（或融合后的子图）的属性保存为 JSON 格式。
    - 提取并保存每个 Kernel 的元数据（算子类型、FLOPs估算、输入/输出张量大小）到 `ort_kernel_record/`。
    - 其中，算子类型要求不同模型的相同融合策略会被识别为同一种内核。比如模型A和模型B都有将Conv+Add融合的内核，则识别为同一个，要在命名上保证统一性，比如Conv_Add_Relu，不同张量输入但是内核结构相同的用同一个命名，以便后续预测时不会预测成多个内核，且预测输入能接受张量大小。
- **Acceptance Criteria**:
    - 为 ResNet18 生成一个 JSON 文件，包含其所有内核的结构化描述。
    - JSON里还包含不同的输入输出张量

---

## Phase 3: Kernel Model Generation
*目标：将解析出的每个内核提取为独立可运行的小模型，用于后续Benchmark。*

### Task 3.1: Single kernel Extractor
- **Files**: `kernel_model/kernel_builder.py`
- **Description**:
    - 实现 `kernel_build` 函数。
    - 输入：`ort_kernel_record/`下的kernel的json文件，
    - 功能：使用 onnx runtime 将该单独的 kernel 保存为一个独立的 `.onnx` 文件。
    - 自动为提取出的模型生成对应的 Dummy Input 数据（随机数），用于验证模型合法性。
- **Acceptance Criteria**:
    - 指定 `ort_kernel_record/`目录下的某个kernel元数据文件，成功生成 `<kernel_fusion_rule>.onnx`，如`Conv_Add_Relu_255x255.onnx`，保存到`kernel_model/kernel_onnx/`下
    - 能用 ONNX Runtime 成功运行这个小模型。


---

## Phase 4: Benchmark & Data Collection
*目标：构建真实环境下的数据采集系统，建立“负载-时延”数据集。*

### Task 4.1: Single Kernel Runner
- **Files**: `benchmark/ort_kernel_runner.py`
- **Description**:
    - 接受一个 `.onnx` 文件路径作为输入。
    - 使用 ONNX Runtime (CUDAExecutionProvider) 运行该模型。
    - 实现 Warmup (预热) 和 Loop (循环) 机制。（函数可选择关闭预热以及设置loop次数）
    - 使用 `time.perf_counter` 或 `torch.cuda.Event` 测量纯推理耗时（平均值）。
    - 记录启动和结束时间
- **Acceptance Criteria**:
    - `python ort_kernel_runner.py --model path/to/conv.onnx` 输出该 Kernel 的平均执行时间 (ms)，以及启动结束时间。

### Task 4.2: Background Stressor
- **Files**: `benchmark/stressor.py`
- **Description**:
    - 实现一个产生 GPU 负载的脚本。
    - 使用 PyTorch 进行持续的大矩阵乘法或显存拷贝操作。
    - 接受一个参数 `intensity` (0.0 - 1.0) 来控制负载强度（例如通过控制 sleep 时间或矩阵大小）。
    - 作为一个独立的 Process 运行。
- **Acceptance Criteria**:
    - 启动 Stressor 后，使用 `nvidia-smi` 可以看到 GPU 利用率上升。

### Task 4.3: Data Collector Orchestrator
- **Files**: `benchmark/collector.py`
- **Description**:
    - 编排采集流程：
        1. 启动 Stressor (Task 4.2)，遍历不同的 Stress Level (例如: 无负载, 30%, 60%, 90%，由`benchmark/config.yaml`中读取)。
        2. 在每个stress level中，遍历 `kernel_model/kernel_onnx/` 中的所有 Kernel。
        3. 切换stressor的level
    - 在每个组合下：
        1. 启动预热，然后运行 Kernel Runner (Task 4.1) 获取时延。
        2. 读取 DCGM 指标（根据`benchmark/config.yaml`里的metric和kernel的启动结束时间读取）。
    - 将结果保存为 CSV：`[Kernel_ID, OpType, Input_Shape, DCGM_Metrics(有多个，由config.yaml决定), Latency_ms]`。
- **Acceptance Criteria**:
    - 运行脚本后，生成一个包含数百条数据的 CSV 文件。

---
## Phase 5: Kernel Latency Modeling
*目标：训练一个回归模型，输入算子特征和负载，输出预测时延。*

### Task 5.1: Dataset Preparation
- **Files**: `training/train_kernel_model.py` (前半部分)
- **Description**:
    - 加载 Phase 4 生成的 CSV 数据。
    - 特征工程：将 `Input_Shape` (String) 解析为数值特征 (H, W, C, Batch)。
    - 对 `OpType` 进行 One-Hot 编码或 Label Encoding。
    - 划分 Train/Test 集。
- **Acceptance Criteria**:
    - 打印出处理后的 DataFrame 头部，确认特征已数值化。

### Task 5.2: Regressor Training & Saving
- **Files**: `training/train_kernel_model.py`
- **Description**:
    - 在`training/train_kernel_model.py`中定义多个回归模型（比如XGBoost 和 RandomForest）。
    - 在 `train_kernel_model.py` 中训练模型，目标是预测 `Latency_ms`。
    - 评估模型在 Test 集上的 MAPE (Mean Absolute Percentage Error)。
    - 保存训练好的模型到磁盘。
- **Acceptance Criteria**:
    - 训练脚本运行结束，打印出 MAPE 误差值（例如 <15%），并生成模型文件。

---

## Phase 6: Graph Modeling & Prediction
*目标：结合 DAG 结构和 Kernel 预测值，预测端到端时延。*

### Task 6.1: DAG Critical Path Analyzer
- **Files**: `graph_model/dag_critical_path.py`
- **Description**:
    - 重新加载 Phase 2 生成的 JSON 或原始模型。
    - 构建 DAG (有向无环图)，表示算子间的依赖关系，并写入到`graph_model/model_DAG`下。
    - 实现一个简单的 `calculate_latency` 函数：
        - 输入：DAG + 每个 Node 的预测耗时。
        - 逻辑：由于 ORT 默认是顺序流执行 (Stream 0)，初始版本只需对拓扑排序后的节点耗时进行求和。
- **Acceptance Criteria**:
    - 给定`ort_analysis/ort_kernel_record/`下的JSON 内核描述文件，生成对应的DAG文件。

### Task 6.2: Overlap Analyze
- **Files**: `overlap_simulator.py`
- **Description**:
     - 分析可能存在的kernel并行，获得内核执行的关键路径。如果onnx runtime确实是将kernel依次串行执行的则罢了

---

## Phase 7: Graph Prediction & Evaluation
### Task 7.1: Predictor Integration
- **Files**: `inference/predict_model_latency.py`,`inference/config.yaml`
- **Description**:
    - `inference/config.yaml`:
        设定指定的Prometheus url
    - `predict_model_latency.py`:
        1. 解析输入模型，构建 DAG。
        2. 对每个 kernel，调用Phase5训练好的模型，预测在当前状态下的耗时。（stress level由直接调用Prometheus获取实时的值）
        3. 调用 `dag_critical_path.py` 累加耗时。
- **Acceptance Criteria**:
    - 输入：`resnet18.onnx`，`inference/config.yaml` 。
    - 输出：预测的总耗时 (ms)。

### Task 7.2: Evaluation
- **Files**:`evaluator.py`
- **Description**:
    调用`inference/predict_model_latency.py`，对比实际的模型运行延迟和用预测器预测出的延迟
- **Acceptance Criteria**:
    - 输入：`resnet18.onnx`或者一个模型文件目录（包含多个模型）
    - 输出：单个或多个模型的延迟预测差距
---
