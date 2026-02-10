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
    - 接受数个参数 `intensity` (0.0 - 1.0) 来控制负载强度（包括SM_ACTIVE、SM_OCCUPANCY、DRAM_ACTIVE）。
    - 作为一个独立的 Process 运行。
- **Acceptance Criteria**:
    - 启动 Stressor 后，使用 `nvidia-smi` 可以看到 GPU 利用率上升。
- **注意事项** 确保MPS启动

### Task 4.3: Data Collector Orchestrator
- **Files**: `benchmark/collector.py`
- **Description**:
    -  全流程编排 (Workflow Orchestration)：
        1. 资源调度：根据输入参数或配置，指定用于 Benchmark 的目标 GPU 设备（支持通过设备 ID 如 0-7 进行指定），如果有多个GPU卡可以用，对队列进行分配
        2. 干扰注入循环：读取 benchmark/config.yaml 中的 Stress Level 配置组合（例如：不同的 sm-active, sm-occ, dram 强度配比），调用 Stressor (Task 4.2) 动态调整 GPU 负载环境。
        3. Kernel 遍历循环：在每一种 Stress Level 环境下，遍历 kernel_model/kernel_onnx/ 目录下的所有 ONNX Kernel 文件。记录下stress level开启后的DCGM指标，作为当前轮次所有kernel运行的背景负载
    - 执行与采集 (Execution & Acquisition)：
    在“特定干扰环境 + 特定 Kernel”的组合下执行以下原子操作：
        1. 性能测试：调用 Kernel Runner (Task 4.1)，先执行预热（Warmup），随后正式运行并记录端到端时延（Latency）。
        <!-- 2. 指标同步：根据 Kernel 运行的精确起止时间戳，结合 benchmark/config.yaml 中定义的监控指标列表，从 DCGM 中提取对应的硬件性能数据。注意，提取的时间戳不是kernel开始时，而是kernel运行开始前的背景负载，所以每个kernel之间要留一定时间间隙，使背景负载稳定下。 -->
    - 数据持久化 (Data Persistence)：
    将单次测试结果聚合，格式化写入 CSV 文件，并记录当前的GPU型号
        - Schema: [GPU, Kernel_ID, OpType, Input_Shape, DCGM_Metrics (动态列), Latency_ms]。
    - **注意事项**：注意节点间的时间同步，以及dcgm-exporter和Prometheus设置的scrape interval
- **Acceptance Criteria**:
    - 运行脚本后，能够自动化完成所有组合的测试，并生成一个包含数百条样本数据的 CSV 文件，数据完整且格式符合定义。

---
## Phase 5: Kernel Latency Modeling
*目标：训练一个回归模型，输入算子特征和负载，输出预测时延。*

### Task 5.1: 硬件性能特征算子定义
- **Files**: `training/performance_kernel.py`
- **Description**:
    - 在`training/performance_kernel.py`中定义多个算子，用来体现硬件的性能。这些算子要能分别体现出硬件对计算密集型、显存密集型算子的敏感性。
    - 在硬件上获取这些算子的执行时间
    - 支持 `--device` 参数（`auto/cpu/cuda/cuda:N`）选择执行设备并校验可用性。
    - 当输出为 `training/performance_kernel_times.json` 时采用追加语义（历史记录保留为数组），而非覆写。
- **Acceptance Criteria**:
    - 训练脚本运行结束，保存硬件性能特征算子的执行时间

### Task 5.2: Dataset Preparation
- **Files**: `training/train_kernel_model.py` (前半部分)
- **Description**:
    - 加载 Phase 4 生成的 CSV 数据和Task 5.1获得的硬件性能特征算子执行时间
    - 特征工程：将 `Input_Shape` (String) 解析为数值特征 (H, W, C, Batch)。
    - 对 `OpType` 进行 One-Hot 编码或 Label Encoding。
    - 划分 Train/Test 集。
    - 从 `training/performance_kernel_times.json` 按 `gpu_name` 映射 4 个 `hwpk_*` 特征到数据集。
    - 数据集数据不需要 `Kernel_ID`、`Input_Shape`、`Source_File`；保留单列 `GPU Type`，不再生成 GPU one-hot 列。
    - `OpType` 与 `kernel_group` 使用单列编码+标准化，不再展开成多列 one-hot。
    - 新增 `preprocess_meta.json`，保存类别映射、标准化参数与 `feature_cols`，用于训练/预测一致性对齐。
- **Acceptance Criteria**:
    - 打印出处理后的 DataFrame 头部，确认特征已数值化。

### Task 5.3: Regressor Training & Saving
- **Files**: `training/train_kernel_model.py`
- **Description**:
    - 在`training/train_kernel_model.py`中定义多个回归模型（比如XGBoost、RandomForest和LightGBM）。
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

## FINAL TASK
根据目前的环境配置编写requirmnet文件（包括cuda版本等其他信息），以及READ.ME文件