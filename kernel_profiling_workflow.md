# 内核级 GPU 时间数据收集流程

## 核心原则

ORT 的 `sess.run()` 计时包含 CPU-GPU 同步开销，不反映真实 GPU 执行时间。  
正确做法是通过 **nsys 读取 CUDA HW 时间线**，只统计 GPU 上实际执行的时间。

---

## 准备工作

- 安装 NVIDIA Nsight Systems（`nsys` 命令行工具）
- 安装 Python 包：`onnxruntime`（CUDAExecutionProvider）、`nvtx`、`onnx`
- 确认模型文件路径：`model_zoo/models/resnet18_bs32_224x224.onnx`

---

## 方案一：独立小模型逐个 nsys 采集（构建训练数据集）

适合场景：需要覆盖多种内核配置，构建 `(config → latency)` 训练集。

### Step 1：从模型中提取所有内核配置

使用 `ort_analysis/ort_graph_parser.py` 解析模型，生成 `ort_kernel_record/resnet18.json`，记录每个节点的类型与输入形状。

### Step 2：为每个内核配置生成独立 ONNX 文件

使用 `kernel_model/kernel_builder.py` 读取 Step 1 的 JSON，为每个唯一的 `(kernel_type, shape)` 组合生成一个只含该内核的独立 `.onnx` 文件，输出到 `kernel_model/kernel_onnx/`。

### Step 3：编写采集脚本

脚本逻辑如下，**不使用 ORT 计时**：

1. 加载独立内核 ONNX，创建 ORT InferenceSession（CUDA EP）
2. 执行 20 次 warmup，确保 cuDNN 完成算法选择和显存分配
3. 用 `nvtx.annotate("measure")` 标记正式采集区间
4. 在标记区间内循环推理 50 次

### Step 4：对每个内核文件执行 nsys 采集

```
nsys profile
  --trace cuda,nvtx
  --capture-range nvtx
  --nvtx-capture measure
  python run_single_kernel.py --model <kernel.onnx>
```

每个内核单独生成一个 `.nsys-rep` 文件。

### Step 5：将 .nsys-rep 导出为 SQLite

```
nsys export --type sqlite --output <kernel>.sqlite <kernel>.nsys-rep
```

### Step 6：从 SQLite 中读取 GPU 执行时间

查询 `CUPTI_ACTIVITY_KIND_KERNEL` 表，统计 `measure` 区间内所有 CUDA kernel 的 `(end - start)` 之和，除以循环次数，换算为毫秒，即为该内核的真实平均 GPU 执行时间。

### Step 7：汇总为数据集

将所有内核的 `(kernel_id, config_features, gpu_latency_ms)` 写入 CSV，作为后续回归模型的训练数据。

---

## 方案二：单次全模型 nsys 采集（快速验证参考基准）

适合场景：仅需了解某一模型的逐层 GPU 耗时分布，不需要覆盖多配置。

### Step 1：编写推理脚本

用 `nvtx.annotate("inference_run")` 包裹正式推理循环，warmup 放在标记区间外。

### Step 2：nsys 采集

```
nsys profile
  --trace cuda,nvtx
  --capture-range nvtx
  --nvtx-capture inference_run
  python profile_full_model.py
```

### Step 3：导出并查询

导出 SQLite 后，查询 `CUPTI_ACTIVITY_KIND_KERNEL` 表，按 `shortName`（cuDNN kernel 名称）分组统计各类 GPU kernel 的累计执行时间。

> **注意**：该方案得到的是 cuDNN kernel 名（如 `cudnn::ops::scudnnConvolution_...`），而非 ONNX 节点名，两者之间的映射需要额外处理（见方案三）。

---

## 方案三：编译 ORT 开启 NVTX 支持（节点名精确对应）

适合场景：需要将 GPU 执行时间精确对应到每个 ONNX 节点名。

### Step 1：从源码编译 ORT，开启 NVTX

编译时添加 CMake 参数 `onnxruntime_ENABLE_NVTX_PROFILE=ON`，ORT 会在每个节点执行前后向 CUDA stream 插入 NVTX marker，标注节点名。

### Step 2：nsys 采集

与方案二相同，nsys 采集全模型推理。

### Step 3：在 Nsight Systems GUI 或 SQLite 中直接读取节点级时间

由于 NVTX marker 与 CUDA kernel 在同一时间线对齐，每个 ONNX 节点对应的 GPU 执行时间可以直接从 NVTX range 内的 kernel 时间求和得到，无需手动映射。

---

## 三种方案选择建议

| 目标 | 推荐方案 |
|---|---|
| 构建多内核配置的训练数据集 | 方案一 |
| 快速了解单个模型的逐层耗时分布 | 方案二 |
| 需要 ONNX 节点名与 GPU 时间精确对应 | 方案三 |
| 生产环境，不需要内核级细节 | 直接用 nsys 测整体推理延迟 |

---

## 关键注意事项

- **warmup 必须在 nvtx capture range 之外**，否则 cuDNN 算法选择耗时会污染测量结果
- **nsys capture-range 设置为 nvtx 模式**，只采集标记区间，避免 session 初始化干扰
- 独立小模型的 GPU 执行时间与在完整模型中的执行时间**不完全相等**（显存局部性、cuDNN workspace 复用等因素），但对于同规格的配置样本，偏差是系统性一致的，回归模型可以学习并校正
- `CUPTI_ACTIVITY_KIND_KERNEL` 中的时间单位为**纳秒**，换算为毫秒需除以 `1e6`
