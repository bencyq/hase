# ORT 完整模型原位插桩实验方案

日期：2026-03-29

## 1. 目标与结论

本文档定义一套新的主实验方案，用于在 **完整模型原位执行** 条件下，获得 ONNX Runtime（ORT）优化后融合节点的真实 GPU 运行时间分解，并将其稳定映射回项目内的 `kernel_type` / `signature_id` 体系。

当前结论如下：

- 旧 `nsys_stage1` 独立小模型方案不再作为主归因方案继续推进。
- 新主方案采用：
  1. 完整模型原位执行
  2. ORT profiler / ORT GPU profiling 作为节点和 device kernel 事件源
  3. 必要时增加 ORT 最小插桩，补充稳定 join key
  4. 必要时用 NVTX + nsys 交叉校验
- 新方案的目标不是得到“上下文无关的静态 kernel 常数”，而是得到：
  - 在当前 ORT 配置
  - 当前 GPU / CUDA / cuDNN 环境
  - 当前模型与输入形状
  - 当前图优化和 layout 变换条件下
  的 **真实 in-graph 执行时间分解**

## 2. 背景与问题定义

旧 `nsys_stage1` 的核心假设是：

```text
独立 signature ONNX 的 pure_gpu_ms
≈
完整模型中同类融合实例的 pure_gpu_ms
```

现有实验已经说明该假设不成立。主要原因：

- Conv / Gemm 的算法选择依赖上下文，不只依赖 op 属性与 shape
- 完整图中存在 layout transform、辅助 kernel、额外 memcpy/memset
- 独立最小图与完整图在 workspace、memory layout、stream、邻接节点等方面不等价
- 当前 Stage 1 只统计 GPU kernel 总时长，未分离 copy / span / dispatch gap

因此，新的实验方案改为：

```text
完整模型原位执行
  -> 记录节点执行实例
  -> 记录节点关联的 device kernel
  -> 记录节点 span / kernel / copy / dispatch gap
  -> 再映射回 kernel_type / signature_id
```

## 3. 核心目标与非目标

### 3.1 核心目标

本实验方案必须回答以下问题：

1. 某个 ORT 优化后融合节点，在完整模型里真实触发了哪些 GPU kernel？
2. 这些 kernel 的累计 GPU 执行时间是多少？
3. 该节点在框架中的完整原位执行跨度是多少？
4. 节点间 CPU dispatch gap 有多少？
5. 节点执行记录能否稳定 join 回 `kernel_type` / `signature_id`？

### 3.2 非目标

本方案明确不追求：

- 得到跨模型、跨上下文都成立的静态 kernel 固有时间
- 在第一阶段支持 TensorRT EP、CUDA Graph、多 stream 混合场景
- 在第一阶段覆盖动态 shape、控制流图、批量模型并发采集
- 在第一阶段把全部业务语义逻辑直接嵌入 ORT C++

## 4. 统一时间口径

新方案必须同时保留以下 5 个指标，禁止再把它们混为一个值：

### 4.1 `gpu_kernel_ms`

定义：

- 某节点执行期间，所有 device `Kernel` 事件的 `(end - start)` 求和
- 单位：毫秒

意义：

- 最接近“真实 GPU kernel 计算时间”
- 不包含 host dispatch gap
- 不包含 memcpy/memset，除非它们被 device kernel 形式记录为 `Kernel`

### 4.2 `gpu_copy_ms`

定义：

- 某节点执行期间，所有 GPU `memcpy` / `memset` 类事件时长之和

意义：

- 用于处理 `Flatten`、layout transform、host-device/device-device copy 等场景
- 避免把“无 kernel 但有 GPU copy”的节点误记为 0

### 4.3 `node_span_ms`

定义：

- `fence_before` 到 `fence_after` 的时间跨度

意义：

- 表示节点在完整模型中的原位执行代价
- 包含 launch、同步、节点内等待和部分框架开销

### 4.4 `dispatch_gap_ms`

定义：

- 上一个节点 `fence_after` 到当前节点 `fence_before` 的 gap

意义：

- 表示节点间 CPU 调度/提交空隙
- 不属于某个节点的纯 GPU kernel 时间
- 只能单独统计，不能偷偷加回 `gpu_kernel_ms`

### 4.5 `model_e2e_ms`

定义：

- 完整模型一次推理循环的端到端平均时间

意义：

- 用于验证 `sum(node_span_ms)` 与整体时延是否处于同一数量级

## 5. 方案架构

## 5.1 组件分层

新方案分为 5 个组件：

1. 优化图静态分析层
2. ORT 原位执行采样层
3. ORT 插桩层
4. profile / nsys 解析层
5. join 与聚合验证层

### 5.1.1 优化图静态分析层

职责：

- 导出 ORT 优化后图
- 为每个优化后节点计算：
  - `node_name`
  - `node_index`
  - `kernel_type`
  - `signature_id`
  - `attributes`
  - shape 信息

复用现有能力：

- `ort_analysis/ort_graph_parser.py`
- `ort_analysis/fusion_detector.py`
- `nsys_stage1/pipeline.py` 中的实例/签名提取逻辑

输出：

- `optimized_node_metadata.json`

### 5.1.2 ORT 原位执行采样层

职责：

- 在完整模型条件下执行 warmup + loops
- 开启 ORT profiling
- 在构建支持时，开启 ORT GPU profiling
- 在需要 GUI 校验时，外层再加 NVTX range

输出：

- ORT raw profile JSON
- 可选 nsys `.nsys-rep` / `.sqlite`

### 5.1.3 ORT 插桩层

职责：

- 不改变计算语义
- 只补齐稳定 join key 和执行实例标识

第一阶段只允许添加：

- `run_id`
- `iteration_id`
- `exec_id`
- `node_index`
- `node_name`
- `provider`
- 可选 `stream_id`

禁止第一阶段直接把完整业务 `kernel_type` 识别逻辑塞进 ORT。

### 5.1.4 profile / nsys 解析层

职责：

- 解析 Node 事件
- 解析 Kernel 事件
- 解析 copy 事件
- 重建每次节点执行实例
- 计算 4 个时间口径

输出：

- `node_exec.json`
- `node_exec_summary.json`

### 5.1.5 join 与聚合验证层

职责：

- 将 `node_exec.json` 与 `optimized_node_metadata.json` join
- 聚合到：
  - `node_name`
  - `kernel_type`
  - `signature_id`
- 做总量校验与一致性校验

输出：

- `kernel_type_summary.json`
- `signature_summary.json`
- `validation.json`

## 5.2 数据流

```text
原始 ONNX
  -> ORT 优化图
  -> optimized_node_metadata.json

完整模型原位运行
  -> ORT raw profile JSON
  -> 可选 nsys sqlite

raw profile + metadata
  -> node_exec.json
  -> kernel_type_summary.json
  -> signature_summary.json
  -> validation.json
```

## 6. 实验阶段划分

本方案分为 5 个阶段，必须按顺序推进。

## 6.1 阶段 A：零补丁基线验证

### 6.1.1 目标

在不修改 ORT 的情况下，先验证完整模型 profiler 路线是否成立。

### 6.1.2 输入

- 现有 ORT released package
- `SessionOptions.enable_profiling = True`
- 代表模型：
  - `resnet18_bs64_224x224.onnx`
  - `vgg11_bs16_224x224.onnx`

### 6.1.3 流程

1. 完整模型执行 warmup
2. 开启 ORT profiling
3. 跑固定 loops
4. 导出 profile JSON
5. 解析：
   - `fence_before`
   - `_kernel_time`
   - `fence_after`
6. 计算：
   - `node_span_ms`
   - `_kernel_time_ms`
   - `dispatch_gap_ms`
7. 比较：
   - `sum(_kernel_time_ms)` vs `model_e2e_ms`
   - `sum(node_span_ms)` vs `model_e2e_ms`

### 6.1.4 预期现象

- `sum(_kernel_time_ms)` 明显小于 `model_e2e_ms`
- `sum(node_span_ms)` 比 `sum(_kernel_time_ms)` 更接近 `model_e2e_ms`

### 6.1.5 输出

- `benchmark/results/ort_stageA/<model>/raw_profile.json`
- `benchmark/results/ort_stageA/<model>/node_exec_baseline.json`
- `benchmark/results/ort_stageA/<model>/validation.json`

### 6.1.6 验收标准

- 每个节点至少能识别出完整的 `fence_before/_kernel_time/fence_after` 事件链
- `node_span_ms >= _kernel_time_ms` 对全部节点成立
- `sum(node_span_ms)` 与 `model_e2e_ms` 的相对误差不高于 15%
- 如果误差高于 15%，必须先定位 gap 分布，禁止直接进入下一阶段

## 6.2 阶段 B：构建 ORT GPU profiling 版本

### 6.2.1 目标

获得 Node 事件对应的 device `Kernel` 事件，不再只依赖 host `_kernel_time`。

### 6.2.2 输入

- ORT 源码
- CUDA / cuDNN 环境
- 构建参数：
  - `--enable_cuda_profiling`
  - `onnxruntime_ENABLE_NVTX_PROFILE=ON`

### 6.2.3 流程

1. 从源码构建 ORT profiling 版本
2. 安装或导出对应 Python wheel / shared lib
3. 用同一批模型重新跑完整模型 profile
4. 检查 raw JSON 中是否出现 `Kernel` 类事件
5. 将 `Kernel` 事件与前序 `Node` 事件关联

### 6.2.4 输出

- `benchmark/results/ort_stageB/<model>/raw_profile.json`
- `benchmark/results/ort_stageB/<model>/kernel_attachment_check.json`

### 6.2.5 验收标准

- raw profile JSON 中存在 device `Kernel` 事件
- 至少 95% 的 CUDA EP 节点能找到对应的 child `Kernel`
- `sum(child_kernel_ms)` 与 `_kernel_time_ms` 的相对误差不高于 10%
- 若 ORT profile 中 `Kernel` 事件严重缺失，本方案必须回退到 “NVTX + nsys 主计时，ORT 仅做标签”

## 6.3 阶段 C：最小 ORT 插桩

### 6.3.1 目标

为每次节点执行补齐稳定 join key，解决 profile 解析后无法稳定和优化图 metadata 对齐的问题。

### 6.3.2 插桩原则

- 只改公共执行路径和 profiler 发射点
- 不改 kernel 实现
- 不改图优化规则
- 不在第一版引入业务耦合

### 6.3.3 必须新增字段

- `run_id`
- `iteration_id`
- `exec_id`
- `node_index`
- `node_name`
- `provider`

### 6.3.4 可选字段

- `stream_id`
- `graph_partition`
- `thread_id`

### 6.3.5 流程

1. 在 ORT 节点执行公共路径中生成 `exec_id`
2. 在 profiler `Node` 事件中附带新增字段
3. 保证同一 run 内 `exec_id` 单调递增
4. 重新采样并导出 raw profile JSON
5. 将 raw profile 与 `optimized_node_metadata.json` 做 join

### 6.3.6 输出

- `benchmark/results/ort_stageC/<model>/raw_profile.json`
- `benchmark/results/ort_stageC/<model>/joined_node_exec.json`
- `benchmark/results/ort_stageC/<model>/join_report.json`

### 6.3.7 验收标准

- `joined_node_exec.json` 中，95% 以上节点执行实例能稳定 join 到 metadata
- `node_index + node_name + provider` 不出现系统性错配
- `exec_id` 不重复、不缺失、不乱序
- 如果 join 失败率高于 5%，必须先修补字段再继续，禁止进入聚合阶段

## 6.4 阶段 D：NVTX + nsys 交叉校验

### 6.4.1 目标

确认 ORT profiling 里 device `Kernel` 时间和 nsys/CUPTI 设备时间在同一量级，并确认 copy 行为未被遗漏。

### 6.4.2 输入

- 阶段 C 的 ORT 插桩版本
- nsys
- 外层 run-level NVTX range
- 可选 node-level NVTX range

### 6.4.3 流程

1. 在 run 外层打固定 NVTX range
2. 用 nsys 采集完整模型
3. 导出 sqlite
4. 统计：
   - `CUPTI_ACTIVITY_KIND_KERNEL`
   - `CUPTI_ACTIVITY_KIND_MEMCPY`
   - `CUPTI_ACTIVITY_KIND_MEMSET`
5. 与 ORT profile 聚合值对比：
   - `sum(gpu_kernel_ms)` vs nsys kernel total
   - `sum(gpu_copy_ms)` vs nsys copy total

### 6.4.4 输出

- `benchmark/results/ort_stageD/<model>/nsys_raw/*.sqlite`
- `benchmark/results/ort_stageD/<model>/cross_validation.json`

### 6.4.5 验收标准

- `sum(gpu_kernel_ms)` 与 nsys kernel total 的相对误差不高于 5%
- `sum(gpu_copy_ms)` 与 nsys copy total 的相对误差不高于 10%
- 主要 layout transform kernel 在 ORT/NSYS 双侧都能观察到
- 若误差高于阈值，必须先判断是：
  - ORT kernel attach 漏记
  - nsys capture range 不完整
  - memcpy/memset 未正确归档

## 6.5 阶段 E：稳定性与边界实验

### 6.5.1 目标

确认该方案在实际使用范围内稳定，而不是单次碰巧成功。

### 6.5.2 流程

对每个代表模型，在以下维度重复实验：

- 重复运行 5 次
- 更换 batch size
- 更换 `cudnn_conv_algo_search`
- 更换 `cudnn_conv_use_max_workspace`
- 可选更换 `prefer_nhwc`

### 6.5.3 输出

- `benchmark/results/ort_stageE/<model>/stability_report.json`
- `benchmark/results/ort_stageE/<model>/config_sensitivity.json`

### 6.5.4 验收标准

- 同配置重复运行 5 次时：
  - 模型总 `gpu_kernel_ms` 变异系数不高于 5%
  - Top-10 节点 `gpu_kernel_ms` 变异系数不高于 10%
- 当 provider 选项改变时，时间变化必须可解释，不允许出现无法复现的随机漂移
- 若某配置改变导致 kernel attach 或 join 大面积失效，则该配置被标为“超出一期支持边界”

## 7. 详细执行流程

## 7.1 环境准备

### 7.1.1 必备组件

- ORT released package
- ORT instrumented build
- CUDA / cuDNN
- `nsys`
- Python 依赖：
  - `onnxruntime`
  - `onnx`
  - `numpy`
  - `nvtx`

### 7.1.2 环境固定项

必须固定：

- GPU 型号
- CUDA 版本
- cuDNN 版本
- ORT commit / tag
- provider options
- 输入 shape
- loops / warmup

### 7.1.3 环境验收标准

- `ort.get_available_providers()` 仅使用目标 EP 组合
- ORT build 信息可追踪到具体 commit
- `nsys` 可正常导出 sqlite
- 同模型一次 profile 能稳定完成，不出现崩溃或空 trace

## 7.2 元数据生成

### 7.2.1 流程

1. 导出优化图
2. 计算节点实例信息
3. 计算 `kernel_type`
4. 计算 `signature_id`
5. 写出 `optimized_node_metadata.json`

### 7.2.2 最少字段

- `source_model`
- `node_name`
- `node_index`
- `kernel_type`
- `signature_id`
- `attributes`
- `activation_input_shape`
- `weight_shape`
- `bias_shape`
- `residual_shape`
- `output_shape`

### 7.2.3 验收标准

- 优化图中每个 CUDA EP 节点都能生成 metadata
- `node_name + node_index` 在单图内唯一
- 不允许出现 metadata 条数小于 profile 里 CUDA EP 节点数的情况

## 7.3 原位 profile 采样

### 7.3.1 统一执行口径

- `warmup = 20`
- `loops = 50`
- 单模型单进程
- 第一期默认 `CUDAExecutionProvider + CPUExecutionProvider`
- 第一期默认 `ExecutionMode = ORT_SEQUENTIAL`

### 7.3.2 验收标准

- raw profile JSON 完整落盘
- profile 中 Node 事件数量约等于 `优化图节点数 x loops`
- 不允许出现大面积事件缺失

## 7.4 解析与 join

### 7.4.1 解析逻辑

对于每次节点执行，生成一条 `node_exec`：

- `run_id`
- `iteration_id`
- `exec_id`
- `node_name`
- `node_index`
- `provider`
- `kernel_type`
- `signature_id`
- `gpu_kernel_ms`
- `gpu_copy_ms`
- `node_span_ms`
- `dispatch_gap_ms`

### 7.4.2 验收标准

- 解析后记录条数与期望 loops 数量一致
- 对每个节点：
  - `node_span_ms >= gpu_kernel_ms`
  - `node_span_ms >= gpu_copy_ms`
- `dispatch_gap_ms` 非负
- 无法归类的事件比例不高于 2%

## 7.5 聚合与校验

### 7.5.1 聚合层级

- `node_name`
- `signature_id`
- `kernel_type`
- `model`

### 7.5.2 必做校验

1. 设备侧总量校验

```text
sum(node_exec.gpu_kernel_ms)
≈
full_model_nsys_kernel_total_ms
```

2. copy 总量校验

```text
sum(node_exec.gpu_copy_ms)
≈
full_model_nsys_copy_total_ms
```

3. host 侧跨度校验

```text
sum(node_exec.node_span_ms)
≈
model_e2e_ms
```

### 7.5.3 验收标准

- kernel total diff <= 5%
- copy total diff <= 10%
- node span total diff <= 15%
- 若三个校验中任一失败，必须标记为该 run 不可用于训练或回归分析

## 8. 建议目录结构

```text
benchmark/results/ort_fullmodel_stage/
  <model_name>/
    optimized_node_metadata.json
    raw_profile.json
    node_exec.json
    kernel_type_summary.json
    signature_summary.json
    validation.json
    nsys_raw/
      full_model.nsys-rep
      full_model.sqlite
```

## 9. 一期支持边界

一期只支持：

- 单模型
- 固定 shape
- CUDA EP 主执行路径
- 无控制流
- 无 TensorRT EP
- 无 CUDA Graph
- 无多 stream 混合调度

以下场景默认不支持：

- 动态 shape 折返路径
- 多 session 并发
- TensorRT / CUDA 混合分区
- graph capture replay
- 多 GPU 并行

## 10. 风险与处置

## 10.1 风险：ORT profiling 本身带来额外开销

处置：

- profiling 结果只用于分解与归因，不直接作为端到端真实 wall time
- `model_e2e_ms` 需单独在非 profiling 模式下测一次作为参考

## 10.2 风险：device kernel 无法稳定 attach 到 Node

处置：

- 回退到 “ORT 仅提供 node 执行边界 + nsys 做主计时”
- 保留 `exec_id` / `node_index` / `iteration_id` 作为匹配辅助

## 10.3 风险：copy 事件难以归属到具体节点

处置：

- 第一版允许只做 run-level copy total 校验
- 节点级 copy 归属作为增强目标，不作为一期阻塞项

## 10.4 风险：provider 选项变化导致算法漂移

处置：

- 将 provider options 作为实验元数据强制落盘
- 任何跨配置比较都必须先确认：
  - `cudnn_conv_algo_search`
  - `cudnn_conv_use_max_workspace`
  - `prefer_nhwc`
  - `use_tf32`

## 11. 项目验收门槛

当且仅当以下条件全部满足时，本方案视为一期成功：

1. 零补丁阶段确认 `node_span_ms` 比 `_kernel_time_ms` 更接近模型时延
2. ORT GPU profiling 版本可稳定产出 device `Kernel` 事件
3. 95% 以上节点执行实例可稳定 join 回优化图 metadata
4. `sum(gpu_kernel_ms)` 与 nsys kernel total diff <= 5%
5. `sum(node_span_ms)` 与模型端到端时延 diff <= 15%
6. 至少 2 个代表模型在 5 次重复运行下稳定通过

若以上任一条件未满足，则不得进入“生产训练数据采集”或“性能预测建模”阶段。

## 12. 与旧 Stage 1 的关系

- 旧 `nsys_stage1` 保留，但角色改为：
  - 静态图分析工具
  - signature 体系生成工具
  - 独立小模型微基准工具
- 旧 `nsys_stage1` 不再承担：
  - 完整模型真实原位时间归因
  - 全量校验真值生成
  - 后续训练标签主来源

## 13. 下一步落地顺序

执行顺序固定为：

1. 先完成阶段 A 的 parser 扩展
2. 再完成阶段 B 的 ORT profiling build
3. 再做阶段 C 的最小 ORT 插桩
4. 最后做阶段 D 的 nsys 交叉校验

任何阶段未达验收标准，禁止跳过直接推进后续阶段。
