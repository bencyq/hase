# Benchmark3 子任务：模型整体运行 vs 内核逐个运行开销分析

## 目标
对以下 5 个模型执行基准测试，分析“内核分别运行”相对于“直接运行整模型”的多余时间开销比例。

## 待测模型
1. `model_zoo/models/densenet121_bs32_224x224.onnx`
2. `model_zoo/models/mobilenet_v2_bs32_448x448.onnx`
3. `model_zoo/models/resnet18_bs64_224x224.onnx`
4. `model_zoo/models/vgg16_bs64_224x224.onnx`
5. `model_zoo/models/YOLOv8m_bs64_448x448.onnx`

## 测试内容
对每个模型分别完成两组测试（保持相同设备、相同 warmup、相同 loop 次数）：

1. **整模型直跑耗时**
   - 使用 ONNX Runtime 直接运行模型，记录平均推理耗时 `T_model`（ms）。

2. **内核逐个运行总耗时**
   - 将模型拆分为内核后，按执行顺序逐个运行并统计总耗时，记录 `T_kernel_sum`（ms）。

## 关键指标
对每个模型计算：

- 额外耗时（ms）：
  - `T_extra = T_kernel_sum - T_model`
- 额外开销比例（%）：
  - `Overhead_Ratio = (T_extra / T_model) * 100%`

说明：当 `T_extra > 0` 时，表示“内核逐个运行”相对“整模型直跑”存在额外开销。

## 输出要求
1. 生成逐模型结果表，至少包含：
   - `Model`
   - `T_model_ms`
   - `T_kernel_sum_ms`
   - `T_extra_ms`
   - `Overhead_Ratio_percent`
2. 汇总 5 个模型的开销比例并做简要对比分析：
   - 哪些模型开销比例高
   - 可能原因（如内核数量、调度/启动开销、内存访存开销累积）

## 验收标准
1. 5 个模型均完成测试并有完整指标数据。
2. 每个模型都给出 `Overhead_Ratio`。
3. 输出一份对比结论，明确不同模型的额外开销差异。
