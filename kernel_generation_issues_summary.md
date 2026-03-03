# 内核生成问题简报

## 当前主要问题

1. `kernel_builder` 初始只支持少量内核类型，遇到 `QuickGelu` / `AveragePool` 等会中断。
2. `ort_kernel_record` 中部分 `activation_input_shape` / `output_shape` 为空，导致构建器无法直接生成。
3. `fusion_detector` 早期没有把常量信息传给 shape 推导，`Reshape/Slice/Concat` 等链路会断 shape。
4. 全量构建时出现算子细节问题（如 `Conv` bias 维度、`BatchNormalization` 输入数量）。
5. 全量验证命令多次被环境中断（`Tool failed/Aborted`），导致一次性跑完的结果不稳定。

## 已尝试的解决方案

1. **扩展 `kernel_builder` 覆盖范围**
   - 新增支持：`QuickGelu`、`AveragePool`、`BatchNormalization`、`Add/Mul/Div/Sub`、`Concat`、`Split`、`Reshape`、`Transpose`、`Pad`、`Slice`、`Resize`、`Clip`、`HardSigmoid`、`Softmax` 等。
   - 增加“已存在文件跳过”逻辑，避免重复覆盖。

2. **补齐 shape 透传规则（`ort_graph_parser`）**
   - 增加常量读取与推导工具（initializer value map、broadcast、reshape 规则等）。
   - 扩展大量算子的 shape 推导与透传，减少 `null` shape。

3. **修复记录生成链路（`fusion_detector`）**
   - 调整为调用 `_infer_node_output_shape(..., const_map)`，让依赖常量的算子也能正确推导。
   - 重新跑了全量 `fusion_detector`，`null` shape 数量明显下降。

4. **增强构建阶段鲁棒性**
   - 对 `Conv` 的 bias 做 1D 兜底。
   - 对缺失 shape 的记录加最小可运行 fallback（按 `kernel_type/weight_shape/batch_size` 估算）。

## 目前状态

1. `ort_kernel_record` 中绝大多数 kernel 的 shape 已可推导，`both null` 记录显著减少。
2. `kernel_builder` 基本具备“覆盖记录中全部 kernel_type”的代码路径。
3. 仍需一次稳定的全量执行（不被环境中断）来做最终闭环验证。
4. 现阶段剩余风险主要是：个别历史 JSON（如旧前缀文件）和极端 shape 组合下的边界兼容。
