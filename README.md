# 概况
目前版本存在较大问题。
这个版本采用profiler方案，即建议一
目前进度看`debug/解释.md`和`debug/总结.md`
## 根本原因分析

从你的 profiler JSON 数据可以清楚看到，每个节点的事件模式是：

```
fence_before (dur=0~1μs) → kernel_time (dur=实际GPU计算μs) → fence_after (dur=0μs)
```

以第一个 Conv 为例（首次运行数据）：

```
fence_before:   ts=174212, dur=1
kernel_time:    ts=174219, dur=4805    (→ 结束于 ~179024)
fence_after:    ts=179033, dur=0
```

两个节点之间的 **空隙**（`fence_after` → 下一个 `fence_before`）是 **6~20 μs**，这代表 **ORT 框架的 CPU 调度/分发开销**。

**`_kernel_time` 使用 CUDA Events 只计量了 GPU 侧的纯计算时间**，而遗漏了以下开销：

| 遗漏的开销 | 每节点约耗时 | ResNet-18 (~22 节点) 合计 |
|---|---|---|
| ORT CPU 调度/节点分发 | 50~100 μs | 1.1~2.2 ms |
| CUDA kernel launch | 5~10 μs | 0.11~0.22 ms |
| 内存管理 (alloc/dealloc) | 若干 μs | 若干 μs |
| GPU idle（等 CPU 派发下一个 kernel） | 不定 | 不定 |

**合计约 1.2~2.5 ms 额外开销**，恰好解释了你观察到的 2.543 ms 差距。

**核心问题**：在 NVIDIA GPU 上，ResNet-18 这类模型的单个 kernel 计算非常快（每个 <0.1 ms），此时模型变成了 **CPU 调度瓶颈型（dispatch-bound）**，GPU 频繁等待 CPU 提交下一个 kernel，导致纯 kernel 计算时间远小于端到端时延。

---

## nn-Meter 的方案对比

研究了 nn-Meter 源码后，核心差异：

### nn-Meter 的做法
1. **每个 kernel 作为独立小模型单独运行**（类似你的 `ort_kernel_runner.py`），测量端到端时延
2. 这个时延 **自然包含了所有框架开销**（session.run 的调度、launch、同步等）
3. 用 RandomForest 回归模型为每种 kernel 类型单独训练预测器
4. **整模型延迟 = 所有 kernel 预测延迟之和**（简单求和，见 `predict_by_kernel.py`）

```45:55:/cyq/hase/nn-Meter/nn_meter/predictor/prediction/predict_by_kernel.py
def nn_predict(predictors, kernel_units):
    features = get_predict_features(kernel_units)
    py = predict_model(features, predictors)
    return py
```

### 关键启发

nn-Meter 在移动端 CPU/GPU 上达到 99% 准确率，是因为：
- 移动端 kernel 计算时间长，框架开销占比小
- 独立 kernel 的端到端测量自然包含了开销

但在你的场景（高性能 NVIDIA GPU + ORT）中：
- kernel 计算极快，框架开销占比很大
- 独立运行每个 kernel 又会**高估**（因为每个 kernel 都承担了一份 session 级开销）

---

## 针对 Hase 项目的具体建议

### 建议 1：修改 profiler 脚本统计「全节点时间」而非仅 `_kernel_time`

不仅统计 `_kernel_time`，还要统计从 `fence_before` 到 `fence_after` 的全跨度，以及节点间的空隙。这可以验证开销来源。

### 建议 2：你现有的 benchmark 体系已经是正确方向

你的 `debug.py` 和 `ort_kernel_runner.py` 的 `run_kernel()` 方式（每个 kernel 单独端到端测量）本质上就是 nn-Meter 的方式。**这种测量包含了框架开销，是正确的训练数据来源**。

但这引入了反向问题：独立 kernel 测量会**高估**（debug.py 的 KernelSum/Model ratio > 1），因为每个 kernel 都独立承担了一份 session 开销。

### 建议 3：引入模型级修正因子

在 Hase 推理阶段（Phase 7 `predict_model_latency.py`），建议：

```
predicted_model_latency = α × Σ(predicted_kernel_latency) + β
```

其中 `α` 是一个 < 1 的缩放系数（因为独立 kernel 高估了开销），`β` 是一个模型级偏置。`α` 和 `β` 可以通过少量模型的实际测量来回归拟合。

### 建议 4：也可考虑用 profiler 的「全节点时间」作为训练标签

修改数据收集方式：不用独立 kernel 测量，而是用完整模型的 profiler 数据中每个节点从 `fence_before` 到 `fence_after` 的时间跨度作为 kernel 延迟标签。这样 kernel 延迟天然包含了调度开销，Σkernel ≈ model_latency。

---

## 总结

| 方案 | Kernel 延迟来源 | 包含框架开销？ | 求和 vs 实际 |
|---|---|---|---|
| `_kernel_time` (当前脚本) | ORT profiler CUDA Events | ❌ | 严重低估 (~42%) |
| 独立 `run_kernel` (debug.py) | 端到端 session.run | ✅ 但高估 | 高估 (~150%+) |
| `fence_before` → `fence_after` 跨度 | profiler 全节点时间 | ✅ 部分 | 接近但仍有少量 gap |
| **nn-Meter 方式** | 独立端到端 + RF 预测 + 简单求和 | ✅ | 在移动端很准，GPU 上需修正 |

**我的建议是**：继续用你现有的独立 kernel benchmark 体系作为训练数据（和 nn-Meter 一致），然后在模型级推理时加一个线性修正因子 `α`。这是最小改动、最务实的方案。
