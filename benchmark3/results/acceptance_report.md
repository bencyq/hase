# Benchmark3 验收报告

- 开始时间: 2026-03-03T06:31:34
- 结束时间: 2026-03-03T06:57:23
- 模型数量: 5

## 开销比例对比（高 -> 低）

1. /cyq/hase/model_zoo/models/mobilenet_v2_bs32_448x448.onnx | Overhead=13533.661468% | T_model=39.654559 ms | T_kernel_sum=5406.368317 ms
2. /cyq/hase/model_zoo/models/densenet121_bs32_224x224.onnx | Overhead=10009.579004% | T_model=27.433214 ms | T_kernel_sum=2773.382458 ms
3. /cyq/hase/model_zoo/models/vgg16_bs64_224x224.onnx | Overhead=6744.580138% | T_model=50.168124 ms | T_kernel_sum=3433.797427 ms
4. /cyq/hase/model_zoo/models/YOLOv8m_bs64_448x448.onnx | Overhead=5796.797646% | T_model=221.517459 ms | T_kernel_sum=13062.436304 ms
5. /cyq/hase/model_zoo/models/resnet18_bs64_224x224.onnx | Overhead=4209.369051% | T_model=12.444931 ms | T_kernel_sum=536.298016 ms

## 简要分析

- 开销最高: /cyq/hase/model_zoo/models/mobilenet_v2_bs32_448x448.onnx (13533.661468%)
- 开销最低: /cyq/hase/model_zoo/models/resnet18_bs64_224x224.onnx (4209.369051%)
- 可能原因: 内核数量越多、逐个调用的调度/启动开销越容易累积；访存密集层较多时，分段执行也可能放大额外开销。

## 验收结果

- 5 个模型是否全部完成: 是
- 每个模型是否都有 Overhead_Ratio: 是
- 是否产出对比结论: 是
