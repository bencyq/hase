hase/
│
├── benchmark/                     # 数据集采集模块（真实GPU实验）
│   ├── collector.py               # 在不同背景GPU负载的情况下运行kernel，采集执行时间 + 采集DCGM指标，生成训练数据集
│   ├── stressor.py                # 已有：制造背景GPU负载
│   ├── ort_kernel_runner.py       # 运行单kernel
│   └── config.yaml                # benchmark参数配置
│
├── model_zoo/                     # 模型来源与导出
│   ├── torchvision_exporter.py    # pytorch → onnx → ort支持的优化格式
│   └── models/                    # 导出的onnx模型
│
├── ort_analysis/                  # ORT图与kernel切分分析
│   ├── ort_graph_parser.py        # 解析ORT优化后图
│   ├── fusion_detector.py         # 检测算子融合策略，存储融合后的kernel
│   ├── ort_kernel_record/         # 存放每个模型ort优化后的内核结构以及内核融合策略
│
├── kernel_model/                  # Kernel级时间预测模型
│   └── kernel_builder.py          # 生成kernel文件
│
├── graph_model/                   # 图级性能预测
│   ├── dag_critical_path.py       # DAG关键路径分析
│   └── overlap_simulator.py       # kernel并行重叠建模
│
├── training/                      # 训练流程
│   └── train_kernel_model.py
│
├── inference/                     # 推理时间预测入口
|   ├──evaluator.py
│   └──predict_model_latency.py
│
├── utils/
│   └── logger.py
│
└── README.md
