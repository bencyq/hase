#!/bin/bash
# MPS 状态检查脚本

echo "=== 检查 MPS 环境变量 ==="
echo "CUDA_MPS_PIPE_DIRECTORY: ${CUDA_MPS_PIPE_DIRECTORY:-未设置}"
echo "CUDA_MPS_LOG_DIRECTORY: ${CUDA_MPS_LOG_DIRECTORY:-未设置}"

if [ -n "$CUDA_MPS_PIPE_DIRECTORY" ]; then
    echo ""
    echo "=== 检查 MPS 管道目录 ==="
    if [ -d "$CUDA_MPS_PIPE_DIRECTORY" ]; then
        echo "目录存在: $CUDA_MPS_PIPE_DIRECTORY"
        echo "目录内容:"
        ls -la "$CUDA_MPS_PIPE_DIRECTORY" 2>/dev/null || echo "无法列出目录内容"
    else
        echo "错误: 目录不存在: $CUDA_MPS_PIPE_DIRECTORY"
    fi
fi

echo ""
echo "=== 检查 CUDA 设备 ==="
python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'设备数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>&1
