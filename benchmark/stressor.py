import datetime
import torch
import threading
import time

class GPUQuantitativeStressor:
    def __init__(self, device_id=0):
        self.device = torch.device(f"cuda:{device_id}")
        self.properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.properties.multi_processor_count
        # max_threads_per_multi_processor is not available in some PyTorch versions
        # Default to 2048 which is standard for most modern NVIDIA GPUs (CC 7.0+)
        self.max_threads_per_sm = 2048
        
        self.running = False
        self.stream_compute = torch.cuda.Stream(device=self.device)
        self.stream_mem = torch.cuda.Stream(device=self.device)
        
        # 预分配显存 (1GB) 用于 DRAM 压力测试，确保穿透 L2 Cache
        self.dram_tensor_src = torch.rand(1024 * 1024 * 256, device=self.device, dtype=torch.float32)
        self.dram_tensor_dst = torch.empty_like(self.dram_tensor_src)
        
        # 负载参数
        self.sm_active_ratio = 0.0
        self.sm_occupancy_ratio = 0.0
        self.dram_active_ratio = 0.0

    def start(self):
        self.running = True
        self.t_compute = threading.Thread(target=self._run_compute_load)
        self.t_mem = threading.Thread(target=self._run_mem_load)
        self.t_compute.start()
        self.t_mem.start()

    def stop(self):
        self.running = False
        self.t_compute.join()
        self.t_mem.join()

    def set_targets(self, sm_active=0.0, sm_occ=0.0, dram=0.0):
        """
        sm_active: 0.0-1.0 (控制活跃的SM比例)
        sm_occ: 0.0-1.0 (控制每个活跃SM内的Warp占用率)
        dram: 0.0-1.0 (控制显存带宽占用率)
        """
        self.sm_active_ratio = sm_active
        self.sm_occupancy_ratio = sm_occ
        self.dram_active_ratio = dram
        print(f"正在启动背景负载: SM_ACTIVE={sm_active}, SM_OCCUPANCY={sm_occ}, DRAM={dram}")
        print(datetime.datetime.now())

    def _run_compute_load(self):
        """
        通过调节 Grid Size 和 Block Size 定量控制 SM 活跃度和占用率
        """
        with torch.cuda.stream(self.stream_compute):
            while self.running:
                if self.sm_active_ratio > 0:
                    # 1. 计算需要的 Block 数量 (映射 SM_ACTIVE)
                    # 每个 SM 至少分配 1 个 Block 即可使该 SM 变为 ACTIVE
                    active_blocks = int(self.num_sms * self.sm_active_ratio)
                    active_blocks = max(1, active_blocks)
                    
                    # 2. 计算每个 Block 的线程数 (映射 SM_OCCUPANCY)
                    # 注意：实际 Occupancy 受寄存器和共享内存限制，这里简单通过线程数模拟
                    threads_per_block = int(self.max_threads_per_sm * self.sm_occupancy_ratio)
                    # Single block cannot exceed 1024 threads
                    threads_per_block = min(1024, threads_per_block)
                    threads_per_block = max(32, (threads_per_block // 32) * 32) # Warp 对齐
                    
                    # 3. 启动一个极小但高频的计算任务
                    # 使用 torch.mm 模拟计算压力
                    # 矩阵大小经过特殊设计，使其刚好能被分配到指定的 Block 数量上
                    size = 64 # 小矩阵，快速完成
                    a = torch.randn(active_blocks, size, size, device=self.device)
                    b = torch.randn(active_blocks, size, size, device=self.device)
                    
                    # 连续提交一批任务减少 Python 调度开销
                    for _ in range(50):
                        torch.bmm(a, b) 
                    
                else:
                    time.sleep(0.01)
                
                # 这里的同步是为了防止 Python 提交过快导致 OOM，但不影响 GPU 内部并行
                self.stream_compute.synchronize()

    def _run_mem_load(self):
        """
        通过控制 D2D Copy 的频率来定量控制 DRAM_ACTIVE
        """
        with torch.cuda.stream(self.stream_mem):
            while self.running:
                if self.dram_active_ratio > 0:
                    # 根据目标比例决定“忙”与“闲”
                    # DRAM_ACTIVE 是时间片占比。我们通过连续拷贝 + 精确睡眠控制。
                    start_time = time.time()
                    
                    # 这里的 burst 决定了最小控制粒度
                    burst = 10 
                    for _ in range(burst):
                        self.dram_tensor_dst.copy_(self.dram_tensor_src)
                    
                    self.stream_mem.synchronize()
                    elapsed = time.time() - start_time
                    
                    # 计算需要 sleep 的时间以达到目标占空比
                    # ratio = elapsed / (elapsed + sleep) => sleep = elapsed * (1/ratio - 1)
                    if self.dram_active_ratio < 1.0:
                        sleep_time = elapsed * (1.0 / self.dram_active_ratio - 1.0)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                else:
                    time.sleep(0.01)

# --- 使用示例 ---
if __name__ == "__main__":
    stressor = GPUQuantitativeStressor(device_id=0)
    
    stressor.set_targets(sm_active=0, sm_occ=0, dram=0)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()
    
    stressor.set_targets(sm_active=0.3, sm_occ=0.3, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.5, sm_occ=0.3, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.9, sm_occ=0.3, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.3, sm_occ=0.5, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.3, sm_occ=0.7, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.3, sm_occ=0.9, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.3, sm_occ=0.3, dram=0.5)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()
    
    stressor.set_targets(sm_active=0.3, sm_occ=0.3, dram=0.7)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.3, sm_occ=0.3, dram=0.9)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.5, sm_occ=0.5, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.7, sm_occ=0.7, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()

    stressor.set_targets(sm_active=0.9, sm_occ=0.9, dram=0.3)
    stressor.start()
    time.sleep(30) # 维持一段时间观察 DCGM
    stressor.stop()