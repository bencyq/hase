import argparse
import datetime
import threading
import time

import torch
from torchvision import models


class GPUQuantitativeStressor:
    def __init__(self, device_id=0, max_streams=8):
        self.device = torch.device(f"cuda:{device_id}")
        self.running = False
        self.max_streams = max_streams

        # 负载参数
        self.sm_active_ratio = 0.0
        self.sm_occupancy_ratio = 0.0
        self.dram_active_ratio = 0.0

        # 模型池：无需权重，随机输入
        self.model_pool = {
            "compute": [
                models.resnet50(weights=None),
                models.vgg16(weights=None),
            ],
            "memory": [
                models.vgg11(weights=None),
                models.alexnet(weights=None),
            ],
            "light": [
                models.squeezenet1_1(weights=None),
                models.mobilenet_v2(weights=None),
            ],
        }
        for group in self.model_pool.values():
            for model in group:
                model.eval().to(self.device)

        # 预生成不同 batch 的输入张量，避免 CPU->GPU 传输干扰
        self.batch_sizes = [1, 2, 4, 8, 16]
        self.inputs = {}
        for bs in self.batch_sizes:
            self.inputs[bs] = torch.randn(
                bs, 3, 224, 224, device=self.device, dtype=torch.float32
            )

        self.workers = []
        self.streams = []

    def start(self):
        self.running = True
        self._spawn_workers()

    def stop(self):
        self.running = False
        for t in self.workers:
            t.join()
        for s in self.streams:
            s.synchronize()

    def set_targets(self, sm_active=0.0, sm_occ=0.0, dram=0.0):
        """
        sm_active: 0.0-1.0 (控制并行流数量)
        sm_occ: 0.0-1.0 (控制每次前向的计算强度/批大小)
        dram: 0.0-1.0 (控制 memory-heavy 模型占比)
        """
        self.sm_active_ratio = sm_active
        self.sm_occupancy_ratio = sm_occ
        self.dram_active_ratio = dram
        plan = self._build_plan()
        print(
            f"正在启动背景负载: SM_ACTIVE={sm_active}, "
            f"SM_OCCUPANCY={sm_occ}, DRAM={dram}"
        )
        print(f"并行流数={plan['active_streams']}, batch={plan['batch']}")
        print(f"模型比例(memory/compute/light)={plan['mix']}")
        print(datetime.datetime.now())

        if self.running:
            self._respawn_workers()

    def _build_plan(self):
        active_streams = int(round(self.max_streams * self.sm_active_ratio))
        active_streams = max(1, active_streams)

        idx = int(round(self.sm_occupancy_ratio * (len(self.batch_sizes) - 1)))
        idx = max(0, min(len(self.batch_sizes) - 1, idx))
        batch = self.batch_sizes[idx]

        mem_ratio = max(0.0, min(1.0, float(self.dram_active_ratio)))
        light_ratio = max(0.0, 1.0 - self.sm_occupancy_ratio) * 0.5
        compute_ratio = max(0.0, 1.0 - mem_ratio - light_ratio)
        if compute_ratio < 0:
            compute_ratio = 0.0

        return {
            "active_streams": active_streams,
            "batch": batch,
            "mix": (round(mem_ratio, 2), round(compute_ratio, 2), round(light_ratio, 2)),
        }

    def _respawn_workers(self):
        for t in self.workers:
            t.join()
        self.workers = []
        self.streams = []
        self._spawn_workers()

    def _spawn_workers(self):
        if self.sm_active_ratio <= 0:
            return

        plan = self._build_plan()
        num_workers = plan["active_streams"]
        mix = plan["mix"]
        batch = plan["batch"]

        # 依据 mix 为每个 worker 选定模型类型
        for i in range(num_workers):
            tag = self._choose_model_tag(i, num_workers, mix)
            model = self.model_pool[tag][i % len(self.model_pool[tag])]
            stream = torch.cuda.Stream(device=self.device)
            t = threading.Thread(
                target=self._run_worker, args=(model, stream, batch), daemon=True
            )
            self.workers.append(t)
            self.streams.append(stream)
            t.start()

    def _choose_model_tag(self, idx, total, mix):
        mem_ratio, compute_ratio, light_ratio = mix
        mem_count = int(round(total * mem_ratio))
        compute_count = int(round(total * compute_ratio))
        if idx < mem_count:
            return "memory"
        if idx < mem_count + compute_count:
            return "compute"
        return "light"

    def _run_worker(self, model, stream, batch):
        inp = self.inputs[batch]
        with torch.cuda.stream(stream), torch.no_grad():
            while self.running:
                _ = model(inp)
                # 控制节奏，避免 CPU 端空转过多
                if self.sm_occupancy_ratio <= 0.1:
                    time.sleep(0.001)


def _clamp_ratio(x):
    return max(0.0, min(1.0, float(x)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Stressor")
    parser.add_argument("--device", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--sm-active", type=float, default=0.0, help="SM_ACTIVE (0.0-1.0)")
    parser.add_argument("--sm-occ", type=float, default=0.0, help="SM_OCCUPANCY (0.0-1.0)")
    parser.add_argument("--dram", type=float, default=0.0, help="DRAM_ACTIVE (0.0-1.0)")
    parser.add_argument("--duration", type=float, default=10.0, help="持续时间(秒)，0 表示一直运行")
    args = parser.parse_args()

    stressor = GPUQuantitativeStressor(device_id=args.device)
    stressor.set_targets(
        sm_active=_clamp_ratio(args.sm_active),
        sm_occ=_clamp_ratio(args.sm_occ),
        dram=_clamp_ratio(args.dram),
    )
    stressor.start()

    try:
        if args.duration and args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        stressor.stop()