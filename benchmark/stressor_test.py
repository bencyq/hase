import os
import time
import yaml
import datetime
from stressor import GPUQuantitativeStressor


def _load_config(config_path):
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_stress_levels(cfg):
    """从配置中提取 STRESS_LEVELS"""
    levels = cfg.get("STRESS_LEVELS") or []
    if not levels:
        raise ValueError("配置文件中未找到 STRESS_LEVELS")
    fixed = []
    for lv in levels:
        fixed.append({
            "sm_active": max(0.0, min(1.0, float(lv.get("sm_active", 0.0)))),
            "sm_occ": max(0.0, min(1.0, float(lv.get("sm_occ", 0.0)))),
            "dram": max(0.0, min(1.0, float(lv.get("dram", 0.0)))),
        })
    return fixed


# 配置文件路径
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# 加载配置
config = _load_config(CONFIG_PATH)
STRESS_LEVELS = _get_stress_levels(config)

# 测试配置
DEVICE_ID = 0
TEST_DURATION = 60  # 每个组合运行 30 秒
INTERVAL_BETWEEN_TESTS = 30  # 测试之间的间隔（秒）

if __name__ == "__main__":
    print(f"开始测试 stressor，共 {len(STRESS_LEVELS)} 个参数组合")
    print(f"每个组合运行 {TEST_DURATION} 秒，间隔 {INTERVAL_BETWEEN_TESTS} 秒")
    print("=" * 60)
    
    for idx, stress_config in enumerate(STRESS_LEVELS, 1):
        print(f"\n[{idx}/{len(STRESS_LEVELS)}] 测试参数组合:")
        print(f"  SM_ACTIVE={stress_config['sm_active']}, "
              f"SM_OCCUPANCY={stress_config['sm_occ']}, "
              f"DRAM={stress_config['dram']}")
        print(f"  开始时间: {datetime.datetime.now()}")
        
        # 创建并启动 stressor
        stressor = GPUQuantitativeStressor(device_id=DEVICE_ID)
        stressor.set_targets(
            sm_active=stress_config['sm_active'],
            sm_occ=stress_config['sm_occ'],
            dram=stress_config['dram']
        )
        stressor.start()
        
        try:
            # 运行指定时长
            time.sleep(TEST_DURATION)
        except KeyboardInterrupt:
            print("  收到中断信号，停止当前测试")
        finally:
            # 停止 stressor
            stressor.stop()
            print(f"  结束时间: {datetime.datetime.now()}")
            print(f"  测试完成")
        
        # 如果不是最后一个测试，等待间隔时间
        if idx < len(STRESS_LEVELS):
            print(f"\n等待 {INTERVAL_BETWEEN_TESTS} 秒后开始下一个测试...")
            time.sleep(INTERVAL_BETWEEN_TESTS)
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
