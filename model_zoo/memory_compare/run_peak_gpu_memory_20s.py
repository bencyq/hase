#!/usr/bin/env python3
import argparse
import ctypes
import csv
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


class _NvmlMemInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


_nvml_lib = None
_nvml_handle = None


def _nvml_init(gpu_index: int) -> bool:
    global _nvml_lib, _nvml_handle
    if _nvml_lib is not None:
        return True
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        lib.nvmlInit()
        handle = ctypes.c_void_p()
        lib.nvmlDeviceGetHandleByIndex(gpu_index, ctypes.byref(handle))
        _nvml_lib = lib
        _nvml_handle = handle
        return True
    except Exception:
        return False


def gpu_mem_used_mb(gpu_index: int) -> float:
    if _nvml_lib is not None:
        info = _NvmlMemInfo()
        _nvml_lib.nvmlDeviceGetMemoryInfo(_nvml_handle, ctypes.byref(info))
        return info.used / (1024.0 * 1024.0)
    # Fallback: nvidia-smi subprocess
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    line = proc.stdout.strip().splitlines()[0].strip()
    try:
        return float(line)
    except ValueError:
        return 0.0


def normalize_shape(shape):
    out = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            out.append(dim)
        else:
            out.append(1)
    return out


def build_input(onnx_type: str, shape):
    dtype = DTYPE_MAP.get(onnx_type, np.float32)
    if dtype == np.bool_:
        return (np.random.rand(*shape) > 0.5).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 10, size=shape, dtype=dtype)
    return np.random.rand(*shape).astype(dtype)


def wait_for_gpu_idle(
    gpu_index: int, threshold_mb: float, stable_samples: int, timeout_s: float, poll_interval_s: float
) -> float:
    deadline = time.time() + timeout_s
    stable = 0
    last_mem = gpu_mem_used_mb(gpu_index)
    while time.time() < deadline:
        last_mem = gpu_mem_used_mb(gpu_index)
        if last_mem <= threshold_mb:
            stable += 1
            if stable >= stable_samples:
                return last_mem
        else:
            stable = 0
        time.sleep(poll_interval_s)
    return last_mem


def run_single_model(model_path: Path, seconds: float, gpu_index: int, sample_interval: float):
    _nvml_init(gpu_index)

    # Robust baseline: 3 samples, take minimum
    baseline = min(gpu_mem_used_mb(gpu_index) for _ in range(3))
    peak = baseline
    status = "ok"
    error = ""
    iterations = 0
    input_name = ""
    input_type = ""
    input_shape = ""
    stop_event = threading.Event()

    def _sampler(holder: list):
        while not stop_event.is_set():
            mem = gpu_mem_used_mb(gpu_index)
            if mem > holder[0]:
                holder[0] = mem
            stop_event.wait(sample_interval)

    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        inp = session.get_inputs()[0]
        input_name = inp.name
        input_type = inp.type
        shape = normalize_shape(inp.shape)
        input_shape = "x".join(str(x) for x in shape)

        data = build_input(input_type, shape)
        feed = {input_name: data}

        for _ in range(5):
            session.run(None, feed)

        # Reset peak to steady-state after warmup, then start background sampler
        peak = gpu_mem_used_mb(gpu_index)
        peak_holder = [peak]
        sampler_thread = threading.Thread(target=_sampler, args=(peak_holder,), daemon=True)
        sampler_thread.start()

        end_time = time.time() + seconds
        while time.time() < end_time:
            session.run(None, feed)
            iterations += 1

        stop_event.set()
        sampler_thread.join(timeout=2.0)
        peak = peak_holder[0]
    except Exception as exc:
        stop_event.set()
        status = "failed"
        error = str(exc).replace("\n", " ")[:300]
    finally:
        stop_event.set()
        try:
            del session
        except Exception:
            pass
        gc.collect()

    peak_delta = max(0.0, peak - baseline)
    return {
        "model": model_path.name,
        "input_name": input_name,
        "input_type": input_type,
        "input_shape": input_shape,
        "run_seconds": seconds,
        "iterations": iterations,
        "baseline_gpu_mem_mb": baseline,
        "peak_gpu_mem_mb": peak,
        "peak_delta_mb": peak_delta,
        "status": status,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(description="Run all ONNX models and record 20s peak GPU memory.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("/cyq/hase/model_zoo/models"),
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/cyq/hase/model_zoo/memory_compare/peak_gpu_memory_20s.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--seconds", type=float, default=20.0, help="Runtime per model")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index")
    parser.add_argument("--sample-interval", type=float, default=0.2, help="Memory sample interval in seconds")
    parser.add_argument("--single-model", type=Path, default=None, help="Run one model and print JSON result")
    parser.add_argument(
        "--idle-mem-threshold-mb",
        type=float,
        default=256.0,
        help="GPU memory threshold to consider idle before next model",
    )
    parser.add_argument(
        "--idle-stable-samples",
        type=int,
        default=3,
        help="Consecutive idle samples required before next model",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=float,
        default=120.0,
        help="Max wait time for GPU idle before forcing next model",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Measure each model N times; record the minimum peak (reduces outliers)",
    )
    args = parser.parse_args()

    if args.single_model is not None:
        result = run_single_model(
            model_path=args.single_model,
            seconds=args.seconds,
            gpu_index=args.gpu_index,
            sample_interval=args.sample_interval,
        )
        print(json.dumps(result, ensure_ascii=True))
        return

    model_paths = sorted(args.models_dir.glob("*.onnx"))
    if not model_paths:
        raise SystemExit(f"No ONNX files found in {args.models_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"Total models: {len(model_paths)}")
    print(f"Models dir: {args.models_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Per model runtime: {args.seconds}s, repeat: {args.repeat}")
    print(f"GPU index: {args.gpu_index}")

    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "input_name",
                "input_type",
                "input_shape",
                "run_seconds",
                "iterations",
                "baseline_gpu_mem_mb",
                "peak_gpu_mem_mb",
                "peak_delta_mb",
                "status",
                "error",
            ]
        )

        def _parse_proc_result(proc, fallback_mem):
            if proc.returncode == 0:
                lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
                if lines:
                    try:
                        return json.loads(lines[-1])
                    except json.JSONDecodeError:
                        err = "child process output is not valid JSON"
                else:
                    err = "child process returned empty output"
            else:
                err = proc.stderr.replace("\n", " ")[:300]
            return {
                "input_name": "", "input_type": "", "input_shape": "",
                "iterations": 0,
                "baseline_gpu_mem_mb": fallback_mem,
                "peak_gpu_mem_mb": fallback_mem,
                "peak_delta_mb": 0.0,
                "status": "failed", "error": err,
            }

        for i, model_path in enumerate(model_paths, start=1):
            start = time.time()
            best_result = None

            for rep in range(args.repeat):
                idle_mem = wait_for_gpu_idle(
                    gpu_index=args.gpu_index,
                    threshold_mb=args.idle_mem_threshold_mb,
                    stable_samples=args.idle_stable_samples,
                    timeout_s=args.idle_timeout_s,
                    poll_interval_s=args.sample_interval,
                )
                cmd = [
                    "python",
                    str(Path(__file__).resolve()),
                    "--single-model",
                    str(model_path),
                    "--seconds",
                    str(args.seconds),
                    "--gpu-index",
                    str(args.gpu_index),
                    "--sample-interval",
                    str(args.sample_interval),
                ]
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=os.environ.copy())
                result = _parse_proc_result(proc, idle_mem)

                if best_result is None or (
                    result["status"] == "ok"
                    and float(result["peak_gpu_mem_mb"]) < float(best_result["peak_gpu_mem_mb"])
                ):
                    best_result = result

            result = best_result
            writer.writerow(
                [
                    model_path.name,
                    result["input_name"],
                    result["input_type"],
                    result["input_shape"],
                    args.seconds,
                    result["iterations"],
                    f"{float(result['baseline_gpu_mem_mb']):.1f}",
                    f"{float(result['peak_gpu_mem_mb']):.1f}",
                    f"{float(result['peak_delta_mb']):.1f}",
                    result["status"],
                    result["error"],
                ]
            )
            f.flush()
            elapsed = time.time() - start
            rep_info = f" (x{args.repeat})" if args.repeat > 1 else ""
            print(
                f"[{i}/{len(model_paths)}] {model_path.name} "
                f"| base={float(result['baseline_gpu_mem_mb']):.1f} MB "
                f"peak={float(result['peak_gpu_mem_mb']):.1f} MB delta={float(result['peak_delta_mb']):.1f} MB "
                f"| iters={result['iterations']} | {result['status']}{rep_info} | {elapsed:.1f}s"
            )

    print("Done.")


if __name__ == "__main__":
    main()
