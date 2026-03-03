#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import subprocess
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


def gpu_mem_used_mb(gpu_index: int) -> float:
    cmd = [
        "nvidia-smi",
        f"--query-gpu=memory.used",
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
    baseline = gpu_mem_used_mb(gpu_index)
    peak = baseline
    status = "ok"
    error = ""
    iterations = 0
    input_name = ""
    input_type = ""
    input_shape = ""

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

        for _ in range(3):
            session.run(None, feed)

        end_time = time.time() + seconds
        next_sample = time.time()
        while time.time() < end_time:
            session.run(None, feed)
            iterations += 1
            now = time.time()
            if now >= next_sample:
                current = gpu_mem_used_mb(gpu_index)
                if current > peak:
                    peak = current
                next_sample = now + sample_interval
    except Exception as exc:
        status = "failed"
        error = str(exc).replace("\n", " ")[:300]
    finally:
        try:
            del session
        except Exception:
            pass
        gc.collect()
        time.sleep(0.3)
        final_sample = gpu_mem_used_mb(gpu_index)
        if final_sample > peak:
            peak = final_sample

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
    print(f"Per model runtime: {args.seconds}s")
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

        for i, model_path in enumerate(model_paths, start=1):
            start = time.time()
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
            if proc.returncode == 0:
                lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
                if not lines:
                    result = {
                        "input_name": "",
                        "input_type": "",
                        "input_shape": "",
                        "iterations": 0,
                        "baseline_gpu_mem_mb": idle_mem,
                        "peak_gpu_mem_mb": idle_mem,
                        "peak_delta_mb": 0.0,
                        "status": "failed",
                        "error": "child process returned empty output",
                    }
                else:
                    try:
                        result = json.loads(lines[-1])
                    except json.JSONDecodeError:
                        result = {
                            "input_name": "",
                            "input_type": "",
                            "input_shape": "",
                            "iterations": 0,
                            "baseline_gpu_mem_mb": idle_mem,
                            "peak_gpu_mem_mb": idle_mem,
                            "peak_delta_mb": 0.0,
                            "status": "failed",
                            "error": "child process output is not valid JSON",
                        }
            else:
                result = {
                    "input_name": "",
                    "input_type": "",
                    "input_shape": "",
                    "iterations": 0,
                    "baseline_gpu_mem_mb": idle_mem,
                    "peak_gpu_mem_mb": idle_mem,
                    "peak_delta_mb": 0.0,
                    "status": "failed",
                    "error": proc.stderr.replace("\n", " ")[:300],
                }

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
            print(
                f"[{i}/{len(model_paths)}] {model_path.name} "
                f"| idle={idle_mem:.1f} MB base={float(result['baseline_gpu_mem_mb']):.1f} MB "
                f"peak={float(result['peak_gpu_mem_mb']):.1f} MB delta={float(result['peak_delta_mb']):.1f} MB "
                f"| iters={result['iterations']} | {result['status']} | {elapsed:.1f}s"
            )

    print("Done.")


if __name__ == "__main__":
    main()
