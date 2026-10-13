#!/usr/bin/env python3
"""
scripts/check_environment.py
============================
InteractMed-VLM Environment and Hardware Verification Script.

Inspects Python environment, dependency versions, Apple Silicon M3 hardware,
Apple Metal Performance Shaders (MPS) availability, and tensor operations.

Usage:
    python scripts/check_environment.py [--json-output docs/env_check.json]
"""

import argparse
import json
import os
import platform
import sys
import time
from typing import Any, Dict


def check_system_and_hardware() -> Dict[str, Any]:
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform_string": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024 ** 3), 2)
        info["available_ram_gb"] = round(mem.available / (1024 ** 3), 2)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except ImportError:
        info["ram_info"] = "psutil not available"
    return info


def check_python_environment() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "compiler": platform.python_compiler(),
        "prefix": sys.prefix,
        "is_conda": "conda" in sys.prefix or "miniforge" in sys.prefix,
    }


def check_dependencies() -> Dict[str, Any]:
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "albumentations",
        "timm",
        "transformers",
        "pandas",
        "matplotlib",
        "yaml",
        "PIL",
    ]
    results = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "Available (no __version__)")
            results[pkg] = {"status": "OK", "version": str(version)}
        except ImportError as e:
            results[pkg] = {"status": "MISSING", "error": str(e)}
        except Exception as e:
            results[pkg] = {"status": "ERROR", "error": str(e)}
    return results


def check_pytorch_devices() -> Dict[str, Any]:
    import torch

    device_info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "default_device": "cpu",
    }

    # Test CPU Tensor Allocation & Matmul
    t0 = time.perf_counter()
    x_cpu = torch.randn(1000, 1000, dtype=torch.float32, device="cpu")
    y_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time_ms = (time.perf_counter() - t0) * 1000
    device_info["cpu_benchmark_1000x1000_matmul_ms"] = round(cpu_time_ms, 2)

    # Test MPS Tensor Allocation & Matmul if available
    if torch.backends.mps.is_available():
        try:
            t0 = time.perf_counter()
            x_mps = torch.randn(1000, 1000, dtype=torch.float32, device="mps")
            y_mps = torch.matmul(x_mps, x_mps)
            torch.mps.synchronize()
            mps_time_ms = (time.perf_counter() - t0) * 1000
            device_info["mps_status"] = "FUNCTIONAL"
            device_info["mps_benchmark_1000x1000_matmul_ms"] = round(mps_time_ms, 2)
            device_info["speedup_mps_vs_cpu"] = round(cpu_time_ms / max(mps_time_ms, 1e-4), 2)
            device_info["recommended_device"] = "mps"
        except Exception as e:
            device_info["mps_status"] = f"FAILED: {e}"
            device_info["recommended_device"] = "cpu"
    else:
        device_info["mps_status"] = "NOT_AVAILABLE"
        device_info["recommended_device"] = "cpu"

    return device_info


def main():
    parser = argparse.ArgumentParser(description="Check InteractMed-VLM Environment")
    parser.add_argument("--json-output", default=None, help="Save report to JSON file.")
    args = parser.parse_args()

    print("=" * 80)
    print("INTERACTMED-VLM ENVIRONMENT & HARDWARE VERIFICATION")
    print("=" * 80)

    hw = check_system_and_hardware()
    print(f"Platform:      {hw.get('platform_string')}")
    print(f"Architecture:  {hw.get('architecture')}")
    print(f"Total Memory:  {hw.get('total_ram_gb', 'N/A')} GB")

    py = check_python_environment()
    print(f"Python Exec:   {py['executable']}")
    print(f"Python Ver:    {py['version']}")

    deps = check_dependencies()
    print("\nDependency Check:")
    all_deps_ok = True
    for k, v in deps.items():
        status_str = f"[{v['status']}]".ljust(11)
        ver_str = v.get("version") or v.get("error")
        print(f"  {status_str} {k.ljust(16)} : {ver_str}")
        if v["status"] != "OK":
            all_deps_ok = False

    print("\nPyTorch Device Acceleration:")
    devices = check_pytorch_devices()
    print(f"  PyTorch Version:  {devices['torch_version']}")
    print(f"  CUDA Available:   {devices['cuda_available']}")
    print(f"  Apple MPS Avail:  {devices['mps_available']} (Built: {devices['mps_built']})")
    print(f"  MPS Status:       {devices.get('mps_status')}")
    print(f"  CPU MatMul Time:  {devices.get('cpu_benchmark_1000x1000_matmul_ms')} ms")
    if "mps_benchmark_1000x1000_matmul_ms" in devices:
        print(f"  MPS MatMul Time:  {devices.get('mps_benchmark_1000x1000_matmul_ms')} ms (Speedup: {devices.get('speedup_mps_vs_cpu')}x)")
    print(f"  Recommended Dev:  {devices.get('recommended_device').upper()}")
    print("=" * 80)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw,
        "python": py,
        "dependencies": deps,
        "pytorch_devices": devices,
        "all_dependencies_passed": all_deps_ok,
    }

    if args.json_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_output)), exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved environment report to {args.json_output}")


if __name__ == "__main__":
    main()
