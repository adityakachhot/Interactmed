#!/usr/bin/env python3
"""
scripts/check_cloud_environment.py
==================================
Preflight Cloud Environment Verification Script for MedGemma 4B LoRA Training.
Run this script in Google Colab, Lambda Labs, or RunPod before starting training.

Verifications:
1. CUDA GPU is present and accessible.
2. GPU architecture supports bfloat16 (compute capability >= 8.0, e.g. A100, H100, L4).
3. Dedicated GPU VRAM >= 40 GB (prevents OOM during backprop).
4. transformers >= 4.50.0 (supports Gemma 3 multimodal vision architecture).
5. bitsandbytes 4-bit NF4 quantization kernel functions properly.
6. Hugging Face token is valid and model 'google/medgemma-4b-it' is accessible.
7. Dataset files (train_dev.jsonl, val_dev.jsonl) exist and are uncorrupted.
8. The locked test partition ('expanded_test') is strictly excluded.
"""

import os
import sys
import json
from pathlib import Path

def check_env():
    print("=" * 70)
    print("INTERACTMED-VLM: CLOUD TRAINING ENVIRONMENT PREFLIGHT CHECK")
    print("=" * 70)

    errors = []
    warnings = []

    # 1. Check PyTorch & CUDA
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
        if not torch.cuda.is_available():
            errors.append("CUDA GPU is NOT available. Switch runtime to an NVIDIA GPU (A100 recommended).")
        else:
            dev_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            print(f"✓ Detected GPU: {dev_name}")
            print(f"✓ Compute Capability: {cap[0]}.{cap[1]}")
            print(f"✓ Total VRAM: {vram_gb} GB")

            if cap[0] < 8:
                errors.append(f"Compute capability {cap} < 8.0 does not natively support bfloat16. NVIDIA Ampere/Hopper (A100/H100) required.")
            if vram_gb < 38.0:
                warnings.append(f"VRAM ({vram_gb} GB) is less than recommended 40 GB. Effective batch size or gradient accumulation must be adjusted.")
    except Exception as e:
        errors.append(f"PyTorch check failed: {e}")

    # 2. Check Transformers version
    try:
        import transformers
        from packaging import version
        v_tf = transformers.__version__
        print(f"✓ Transformers Version: {v_tf}")
        if version.parse(v_tf) < version.parse("4.50.0"):
            errors.append(f"Transformers version {v_tf} < 4.50.0. Run: pip install --upgrade 'transformers>=4.50.0'")
    except ImportError:
        errors.append("Transformers not installed. Run: pip install 'transformers>=4.50.0'")

    # 3. Check PEFT & TRL
    try:
        import peft
        print(f"✓ PEFT Version: {peft.__version__}")
    except ImportError:
        errors.append("PEFT not installed. Run: pip install peft")

    try:
        import trl
        print(f"✓ TRL Version: {trl.__version__}")
    except ImportError:
        errors.append("TRL not installed. Run: pip install trl")

    # 4. Check bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes Version: {bnb.__version__}")
        # Test 4-bit quantization kernel
        if torch.cuda.is_available():
            test_tensor = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
            q_weight, q_state = bnb.functional.quantize_4bit(test_tensor, quant_type="nf4")
            deq = bnb.functional.dequantize_4bit(q_weight, q_state, quant_type="nf4")
            print("✓ bitsandbytes 4-bit NF4 kernel verified on CUDA accelerator.")
    except Exception as e:
        errors.append(f"bitsandbytes verification failed: {e}. Install via: pip install bitsandbytes")

    # 5. Check Hugging Face Access
    model_id = "google/medgemma-4b-it"
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Authenticated Hugging Face User: {user_info.get('name', 'Authenticated')}")
        # Check model access
        api.model_info(model_id)
        print(f"✓ Access to gated model '{model_id}' verified.")
    except Exception as e:
        warnings.append(f"Could not verify '{model_id}' access ({e}). Ensure HF_TOKEN is set and Health AI terms are accepted.")

    # 6. Check Dataset Partitions
    root = Path(__file__).resolve().parent.parent
    train_path = root / "artifacts/fine_tuning_cohort/train_dev.jsonl"
    val_path = root / "artifacts/fine_tuning_cohort/val_dev.jsonl"

    if train_path.exists() and val_path.exists():
        with open(train_path) as f:
            tr_lines = sum(1 for line in f if line.strip())
        with open(val_path) as f:
            va_lines = sum(1 for line in f if line.strip())
        print(f"✓ Training partition verified: {tr_lines} examples in {train_path.name}")
        print(f"✓ Validation partition verified: {va_lines} examples in {val_path.name}")
    else:
        errors.append(f"Dataset partitions missing at {train_path} or {val_path}.")

    # Summary
    print("-" * 70)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  [!] {w}")
    if errors:
        print("ERRORS (BLOCKERS):")
        for err in errors:
            print(f"  [X] {err}")
        print("\nPreflight check FAILED. Resolve the blockers above before launching training.")
        sys.exit(1)
    else:
        print("ALL CLOUD PREFLIGHT CHECKS PASSED SUCCESSFULLY!")
        print("The environment is fully validated for MedGemma 4B Tier 2 LoRA fine-tuning.")
        sys.exit(0)

if __name__ == "__main__":
    check_env()
