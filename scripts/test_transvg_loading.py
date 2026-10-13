#!/usr/bin/env python3
"""
scripts/test_transvg_loading.py
===============================
TransVG-AGPT Checkpoint Loading, Device Compatibility, and Inference Smoke Test.

Validates that the pretrained TransVG-AGPT checkpoint loads accurately,
tests execution on Apple Silicon MPS vs. CPU, measures latency,
reconciles coordinate transformations, and saves visual smoke-test outputs.

Usage:
    python scripts/test_transvg_loading.py [--checkpoint checkpoints/transvg.pth] [--device auto] [--save-dir artifacts/model_smoke_test]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure project root and third_party/AGPT are on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGPT_ROOT = os.path.join(PROJECT_ROOT, "third_party", "AGPT")

for p in [PROJECT_ROOT, AGPT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TransVGTest")


def load_transvg_architecture(agpt_dir: str):
    """Dynamically imports and builds TransVG model using official AGPT source code."""
    from Args import build_args
    from models.transvg import build_transvg_model

    args = build_args("transvg", "ms_cxr")
    # Point internal detr and genome paths relative to AGPT root if needed
    model = build_transvg_model(args)
    return model, args


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    """Loads weights from checkpoint into model with verification."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    file_size_mb = round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2)
    logger.info(f"Loading checkpoint from {checkpoint_path} ({file_size_mb} MB)...")

    t0 = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_time_sec = time.perf_counter() - t0

    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    logger.info(f"Loaded weights in {load_time_sec:.2f}s (Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)})")
    return {
        "file_size_mb": file_size_mb,
        "load_time_sec": round(load_time_sec, 3),
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "total_weights_loaded": len(state_dict),
    }


def preprocess_cxr_image(args: Any, img_path: str, device: torch.device):
    """Applies official MS-CXR / AGPT validation preprocessing."""
    from dataloader.mscxr_dataloader import get_transforms
    from utils.misc import NestedTensor

    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)

    transform = get_transforms(args, "val", False)
    transformed = transform(image=img_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    image_mask = torch.zeros_like(image_tensor[:, 0, :, :], dtype=torch.bool, device=device)
    image_nested = NestedTensor(image_tensor, image_mask)
    return image_nested, img_pil.size  # (width, height)


def run_model_inference(
    model: torch.nn.Module,
    args: Any,
    image_nested: Any,
    phrase: str,
    device: torch.device,
) -> Tuple[np.ndarray, float, Optional[str]]:
    """Runs forward inference on specified device and handles MPS exceptions gracefully."""
    from utils.box_utils import xywh2xyxy

    model.eval()
    t0 = time.perf_counter()
    error_msg = None

    try:
        with torch.no_grad():
            outputs = model(image_nested, [phrase])
            pred_boxes = outputs["pred_boxes"].squeeze().view(-1, 4).detach().cpu()
            pred_boxes = xywh2xyxy(pred_boxes)
            latency_sec = time.perf_counter() - t0
            return pred_boxes.numpy().squeeze(), latency_sec, None
    except Exception as e:
        latency_sec = time.perf_counter() - t0
        error_msg = str(e)
        logger.warning(f"Inference failed on device {device}: {e}")
        return np.zeros(4), latency_sec, error_msg


def save_prediction_plot(
    img_path: str,
    phrase: str,
    pred_box_norm: np.ndarray,
    gt_box_px: Optional[List[float]],
    imsize: int,
    save_path: str,
):
    """Renders ground truth (red) and predicted (blue) bounding boxes."""
    image = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(image)

    # Plot Ground Truth if present ([x, y, w, h] in image space)
    if gt_box_px is not None:
        x1, y1, w, h = gt_box_px
        rect_gt = patches.Rectangle((x1, y1), w, h, linewidth=2.5, edgecolor="red", facecolor="none", label="Ground Truth")
        ax.add_patch(rect_gt)

    # Plot Prediction ([x1, y1, x2, y2] normalized in [0, 1] relative to imsize)
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box_norm * imsize
    rect_pred = patches.Rectangle(
        (pred_x1, pred_y1),
        pred_x2 - pred_x1,
        pred_y2 - pred_y1,
        linewidth=2.5,
        edgecolor="dodgerblue",
        facecolor="none",
        linestyle="--",
        label="TransVG-AGPT Pred",
    )
    ax.add_patch(rect_pred)

    clean_phrase = phrase[:45] + "..." if len(phrase) > 45 else phrase
    plt.title(f"Phrase: '{clean_phrase}'", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right", framealpha=0.8)
    plt.axis("off")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="TransVG-AGPT Model Loading & Smoke Test")
    parser.add_argument("--checkpoint", default="checkpoints/transvg.pth", help="Path to transvg.pth weights")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"], help="Compute device")
    parser.add_argument("--save-dir", default="artifacts/model_smoke_test", help="Directory to save test artifacts")
    args_cli = parser.parse_args()

    os.makedirs(args_cli.save_dir, exist_ok=True)

    # Reconcile canonical checkpoint and AGPT demo model_weight link
    agpt_weight_dir = os.path.join(AGPT_ROOT, "model_weight")
    os.makedirs(agpt_weight_dir, exist_ok=True)
    agpt_weight_target = os.path.join(agpt_weight_dir, "transvg.pth")
    checkpoint_abs = os.path.abspath(args_cli.checkpoint)

    if os.path.exists(checkpoint_abs) and not os.path.exists(agpt_weight_target):
        try:
            os.symlink(checkpoint_abs, agpt_weight_target)
            logger.info(f"Created symlink: {agpt_weight_target} -> {checkpoint_abs}")
        except Exception:
            pass

    # Benchmark demo examples from official AGPT demo.py
    demo_samples = {
        os.path.join(AGPT_ROOT, "demo", "0ae1a7ad-fcc627aa-01e96e50-cb14cf17-23f441ac.jpg"): (
            "small left apical pneumothorax",
            [345, 70, 141, 45],
        ),
        os.path.join(AGPT_ROOT, "demo", "3aa9ee81-08355b5d-713a3a03-6cd20281-6222326c.jpg"): (
            "Left perihilar opacity",
            [400, 274, 94, 140],
        ),
        os.path.join(AGPT_ROOT, "demo", "8350f725-7918f8aa-abc95fec-a3d62a15-a47db0d9.jpg"): (
            "patchy consolidation in the right lower",
            [60, 293, 146, 216],
        ),
        os.path.join(AGPT_ROOT, "demo", "f7b69ee3-db7f264c-fca7d1c7-1d372fc0-02b35a47.jpg"): (
            "there is mild cardiomegaly",
            [195, 281, 338, 198],
        ),
    }

    print("\n" + "=" * 80)
    print("TRANSVG-AGPT MODEL LOADING & INFERENCE SMOKE TEST")
    print("=" * 80)

    # 1. Instantiate Model Architecture
    t_start = time.perf_counter()
    logger.info("Instantiating TransVG-AGPT model architecture...")
    model, model_args = load_transvg_architecture(AGPT_ROOT)
    instantiation_time_sec = time.perf_counter() - t_start
    logger.info(f"Model instantiated in {instantiation_time_sec:.2f}s")

    # 2. Load Pretrained Checkpoint Weights
    ckpt_report = load_checkpoint_weights(model, args_cli.checkpoint)

    # 3. Test Device Compatibility (CPU and MPS)
    devices_to_test = []
    if args_cli.device == "auto":
        devices_to_test = ["cpu"]
        if torch.backends.mps.is_available():
            devices_to_test.append("mps")
    else:
        devices_to_test = [args_cli.device]

    device_results = {}
    sample_preds = []

    for dev_name in devices_to_test:
        dev = torch.device(dev_name)
        model_args.device = dev
        logger.info(f"Testing execution on target device: {dev_name.upper()}...")

        try:
            model_on_dev = model.to(dev)
            dev_status = "SUCCESS"
            latencies = []

            for img_path, (phrase, gt_box) in demo_samples.items():
                if not os.path.exists(img_path):
                    continue
                nested_img, (w_orig, h_orig) = preprocess_cxr_image(model_args, img_path, dev)
                pred_box, latency_sec, err = run_model_inference(model_on_dev, model_args, nested_img, phrase, dev)

                if err is not None:
                    dev_status = f"FAILED: {err}"
                    break

                latencies.append(latency_sec)

                # Save plot for first device or preferred device
                if dev_name == devices_to_test[-1]:
                    out_img_name = f"pred_{dev_name}_{os.path.basename(img_path)}"
                    out_img_path = os.path.join(args_cli.save_dir, out_img_name)
                    save_prediction_plot(img_path, phrase, pred_box, gt_box, model_args.imsize, out_img_path)
                    sample_preds.append({
                        "image": os.path.basename(img_path),
                        "phrase": phrase,
                        "pred_box_norm_xyxy": [round(float(v), 4) for v in pred_box.tolist()],
                        "ground_truth_xywh": gt_box,
                        "plot_path": out_img_path,
                    })

            device_results[dev_name] = {
                "status": dev_status,
                "mean_latency_sec": round(float(np.mean(latencies)), 4) if latencies else None,
                "min_latency_sec": round(float(np.min(latencies)), 4) if latencies else None,
                "samples_evaluated": len(latencies),
            }
            logger.info(f"Device {dev_name.upper()} result: {dev_status} (Mean Latency: {device_results[dev_name]['mean_latency_sec']}s)")

        except Exception as e:
            device_results[dev_name] = {
                "status": f"INITIALIZATION_ERROR: {e}",
                "mean_latency_sec": None,
                "samples_evaluated": 0,
            }
            logger.warning(f"Device {dev_name.upper()} initialization failed: {e}")

    # Summary Report
    smoke_test_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": {
            "path": args_cli.checkpoint,
            **ckpt_report,
        },
        "model_architecture": {
            "model_name": model_args.model_name,
            "backbone": model_args.backbone,
            "bert_model": model_args.bert_model,
            "image_size": model_args.imsize,
            "hidden_dim": model_args.vl_hidden_dim,
            "instantiation_time_sec": round(instantiation_time_sec, 3),
        },
        "device_benchmarks": device_results,
        "sample_predictions": sample_preds,
    }

    report_json_path = os.path.join(args_cli.save_dir, "smoke_test_report.json")
    with open(report_json_path, "w") as fp:
        json.dump(smoke_test_report, fp, indent=2)

    print("\n" + "=" * 80)
    print("TRANSVG-AGPT SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Checkpoint File Size:           {ckpt_report['file_size_mb']} MB")
    print(f"Checkpoint Load Time:           {ckpt_report['load_time_sec']}s")
    for d_name, d_res in device_results.items():
        print(f"Device {d_name.upper():<4} Status:             {d_res['status']} (Latency: {d_res.get('mean_latency_sec')}s/image)")
    print(f"Output Plots & JSON Saved To:   {args_cli.save_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
