#!/usr/bin/env python3
"""
scripts/train_multipathology_continuation.py
============================================
Controlled Multi-Pathology Continuation Training for InteractMed-VLM (Task 3).

Continues fine-tuning from Task 2 Epoch 3 best checkpoint (Val mIoU = 0.6510)
for up to 3 additional epochs (Epochs 4, 5, 6) with mandatory early stopping:
- Early stopping criterion: Stop after 2 consecutive epochs without >0.001
  improvement over the best validation mIoU.
- Conservative starting LR: 1.0e-5 decaying toward 1.0e-6 via cosine schedule.
- Separate artifacts and checkpoint directory:
  checkpoints/multipathology/run_02_continuation_to_6ep/
- Evaluates on MS-CXR validation split (147 annotations) after every epoch.

Usage:
  # Preflight smoke test:
  /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_multipathology_continuation.py --smoke-test

  # Full continuation experiment:
  /Users/adityakachhot/miniforge3/envs/dl/bin/python scripts/train_multipathology_continuation.py --config configs/multipathology_continuation_to_6ep.yaml
"""

import argparse
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGPT_ROOT = PROJECT_ROOT / "third_party" / "AGPT"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGPT_ROOT))

from Args import build_args
from models.transvg import build_transvg_model
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.misc import NestedTensor

from scripts.train_multipathology import (
    _load_dicom_rgb,
    _load_jpeg_rgb,
    MultiPathologyDataset,
    multipathology_collate_fn,
    dynamic_best_match_loss,
    evaluate_val_dataset,
    prepare_multipathology_data,
)


def run_continuation(cfg: dict, args_cli: argparse.Namespace) -> None:
    req_device = args_cli.device or cfg["model"].get("device", "auto")
    if req_device == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device(req_device)

    print(f"\n=======================================================")
    print(f"Task 3: Controlled Multi-Pathology Continuation Training")
    print(f"Device: {device}")
    print(f"=======================================================")

    imsize = cfg["data"].get("imsize", 640)
    agpt_args = build_args("transvg", "ms_cxr")
    agpt_args.device = device
    agpt_args.imsize = imsize

    print(f"[INFO] Initializing TransVG-AGPT architecture ...")
    model = build_transvg_model(agpt_args)

    base_ckpt = Path(cfg["model"]["base_checkpoint"])
    if not base_ckpt.is_file():
        raise FileNotFoundError(f"Base checkpoint for continuation not found: {base_ckpt}")
    print(f"[INFO] Loading Task 2 Epoch 3 checkpoint: {base_ckpt}")
    ckp = torch.load(str(base_ckpt), map_location="cpu")
    model.load_state_dict(ckp["model"], strict=False)

    # Freeze visual CNN and ClinicalBERT encoders
    if cfg["model"].get("freeze_encoders", True):
        print(f"[INFO] Freezing encoders: Locking visual backbone and ClinicalBERT ...")
        model.visumodel.eval()
        for p in model.visumodel.parameters():
            p.requires_grad = False
        model.textmodel.eval()
        for p in model.textmodel.parameters():
            p.requires_grad = False

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable_count:,} / {total_params:,} ({trainable_count / total_params * 100:.2f}%)")

    # Load data using identical preparation pipeline as Task 2
    train_samples, val_samples = prepare_multipathology_data(
        local_verified_csv=Path(cfg["data"]["local_verified_csv"]),
        mimic_base_dir=Path(cfg["data"]["mimic_base_dir"]),
        rsna_train_csv=Path(cfg["data"]["rsna_train_csv"]),
        rsna_images_dir=Path(cfg["data"]["rsna_images_dir"]),
        rsna_sample_size=cfg["data"].get("rsna_train_sample_size", 200),
    )

    starting_epoch = cfg["training"].get("starting_epoch", 4)
    additional_epochs = cfg["training"].get("additional_epochs", 3)
    max_epochs = cfg["experiment"].get("max_epochs", 6)

    if args_cli.smoke_test:
        train_samples = train_samples[:20]
        val_samples = val_samples[:10]
        additional_epochs = 1
        print(f"[SMOKE TEST] Limited to {len(train_samples)} train and {len(val_samples)} val samples.")

    print(f"[INFO] Training cohort size  : {len(train_samples)} unique query items")
    print(f"[INFO] Validation cohort size: {len(val_samples)} unique query items")
    print(f"[INFO] Epoch range          : Epoch {starting_epoch} to max Epoch {starting_epoch + additional_epochs - 1}")

    train_dataset = MultiPathologyDataset(train_samples, imsize=imsize, is_training=True)
    val_dataset = MultiPathologyDataset(val_samples, imsize=imsize, is_training=False)

    batch_size = args_cli.batch_size or cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=multipathology_collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=multipathology_collate_fn,
        drop_last=False,
    )

    # Continuation optimizer: conservative lr=1e-5
    lr = float(cfg["training"].get("lr", 1.0e-5))
    weight_decay = float(cfg["training"].get("weight_decay", 1.0e-4))
    clip_max_norm = float(cfg["training"].get("clip_max_norm", 0.1))
    l1_coef = float(cfg["training"].get("l1_loss_coef", 5.0))
    giou_coef = float(cfg["training"].get("giou_loss_coef", 2.0))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * additional_epochs
    min_lr = float(cfg["training"].get("min_lr", 1.0e-6))

    # Cosine schedule decaying from 1.0e-5 to 1.0e-6 across remaining epochs
    def lr_lambda(current_step: int) -> float:
        progress = float(current_step) / float(max(1, total_steps))
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping parameters
    es_cfg = cfg["training"].get("early_stopping", {})
    min_improvement = float(es_cfg.get("min_improvement", 0.001))
    patience = int(es_cfg.get("patience", 2))
    max_drop_pause = float(es_cfg.get("max_drop_pause", 0.03))
    baseline_best_val_iou = float(es_cfg.get("baseline_best_val_iou", 0.6510))

    best_val_iou = baseline_best_val_iou
    best_epoch = 3  # Task 2 baseline best
    best_checkpoint_path = str(base_ckpt)
    consecutive_no_improvement = 0

    # Output setup
    out_ckpt_dir = Path(cfg["output"]["checkpoints_dir"])
    out_art_dir = Path(cfg["output"]["artifacts_dir"])
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_art_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "start_time": datetime.now().isoformat(),
        "experiment_name": cfg["experiment"]["name"],
        "continuation_source": str(base_ckpt),
        "baseline_best_val_iou": baseline_best_val_iou,
        "early_stopping_policy": {
            "min_improvement": min_improvement,
            "patience": patience,
            "max_drop_pause": max_drop_pause,
        },
        "config": cfg,
        "epochs": [],
        "stopping_summary": {},
    }

    print(f"\n[INFO] Starting continuation from Epoch {starting_epoch} (Baseline Best Val IoU = {best_val_iou:.4f})")
    print(f"[INFO] Early stopping policy: Patience = {patience} epochs without > {min_improvement} improvement.")

    global_step = 0
    stopped_early = False
    stop_reason = ""

    for epoch_idx in range(additional_epochs):
        current_epoch = starting_epoch + epoch_idx
        epoch_start = time.perf_counter()

        model.train()
        if cfg["model"].get("freeze_encoders", True):
            model.visumodel.eval()
            model.textmodel.eval()

        train_loss_sum = 0.0
        train_l1_sum = 0.0
        train_giou_sum = 0.0
        steps_in_epoch = 0

        for step, batch in enumerate(train_loader, 1):
            nested_imgs = batch["nested_images"].to(device)
            queries = batch["queries"]
            boxes_list = batch["boxes_list"]

            outputs = model(nested_imgs, queries)
            pred_boxes = outputs["pred_boxes"]

            loss, l1_loss, giou_loss = dynamic_best_match_loss(
                pred_boxes, boxes_list, l1_coef=l1_coef, giou_coef=giou_coef
            )

            if not torch.isfinite(loss):
                stop_reason = f"Numerical instability: Non-finite loss {loss.item()} at Epoch {current_epoch} Step {step}."
                print(f"[ERROR] {stop_reason}")
                stopped_early = True
                break

            optimizer.zero_grad()
            loss.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, clip_max_norm)
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_l1_sum += l1_loss.item()
            train_giou_sum += giou_loss.item()
            steps_in_epoch += 1
            global_step += 1

            if step % 25 == 0 or step == len(train_loader):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{current_epoch}/{max_epochs}] Step [{step}/{len(train_loader)}] Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, GIoU: {giou_loss.item():.4f}) LR: {cur_lr:.2e}")

        if stopped_early:
            break

        epoch_duration = time.perf_counter() - epoch_start
        mean_train_loss = train_loss_sum / max(1, steps_in_epoch)

        # Validation evaluation
        print(f"\n[EVAL] Evaluating Epoch {current_epoch} on MS-CXR validation set ...")
        eval_metrics = evaluate_val_dataset(model, val_loader, device=device, imsize=imsize)
        overall = eval_metrics["__OVERALL__"]
        cur_val_iou = overall["mean_iou"]
        cur_r50 = overall["recall_50"]
        cur_r75 = overall["recall_75"]

        delta_vs_best = cur_val_iou - best_val_iou
        print(f"[Epoch {current_epoch} Result] Time: {epoch_duration:.1f}s | Train Loss: {mean_train_loss:.4f} | Val Mean IoU: {cur_val_iou:.4f} (Delta vs best: {delta_vs_best:+.4f}) | Val R@50: {cur_r50:.1f}%")

        # Catastrophic drop check
        if cur_val_iou < (best_val_iou - max_drop_pause):
            stop_reason = (
                f"Severe validation regression: Val IoU ({cur_val_iou:.4f}) dropped by "
                f"{best_val_iou - cur_val_iou:.4f} (> {max_drop_pause}) from best ({best_val_iou:.4f})."
            )
            print(f"[ALERT] {stop_reason}")
            stopped_early = True

        # Check for meaningful improvement
        is_meaningful_improvement = delta_vs_best > min_improvement
        if is_meaningful_improvement:
            best_val_iou = cur_val_iou
            best_epoch = current_epoch
            consecutive_no_improvement = 0
            best_path = out_ckpt_dir / "best_model_val_iou.pth"
            best_checkpoint_path = str(best_path)
            torch.save({
                "epoch": current_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_metrics": eval_metrics,
                "config": cfg,
                "provenance": {
                    "source_run": "run_01_multicxr",
                    "source_checkpoint": str(base_ckpt),
                    "continuation_epoch": current_epoch,
                },
            }, best_path)
            print(f"[CHECKPOINT] ✓ Meaningful improvement (+{delta_vs_best:.4f} > {min_improvement})! Saved new best model -> {best_path}")
        else:
            consecutive_no_improvement += 1
            print(f"[EARLY STOPPING] Epoch {current_epoch} did not achieve > {min_improvement} improvement over best ({best_val_iou:.4f}). Consecutive epochs without improvement: {consecutive_no_improvement} / {patience}")

        # Always save last model
        last_path = out_ckpt_dir / "last_model.pth"
        torch.save({
            "epoch": current_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": eval_metrics,
            "config": cfg,
            "provenance": {
                "source_run": "run_01_multicxr",
                "source_checkpoint": str(base_ckpt),
                "continuation_epoch": current_epoch,
            },
        }, last_path)

        epoch_record = {
            "epoch": current_epoch,
            "train_loss": round(mean_train_loss, 4),
            "epoch_duration_s": round(epoch_duration, 1),
            "learning_rate": scheduler.get_last_lr()[0],
            "eval_metrics": eval_metrics,
            "delta_vs_best": round(delta_vs_best, 4),
            "is_new_best": is_meaningful_improvement,
            "consecutive_no_improvement": consecutive_no_improvement,
        }
        history["epochs"].append(epoch_record)

        with open(cfg["output"]["log_file"], "w") as f:
            json.dump(history, f, indent=2)

        # Check early stopping patience
        if consecutive_no_improvement >= patience:
            stop_reason = f"Early stopping triggered: {consecutive_no_improvement} consecutive epochs without meaningful improvement (> {min_improvement})."
            print(f"\n[STOP] {stop_reason}")
            stopped_early = True
            break

    if not stop_reason:
        stop_reason = f"Completed all {additional_epochs} continuation epochs (reached max Epoch {max_epochs})."

    history["stopping_summary"] = {
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "selected_best_epoch": best_epoch,
        "selected_best_val_iou": round(best_val_iou, 4),
        "selected_checkpoint_path": best_checkpoint_path,
        "baseline_val_iou": baseline_best_val_iou,
        "absolute_gain": round(best_val_iou - baseline_best_val_iou, 4),
    }

    with open(cfg["output"]["log_file"], "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n=======================================================")
    print(f"Task 3 Continuation Finished: {stop_reason}")
    print(f"Selected Best Epoch: Epoch {best_epoch} (Val IoU = {best_val_iou:.4f})")
    print(f"Selected Best Checkpoint: {best_checkpoint_path}")
    print(f"Log saved: {cfg['output']['log_file']}")
    print(f"=======================================================")


def main():
    parser = argparse.ArgumentParser(description="Multi-Pathology TransVG Continuation Training (Task 3)")
    parser.add_argument("--config", default="configs/multipathology_continuation_to_6ep.yaml", help="Path to config yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick preflight smoke test")
    parser.add_argument("--device", default=None, help="Device override: auto, mps, cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_continuation(cfg, args)


if __name__ == "__main__":
    main()
