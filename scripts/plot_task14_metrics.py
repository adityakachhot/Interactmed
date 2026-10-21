#!/usr/bin/env python3
"""
scripts/plot_task14_metrics.py
==============================
Publication-quality visualization generator for Task 14:
1. result/task14_held_out_roc_pr_curve.png: Held-out performance trade-offs (tau=0.62 vs tau=0.64) & per-category metrics.
2. result/task14_held_out_pipeline_comparison.png: Cross-architecture evaluation on held-out cohort (Baseline vs Advisory vs Balanced).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "local_improvements" / "task14_final_evaluation_audit"

def plot_held_out_tradeoffs():
    per_cat_csv = AUDIT_DIR / "task14_held_out_per_category_metrics.csv"
    if not per_cat_csv.exists():
        print(f"Per-category CSV not found at {per_cat_csv}")
        return

    df_cat = pd.read_csv(per_cat_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Panel 1: Per-Category Recall comparison across thresholds (tau=0.62 vs tau=0.64)
    cats = [
        "pneumonia_consolidation",
        "pulmonary_edema",
        "pleural_effusion",
        "pneumothorax",
        "atelectasis",
        "cardiomegaly"
    ]
    labels = [
        "Pneumonia /\nConsol.",
        "Pulmonary\nEdema",
        "Pleural\nEffusion",
        "Pneumothorax",
        "Atelectasis",
        "Cardiomegaly"
    ]

    df_062 = df_cat[df_cat["threshold"] == 0.62].set_index("finding_category").reindex(cats)
    df_064 = df_cat[df_cat["threshold"] == 0.64].set_index("finding_category").reindex(cats)

    x = np.arange(len(cats))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df_062["recall"], width, label="Advisory Review (τ=0.62)", color="#27ae60", alpha=0.9)
    rects2 = ax1.bar(x + width/2, df_064["recall"], width, label="Balanced Alert (τ=0.64)", color="#2980b9", alpha=0.9)

    ax1.set_title("Held-Out Cohort: Per-Category Sensitivity (Recall)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Sensitivity / Recall", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontweight="bold", fontsize=9.5)
    ax1.set_ylim(0, 1.18)
    ax1.axhline(y=1.0, color="#7f8c8d", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=10)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#1e8449")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#1f618d")

    # Panel 2: Operating Point Trade-Off: Alert Burden vs False-Normal Rate
    policies = [
        "Advisory Review\n(τ=0.62, Frozen)",
        "Balanced F1 Alert\n(τ=0.64, Tuned)"
    ]
    fn_rates = [0.0, 26.2]
    alert_burdens = [90.6, 44.4]
    precisions = [42.0, 64.2]

    x2 = np.arange(len(policies))
    w2 = 0.25

    b1 = ax2.bar(x2 - w2, fn_rates, w2, label="False-Normal Rate (% Abnormal Cases Missed)", color="#c0392b")
    b2 = ax2.bar(x2, alert_burdens, w2, label="Alert Burden (% Cases with ≥1 False Alert)", color="#e67e22")
    b3 = ax2.bar(x2 + w2, precisions, w2, label="Finding Precision (%)", color="#2980b9")

    ax2.set_title("Clinical Trade-Off: Safety vs False Alarm Burden (N=117)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Percentage (%)", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(policies, fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.legend(loc="upper right", fontsize=9.5)

    for rect in b1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#922b21")
    for rect in b2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#b9770e")
    for rect in b3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#1f618d")

    plt.tight_layout()
    out_p = RESULT_DIR / "task14_held_out_roc_pr_curve.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Saved: {out_p}")


def plot_pipeline_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    pipelines = [
        "Baseline MedGemma 4B\n(Prompt v3)",
        "Task 9 Hybrid Pipeline\n(Grounding Assisted)",
        "Task 14 Advisory Review\n(TorchXRayVision τ=0.62)",
        "Task 14 Balanced Alert\n(TorchXRayVision τ=0.64)"
    ]

    recall = [29.3, 53.1, 97.0, 69.0]
    precision = [25.8, 43.2, 42.0, 64.2]
    f1 = [27.4, 47.6, 58.6, 66.5]
    fn_rate = [37.8, 24.2, 0.0, 26.2]

    x = np.arange(len(pipelines))
    width = 0.2

    r1 = ax.bar(x - 1.5*width, precision, width, label="Precision (%)", color="#2980b9")
    r2 = ax.bar(x - 0.5*width, recall, width, label="Recall / Sensitivity (%)", color="#27ae60")
    r3 = ax.bar(x + 0.5*width, f1, width, label="F1 Score (%)", color="#e67e22")
    r4 = ax.bar(x + 1.5*width, fn_rate, width, label="Case False-Normal Rate (%)", color="#c0392b")

    ax.set_title("InteractMed-VLM Evolution: Finding & Safety Metrics across Architectures", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score / Rate (%)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines, fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=9.5)

    for bar_group, col in [(r1, "#1f618d"), (r2, "#1e8449"), (r3, "#b9770e"), (r4, "#922b21")]:
        for rect in bar_group:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=8, fontweight="bold", color=col)

    plt.tight_layout()
    out_p = RESULT_DIR / "task14_held_out_pipeline_comparison.png"
    plt.savefig(out_p, dpi=300)
    plt.close()
    print(f"Saved: {out_p}")


if __name__ == "__main__":
    plot_held_out_tradeoffs()
    plot_pipeline_comparison()
