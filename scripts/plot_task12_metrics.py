#!/usr/bin/env python3
"""
scripts/plot_task12_metrics.py
==============================
Generates comparative visualization of the Auxiliary Pretrained Classifier
(TorchXRayVision DenseNet-121) threshold sweep, three-way pipeline performance
comparison, safety error reductions, and latency profile for TASK 12.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_IMP_DIR = PROJECT_ROOT / "artifacts" / "local_improvements"


def generate_plots():
    clf_csv = LOCAL_IMP_DIR / "task12_classifier_evaluation.csv"
    pipe_csv = LOCAL_IMP_DIR / "task12_pipeline_comparison.csv"
    lat_csv = LOCAL_IMP_DIR / "task12_latency_profile.csv"

    if not clf_csv.exists() or not pipe_csv.exists():
        print("Required CSV files not found in artifacts/local_improvements/")
        return

    df_clf = pd.read_csv(clf_csv)
    df_pipe = pd.read_csv(pipe_csv)
    df_lat = pd.read_csv(lat_csv)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "TASK 12: Evidence-Aware Review & Auxiliary Classifier Pipeline Evaluation\n"
        "(Dataset: N=50 val_dev, Hardware: Apple Silicon M3 MPS, Library: TorchXRayVision DenseNet-121)",
        fontsize=14,
        fontweight="bold"
    )

    # -------------------------------------------------------------
    # Panel 1: Classifier Threshold Calibration Curve
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(df_clf["threshold"], df_clf["micro_precision"], "o-", label="Precision", color="#2980b9", lw=2)
    ax1.plot(df_clf["threshold"], df_clf["micro_recall"], "s-", label="Recall", color="#27ae60", lw=2)
    ax1.plot(df_clf["threshold"], df_clf["micro_f1"], "^-", label="Micro-F1", color="#e67e22", lw=2.5)
    ax1.plot(df_clf["threshold"], df_clf["macro_f1"], "d--", label="Macro-F1", color="#8e44ad", lw=1.8)

    # Highlight chosen threshold
    chosen_tau = 0.62
    ax1.axvline(x=chosen_tau, color="#c0392b", linestyle=":", lw=2, label=f"Chosen Review Threshold (τ={chosen_tau})")
    ax1.scatter([chosen_tau], [0.4919], color="#c0392b", s=100, zorder=5)
    ax1.annotate(
        f"Peak F1: 0.492\nRec: 1.000",
        xy=(chosen_tau, 0.4919),
        xytext=(chosen_tau + 0.02, 0.53),
        arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#c0392b"
    )

    ax1.set_xlabel("Classifier Review Threshold (τ)", fontweight="bold")
    ax1.set_ylabel("Metric Value", fontweight="bold")
    ax1.set_title("A. TorchXRayVision DenseNet-121 Operating Curve", fontweight="bold")
    ax1.set_xlim(0.48, 0.77)
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="lower left", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 2: Three-Way Pipeline Comparison (F1, Recall, Device F1)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    pipelines = ["Baseline Prompt v3", "Existing Hybrid (Task 9)", "Evidence-Aware Hybrid (Task 12)"]
    metrics = ["Micro-F1", "Macro-F1", "Finding Recall", "Device F1"]
    
    # Extract values
    v3_vals = [df_pipe.loc[0, "micro_f1"], df_pipe.loc[0, "macro_f1"], df_pipe.loc[0, "recall"], df_pipe.loc[0, "device_f1"]]
    task9_vals = [df_pipe.loc[1, "micro_f1"], df_pipe.loc[1, "macro_f1"], df_pipe.loc[1, "recall"], df_pipe.loc[1, "device_f1"]]
    task12_vals = [df_pipe.loc[2, "micro_f1"], df_pipe.loc[2, "macro_f1"], df_pipe.loc[2, "recall"], df_pipe.loc[2, "device_f1"]]

    x = np.arange(len(metrics))
    width = 0.25

    rects1 = ax2.bar(x - width, v3_vals, width, label="Baseline Prompt v3", color="#95a5a6")
    rects2 = ax2.bar(x, task9_vals, width, label="Existing Hybrid Pipeline", color="#3498db")
    rects3 = ax2.bar(x + width, task12_vals, width, label="Evidence-Aware Pipeline (Task 12)", color="#2ecc71")

    ax2.set_ylabel("Metric Score", fontweight="bold")
    ax2.set_title("B. Three-Way Pipeline Diagnostic Performance", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontweight="bold")
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc="upper right", fontsize=8.5)

    def autolabel(rects, ax, color="black"):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=color)

    autolabel(rects1, ax2)
    autolabel(rects2, ax2)
    autolabel(rects3, ax2, color="#27ae60")

    # -------------------------------------------------------------
    # Panel 3: Safety & Clinical Error Reduction
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    error_metrics = ["False-Normal Rate\n(Case-Level %)", "Finding Omission Rate\n(Finding-Level %)", "Unsupported Assertions\n(Rate %)"]
    
    fn_rates = [df_pipe.loc[0, "false_normal_rate"] * 100, df_pipe.loc[1, "false_normal_rate"] * 100, df_pipe.loc[2, "false_normal_rate"] * 100]
    om_rates = [df_pipe.loc[0, "omission_rate"] * 100, df_pipe.loc[1, "omission_rate"] * 100, df_pipe.loc[2, "omission_rate"] * 100]
    un_rates = [df_pipe.loc[0, "unsupported_assertion_rate"] * 100, df_pipe.loc[1, "unsupported_assertion_rate"] * 100, df_pipe.loc[2, "unsupported_assertion_rate"] * 100]

    x3 = np.arange(len(error_metrics))
    r1 = ax3.bar(x3 - width, [fn_rates[0], om_rates[0], un_rates[0]], width, label="Baseline Prompt v3", color="#e74c3c", alpha=0.7)
    r2 = ax3.bar(x3, [fn_rates[1], om_rates[1], un_rates[1]], width, label="Existing Hybrid", color="#f39c12", alpha=0.8)
    r3 = ax3.bar(x3 + width, [fn_rates[2], om_rates[2], un_rates[2]], width, label="Evidence-Aware (Task 12)", color="#27ae60")

    ax3.set_ylabel("Error Rate (%)", fontweight="bold")
    ax3.set_title("C. Clinical Safety & Error Rates (Lower is Better)", fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(error_metrics, fontweight="bold")
    ax3.set_ylim(0, 105)
    ax3.legend(loc="upper right", fontsize=8.5)

    for rect in r1:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in r2:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in r3:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#27ae60", fontweight="bold")

    # -------------------------------------------------------------
    # Panel 4: Latency Profile & Classifier Overhead
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    comp_names = [
        "1. Whole VLM\n(Ollama)",
        "2. Device\n(Ollama)",
        "3. TransVG\n(MPS)",
        "4. Reg Crop\n(PIL)",
        "5. Reg VLM\n(Ollama)",
        "7. DenseNet\n(MPS)"
    ]
    comp_latencies = [
        14420.0 / 1000.0,
        5726.5 / 1000.0,
        439.5 / 1000.0,
        47.5 / 1000.0,
        9838.0 / 1000.0,
        208.8 / 1000.0
    ]
    colors = ["#34495e", "#2980b9", "#16a085", "#bdc3c7", "#8e44ad", "#e67e22"]
    
    y_pos = np.arange(len(comp_names))
    bars = ax4.barh(y_pos, comp_latencies, color=colors, height=0.6)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(comp_names, fontsize=8.5)
    ax4.invert_yaxis()
    ax4.set_xlabel("Median Execution Latency (seconds)", fontweight="bold")
    ax4.set_title("D. Stage-Wise Latency & Classifier Overhead (Apple Silicon M3)", fontweight="bold")
    ax4.set_xlim(0, 18)

    for bar, lat in zip(bars, comp_latencies):
        w = bar.get_width()
        if lat < 1.0:
            txt = f"{lat * 1000:.0f} ms"
        else:
            txt = f"{lat:.2f} s"
        ax4.text(w + 0.3, bar.get_y() + bar.get_height() / 2, txt, va="center", ha="left", fontsize=8.5, fontweight="bold")

    # Annotate DenseNet overhead
    ax4.annotate(
        "DenseNet-121 overhead: 208.8 ms\n(<0.7% of hybrid pipeline)",
        xy=(0.2088, 5),
        xytext=(3.5, 4.8),
        arrowprops=dict(facecolor="#e67e22", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        fontweight="bold",
        color="#d35400"
    )

    plt.tight_layout()
    out_path = RESULT_DIR / "task12_classifier_and_pipeline_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figure successfully saved to {out_path}")


if __name__ == "__main__":
    generate_plots()
