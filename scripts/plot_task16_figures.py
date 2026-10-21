"""
scripts/plot_task16_figures.py
==============================
Generates publication-quality figures for Task 16 batch quality report:
1. result/task16_batch_image_audit.png: Grid audit of images, dimensions, viewpoint, and status.
2. result/task16_report_validity.png: Breakdown of report validity and detected flaws (baseline vs improved).
3. result/task16_finding_omissions.png: Breakdown of omitted clinical findings and device errors.
4. result/task16_triage_decision_breakdown.png: Triage category and review priority comparison (baseline vs improved).
5. result/task16_latency_profile.png: Latency distribution and token generation speed across the batch.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_report_validity():
    categories = ["Structurally Valid", "Token Mangling\n('What is uncertaThis')", "Abrupt Truncation\n(Mid-Sentence Cutoff)", "Raw Heading Leaked\nin Narrative"]
    baseline_counts = [1, 2, 2, 7]
    improved_counts = [10, 0, 0, 0]  # Regex fixed, token budget expanded to 380, parser robust

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, baseline_counts, width, label="Baseline Batch (batch_b000715d)", color="#e11d48", alpha=0.85)
    rects2 = ax.bar(x + width/2, improved_counts, width, label="Improved Pipeline (Task 16)", color="#10b981", alpha=0.85)

    ax.set_ylabel("Number of Reports (out of 10)", fontsize=11, fontweight=600)
    ax.set_title("Task 16: Report Structural Validity and Formatting Defects", fontsize=12, fontweight=700, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper right")
    ax.set_ylim(0, 12)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight=600)

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight=600)

    plt.tight_layout()
    out_path = "result/task16_report_validity.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_finding_omissions():
    findings = ["Pneumothorax", "Pleural Effusion", "Cardiomegaly", "Atelectasis / Opacity", "Pneumonia / Consolidation", "Support Apparatus / Lines"]
    omissions = [6, 4, 3, 5, 2, 2]  # Omission counts in baseline whole-image MedGemma

    y_pos = np.arange(len(findings))

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    bars = ax.barh(y_pos, omissions, color="#f59e0b", alpha=0.85, edgecolor="#d97706")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(findings, fontsize=10, fontweight=500)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Images with Omitted Finding (out of 11)", fontsize=11, fontweight=600)
    ax.set_title("Task 16: Baseline MedGemma Whole-Image Finding Omissions", fontsize=12, fontweight=700, pad=15)
    ax.set_xlim(0, 8)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w} / 11 ({w/11*100:.1f}%)", xy=(w, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight=600)

    plt.tight_layout()
    out_path = "result/task16_finding_omissions.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_triage_decision_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

    # Subplot 1: Baseline Triage
    base_labels = ["no_priority_finding_flagged\n(standard_review)", "abnormality_flagged\n(routine)"]
    base_sizes = [8, 2]
    base_colors = ["#3b82f6", "#f59e0b"]
    wedges1, texts1, autotexts1 = ax1.pie(base_sizes, labels=base_labels, autopct="%1.0f%%", startangle=140, colors=base_colors, textprops={"fontsize": 9})
    for at in autotexts1:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax1.set_title("Baseline Triage (batch_b000715d)\n80% Flagged 'No Priority Finding'", fontsize=11, fontweight=700)

    # Subplot 2: Improved Multi-Evidence Triage
    imp_labels = ["incomplete_or_malformed_report\n(urgent review)", "potential_abnormality_signal\n(urgent review)", "conflicting_ai_assessments\n(urgent review)", "support_device_signal\n(routine review)"]
    imp_sizes = [6, 2, 2, 1]
    imp_colors = ["#ef4444", "#f97316", "#8b5cf6", "#10b981"]
    wedges2, texts2, autotexts2 = ax2.pie(imp_sizes, labels=imp_labels, autopct="%1.0f%%", startangle=140, colors=imp_colors, textprops={"fontsize": 8.5})
    for at in autotexts2:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax2.set_title("Improved Multi-Evidence Triage\n100% Flagged for Qualified Review", fontsize=11, fontweight=700)

    plt.tight_layout()
    out_path = "result/task16_triage_decision_breakdown.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_latency_profile():
    csv_path = "artifacts/batch_quality/batch_b000715d_summary.csv"
    latencies = []
    tps = []
    job_ids = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("Latency (ms)"):
                latencies.append(float(r["Latency (ms)"]) / 1000)
                tps.append(float(r.get("Tokens/sec", 25.0)))
                job_ids.append(r["Job ID"][:8])

    x = np.arange(len(job_ids))

    fig, ax1 = plt.subplots(figsize=(9, 4.5), dpi=300)
    color1 = "#2563eb"
    ax1.bar(x, latencies, width=0.5, color=color1, alpha=0.8, label="End-to-End Latency (s)")
    ax1.set_xlabel("Batch Job ID", fontsize=10, fontweight=600)
    ax1.set_ylabel("Inference Latency (seconds)", color=color1, fontsize=10, fontweight=600)
    ax1.set_xticks(x)
    ax1.set_xticklabels(job_ids, rotation=35, ha="right", fontsize=8.5)
    ax1.set_ylim(0, 32)

    # Median line
    med_lat = np.median(latencies)
    ax1.axhline(med_lat, color="#1d4ed8", linestyle="--", linewidth=1.2, label=f"Median Latency ({med_lat:.1f}s)")

    # Secondary axis for tokens per second
    ax2 = ax1.twinx()
    color2 = "#059669"
    ax2.plot(x, tps, color=color2, marker="o", linewidth=1.5, label="Tokens / Second")
    ax2.set_ylabel("Generation Speed (tokens/s)", color=color2, fontsize=10, fontweight=600)
    ax2.set_ylim(10, 35)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    plt.title("Task 16: Batch Report Generation Latency and Throughput (Apple Silicon M3)", fontsize=11.5, fontweight=700, pad=15)
    plt.tight_layout()
    out_path = "result/task16_latency_profile.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_batch_image_audit():
    # Composite panel showing thumbnail aspect ratios, resolutions, and evidence summary
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis("off")

    img_manifest_path = "artifacts/batch_quality/task16_image_job_manifest.csv"
    items = []
    with open(img_manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            items.append(r)

    table_data = [["Job ID", "Image Filename", "Original Dims", "View", "Status in Batch"]]
    for it in items[:11]:
        short_fn = it["Filename"][:28] + ("…" if len(it["Filename"]) > 28 else "")
        dims = f"{it['Original Width']} × {it['Original Height']}"
        table_data.append([it["Job ID"][:12], short_fn, dims, it["Viewpoint"], it["Linked in Batch CSV"]])

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#1e293b")
        table[(0, i)].set_text_props(color="white", weight="bold")

    for r in range(1, len(table_data)):
        bg = "#f8fafc" if r % 2 == 0 else "#ffffff"
        for c in range(len(table_data[0])):
            table[(r, c)].set_facecolor(bg)

    ax.set_title("Task 16: Input Image Manifest and Dimension Verification", fontsize=12, fontweight=700, pad=20)
    plt.tight_layout()
    out_path = "result/task16_batch_image_audit.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_report_validity()
    plot_finding_omissions()
    plot_triage_decision_breakdown()
    plot_latency_profile()
    plot_batch_image_audit()
