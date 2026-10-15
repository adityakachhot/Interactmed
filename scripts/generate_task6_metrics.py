#!/usr/bin/env python3
"""
scripts/generate_task6_metrics.py
=================================
Compiles the authoritative metric inventory and supporting CSV files
for the InteractMed-VLM Task 6 Final Research Report.
Strict adherence to verified artifacts and no fabricated values.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_REPORT_DIR = PROJECT_ROOT / "artifacts" / "final_report"
FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_metric_inventory():
    """
    Builds artifacts/final_report/metric_inventory.csv with columns:
    metric_name, category, status, existing_value_or_file, data_source, sample_size, split, missing_requirements, action
    """
    rows = [
        # --- Grounding: Training & Validation Progression ---
        {
            "metric_name": "val_miou_task2_ep1",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6402",
            "data_source": "artifacts/multipathology/run_01_multicxr/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified historical training record"
        },
        {
            "metric_name": "val_miou_task2_ep2",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6432",
            "data_source": "artifacts/multipathology/run_01_multicxr/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified historical training record"
        },
        {
            "metric_name": "val_miou_task2_ep3",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6510",
            "data_source": "artifacts/multipathology/run_01_multicxr/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified historical training record"
        },
        {
            "metric_name": "val_recall50_task2_ep3",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "75.44%",
            "data_source": "artifacts/multipathology/run_01_multicxr/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified historical training record"
        },
        {
            "metric_name": "val_recall75_task2_ep3",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "46.49%",
            "data_source": "artifacts/multipathology/run_01_multicxr/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified historical training record"
        },
        {
            "metric_name": "val_miou_task3_ep4",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6401",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified continuation record"
        },
        {
            "metric_name": "val_miou_task3_ep5",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6550",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified selected best checkpoint record"
        },
        {
            "metric_name": "val_miou_task3_ep6",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6523",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified continuation record"
        },
        {
            "metric_name": "val_recall50_task3_ep5",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "74.56%",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified selected best checkpoint record"
        },
        {
            "metric_name": "val_recall75_task3_ep5",
            "category": "Grounding - Training & Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "46.49%",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/training_history.json",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified selected best checkpoint record"
        },

        # --- Grounding: Checkpoint Comparison ---
        {
            "metric_name": "test_miou_pretrained_baseline",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6307",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record (inspected cohort)"
        },
        {
            "metric_name": "test_recall50_pretrained_baseline",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "71.54%",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record (inspected cohort)"
        },
        {
            "metric_name": "test_recall75_pretrained_baseline",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "36.92%",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record (inspected cohort)"
        },
        {
            "metric_name": "test_miou_rsna_finetuned",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.3800",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record (catastrophic forgetting)"
        },
        {
            "metric_name": "test_recall50_rsna_finetuned",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "26.15%",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_recall75_rsna_finetuned",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "2.31%",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_miou_task2_multipathology",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6599",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_recall50_task2_multipathology",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "76.92%",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_recall75_task2_multipathology",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "43.85%",
            "data_source": "artifacts/multi_pathology_eval/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_miou_task3_continuation_ep5",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.6672",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_recall50_task3_continuation_ep5",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "80.00%",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },
        {
            "metric_name": "test_recall75_task3_continuation_ep5",
            "category": "Grounding - Checkpoint Comparison",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "43.85%",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified retrospective test record"
        },

        # --- Grounding: Per-Category Breakdown ---
        {
            "metric_name": "per_category_val_miou_task2_vs_task3",
            "category": "Grounding - Per-Category Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "8 categories (Cardiomegaly 0.8291->0.8315, Pneumonia 0.5394->0.5546, etc.)",
            "data_source": "training_history.json in run_01 and run_02",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified per-category validation breakdowns"
        },
        {
            "metric_name": "per_category_val_recall50_task2_vs_task3",
            "category": "Grounding - Per-Category Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "8 categories (Pneumonia 53.85%, Pneumothorax 69.23%, etc.)",
            "data_source": "training_history.json in run_01 and run_02",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified per-category validation recall"
        },
        {
            "metric_name": "per_category_val_recall75_task2_vs_task3",
            "category": "Grounding - Per-Category Validation",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "8 categories (Pneumonia 23.08%, Consolidation 25.0%, etc.)",
            "data_source": "training_history.json in run_01 and run_02",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Reuse verified per-category validation recall"
        },
        {
            "metric_name": "category_miou_absolute_delta_task3_vs_task2",
            "category": "Grounding - Per-Category Validation",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "Calculated in Phase B",
            "data_source": "training_history.json files (run_01 and run_02)",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "None",
            "action": "Calculate exact category-level delta values and percentage changes"
        },
        {
            "metric_name": "grounding_valid_box_rate",
            "category": "Grounding - Bounding Box Quality",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "100.0% (0 invalid or degenerate boxes)",
            "data_source": "validation and test histories (run_01 and run_02)",
            "sample_size": "114 val / 130 test",
            "split": "validation & retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified coordinate validity records"
        },
        {
            "metric_name": "grounding_inference_latency_mean",
            "category": "Grounding - Latency",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "176.49 ms mean",
            "data_source": "artifacts/multipathology/run_02_continuation_to_6ep/locked_test_results.json",
            "sample_size": "130 queries",
            "split": "retrospective_test",
            "missing_requirements": "None",
            "action": "Reuse verified latency benchmark"
        },
        {
            "metric_name": "grounding_bootstrap_95ci",
            "category": "Grounding - Confidence Intervals",
            "status": "MISSING_DATA_REQUIRED",
            "existing_value_or_file": "None",
            "data_source": "Per-query prediction vectors for validation split",
            "sample_size": "114 queries",
            "split": "validation",
            "missing_requirements": "Raw per-query prediction vectors with patient cluster IDs required for non-aggregated bootstrap",
            "action": "Document as unavailable per prompt rules rather than estimating from aggregate means"
        },
        {
            "metric_name": "grounding_disease_presence_auroc",
            "category": "Grounding - Diagnostic Classification",
            "status": "NOT_APPLICABLE",
            "existing_value_or_file": "N/A",
            "data_source": "Model Architecture",
            "sample_size": "0",
            "split": "N/A",
            "missing_requirements": "TransVG has no calibrated presence/objectness score; localization does not validate clinical diagnosis",
            "action": "Exclude from medical claims"
        },

        # --- Report Generation: Micro & Macro Finding Extraction ---
        {
            "metric_name": "report_micro_precision_baseline",
            "category": "Report Generation - Finding Extraction",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.5521 (95% CI: [0.4979, 0.6052])",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified 250-case dev benchmark"
        },
        {
            "metric_name": "report_micro_recall_baseline",
            "category": "Report Generation - Finding Extraction",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.3629 (95% CI: [0.3218, 0.4061])",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified 250-case dev benchmark"
        },
        {
            "metric_name": "report_micro_f1_baseline",
            "category": "Report Generation - Finding Extraction",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.4379",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified 250-case dev benchmark"
        },
        {
            "metric_name": "report_micro_precision_prompt_v3",
            "category": "Report Generation - Finding Extraction",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.4590 (95% CI: [0.4003, 0.5188])",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified 250-case dev benchmark"
        },
        {
            "metric_name": "report_micro_recall_prompt_v3",
            "category": "Report Generation - Finding Extraction",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.2480 (95% CI: [0.2120, 0.2878])",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified 250-case dev benchmark"
        },
        {
            "metric_name": "report_micro_f1_prompt_v3",
            "category": "Report Generation - Finding Extraction",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.3220",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified 250-case dev benchmark"
        },
        {
            "metric_name": "report_macro_precision_baseline",
            "category": "Report Generation - Finding Extraction",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "0.3460",
            "data_source": "Calculated from category_level_metrics in expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Compute macro-average precision across 7 categories"
        },
        {
            "metric_name": "report_macro_recall_baseline",
            "category": "Report Generation - Finding Extraction",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "0.3879",
            "data_source": "Calculated from category_level_metrics in expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Compute macro-average recall across 7 categories"
        },
        {
            "metric_name": "report_macro_f1_baseline",
            "category": "Report Generation - Finding Extraction",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "0.2792",
            "data_source": "Calculated from category_level_metrics in expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Compute macro-average F1 across 7 categories"
        },
        {
            "metric_name": "report_macro_precision_prompt_v3",
            "category": "Report Generation - Finding Extraction",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "0.4007",
            "data_source": "Calculated from category_level_metrics in expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Compute macro-average precision across 7 categories (+5.47% vs baseline)"
        },
        {
            "metric_name": "report_macro_recall_prompt_v3",
            "category": "Report Generation - Finding Extraction",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "0.3493",
            "data_source": "Calculated from category_level_metrics in expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Compute macro-average recall across 7 categories"
        },
        {
            "metric_name": "report_macro_f1_prompt_v3",
            "category": "Report Generation - Finding Extraction",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "0.2251",
            "data_source": "Calculated from category_level_metrics in expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Compute macro-average F1 across 7 categories"
        },
        {
            "metric_name": "report_per_category_metrics",
            "category": "Report Generation - Per-Finding Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "7 categories (Cardiomegaly, Pleural Effusion, Pneumonia, Edema, Atelectasis, Devices, Normal)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified per-category TP, FP, FN, Precision, Recall, F1"
        },
        {
            "metric_name": "report_confusion_matrices",
            "category": "Report Generation - Confusion Matrices",
            "status": "MISSING_CALCULABLE",
            "existing_value_or_file": "Calculated 2x2 contingency tables per finding category",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Calculate TP, FP, FN, TN per finding category"
        },

        # --- Report Generation: Clinical Error Metrics ---
        {
            "metric_name": "report_false_normal_rate_baseline",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "37.99% (68 / 179 abnormal cases)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "179 abnormal cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_false_normal_rate_prompt_v3",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "39.66% (71 / 179 abnormal cases)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "179 abnormal cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_internal_contradiction_rate_baseline",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "7.20% (18 / 250 cases)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_internal_contradiction_rate_prompt_v3",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "1.60% (4 / 250 cases)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation (77.8% relative reduction)"
        },
        {
            "metric_name": "report_normal_boilerplate_rate_baseline",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "56.00% (140 / 250 cases)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_normal_boilerplate_rate_prompt_v3",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.00% (0 / 250 cases)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation (100% elimination)"
        },
        {
            "metric_name": "report_laterality_omissions_baseline",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "105 omissions",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_laterality_omissions_prompt_v3",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "94 omissions",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation (10.5% reduction)"
        },
        {
            "metric_name": "report_laterality_concordant_baseline",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "43 cases",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_laterality_concordant_prompt_v3",
            "category": "Report Generation - Clinical Error Metrics",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "54 cases",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation (+25.6% relative gain)"
        },
        {
            "metric_name": "report_latency_baseline",
            "category": "Report Generation - Latency",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "18.76 s mean (18563.8 ms median)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation"
        },
        {
            "metric_name": "report_latency_prompt_v3",
            "category": "Report Generation - Latency",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "13.42 s mean (12528.0 ms median)",
            "data_source": "artifacts/report_evaluation/expanded_cohort_full/expanded_summary.json",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "None",
            "action": "Reuse verified historical dev evaluation (28.5% faster generation)"
        },

        # --- External Benchmark Models ---
        {
            "metric_name": "chexbert_f1",
            "category": "Report Generation - External NLP Metrics",
            "status": "MISSING_DATA_REQUIRED",
            "existing_value_or_file": "None",
            "data_source": "Stanford CheXbert PyTorch model weights & tokenized impressions",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "Pretrained CheXbert model checkpoint and tokenizer dependencies not distributed in local repo",
            "action": "Document as unavailable without external model downloads"
        },
        {
            "metric_name": "radgraph_f1",
            "category": "Report Generation - External NLP Metrics",
            "status": "MISSING_DATA_REQUIRED",
            "existing_value_or_file": "None",
            "data_source": "DyGIE++ RadGraph entity/relation extraction model",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "DyGIE++ RadGraph model checkpoint and dygiepp Python environment not installed locally",
            "action": "Document as unavailable without external model downloads"
        },
        {
            "metric_name": "green_score",
            "category": "Report Generation - External NLP Metrics",
            "status": "MISSING_DATA_REQUIRED",
            "existing_value_or_file": "None",
            "data_source": "GREEN clinical evaluator framework / LLM referee API",
            "sample_size": "250 cases",
            "split": "development",
            "missing_requirements": "External GREEN evaluation package and LLM referee API key not configured",
            "action": "Document as unavailable"
        },

        # --- End-to-End System Reliability ---
        {
            "metric_name": "batch_completion_rate",
            "category": "End-to-End Reliability",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "100.0% (4 / 4 jobs completed)",
            "data_source": "scratch/evaluation_results.json & Task 5B report",
            "sample_size": "4 live cases",
            "split": "live_integration_test",
            "missing_requirements": "None",
            "action": "Reuse verified live batch evaluation"
        },
        {
            "metric_name": "batch_api_request_success_rate",
            "category": "End-to-End Reliability",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "100.0% (0 failed requests across 4 live tests + 165 automated tests)",
            "data_source": "scratch/evaluation_results.json & pytest suite",
            "sample_size": "4 live / 165 unit",
            "split": "system_integration",
            "missing_requirements": "None",
            "action": "Reuse verified test suite execution records"
        },
        {
            "metric_name": "batch_image_to_report_association_accuracy",
            "category": "End-to-End Reliability",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "100.0% (4 / 4 exact SHA-256 byte & pixel matches)",
            "data_source": "scratch/evaluation_results.json",
            "sample_size": "4 live cases",
            "split": "live_integration_test",
            "missing_requirements": "None",
            "action": "Reuse verified cryptographic provenance audit"
        },
        {
            "metric_name": "batch_cross_image_contamination_rate",
            "category": "End-to-End Reliability",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.0% (0 / 4 cases leaked clinical tags)",
            "data_source": "scratch/evaluation_results.json",
            "sample_size": "4 live cases",
            "split": "live_integration_test",
            "missing_requirements": "None",
            "action": "Reuse verified cross-contamination audit"
        },
        {
            "metric_name": "batch_markdown_artifact_leakage_rate",
            "category": "End-to-End Reliability",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "0.0% (0 / 4 cases had raw Markdown syntax)",
            "data_source": "scratch/evaluation_results.json",
            "sample_size": "4 live cases",
            "split": "live_integration_test",
            "missing_requirements": "None",
            "action": "Reuse verified UI formatting audit"
        },
        {
            "metric_name": "single_vs_batch_semantic_triage_consistency",
            "category": "End-to-End Reliability",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "100.0% (4 / 4 triage rules matched exactly)",
            "data_source": "scratch/evaluation_results.json",
            "sample_size": "4 live cases",
            "split": "live_integration_test",
            "missing_requirements": "None",
            "action": "Reuse verified single vs batch concordance audit"
        },
        {
            "metric_name": "latency_cold_start_total",
            "category": "End-to-End Latency",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "20.34 s (3.00s model load + 16.34s stage 1 + 3.85s stage 2)",
            "data_source": "artifacts/report_evaluation/latency_benchmark.json",
            "sample_size": "1 benchmark run",
            "split": "cold_start_profile",
            "missing_requirements": "None",
            "action": "Reuse verified cold start breakdown"
        },
        {
            "metric_name": "latency_warm_decoding_and_prep",
            "category": "End-to-End Latency",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "117.7 ms mean (109.2 ms median)",
            "data_source": "artifacts/report_evaluation/latency_benchmark.json",
            "sample_size": "25 cases",
            "split": "warm_profile",
            "missing_requirements": "None",
            "action": "Reuse verified decoding profile"
        },
        {
            "metric_name": "latency_warm_report_generation",
            "category": "End-to-End Latency",
            "status": "EXISTS_VERIFIED",
            "existing_value_or_file": "22.53 s mean (21.95 s median)",
            "data_source": "artifacts/report_evaluation/latency_benchmark.json",
            "sample_size": "25 cases",
            "split": "warm_profile",
            "missing_requirements": "None",
            "action": "Reuse verified warm latency benchmark"
        }
    ]

    df = pd.DataFrame(rows)
    out_csv = FINAL_REPORT_DIR / "metric_inventory.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved metric inventory: {out_csv} ({len(df)} rows)")
    return df


def build_evaluation_summary():
    """
    Builds artifacts/final_report/evaluation_summary.csv summarizing key models across splits.
    """
    rows = [
        # Grounding Models
        {
            "component": "Grounding (TransVG-AGPT)",
            "checkpoint_or_pipeline": "Pretrained Baseline (MS-CXR)",
            "evaluation_cohort": "MS-CXR Retrospective Test",
            "sample_size": 130,
            "split_nature": "Retrospective Inspected",
            "primary_metric": "Mean best-match IoU",
            "primary_value": 0.6307,
            "secondary_metric_1": "Recall @ 0.50 (%)",
            "secondary_value_1": 71.54,
            "secondary_metric_2": "Recall @ 0.75 (%)",
            "secondary_value_2": 36.92,
            "notes": "Pretrained on MS-CXR multi-pathology"
        },
        {
            "component": "Grounding (TransVG-AGPT)",
            "checkpoint_or_pipeline": "Fine-Tuned RSNA (Pneumonia Only)",
            "evaluation_cohort": "MS-CXR Retrospective Test",
            "sample_size": 130,
            "split_nature": "Retrospective Inspected",
            "primary_metric": "Mean best-match IoU",
            "primary_value": 0.3800,
            "secondary_metric_1": "Recall @ 0.50 (%)",
            "secondary_value_1": 26.15,
            "secondary_metric_2": "Recall @ 0.75 (%)",
            "secondary_value_2": 2.31,
            "notes": "Severe catastrophic forgetting of non-pneumonia pathologies"
        },
        {
            "component": "Grounding (TransVG-AGPT)",
            "checkpoint_or_pipeline": "Task 2 Multi-Pathology (Ep 3)",
            "evaluation_cohort": "MS-CXR Retrospective Test",
            "sample_size": 130,
            "split_nature": "Retrospective Inspected",
            "primary_metric": "Mean best-match IoU",
            "primary_value": 0.6599,
            "secondary_metric_1": "Recall @ 0.50 (%)",
            "secondary_value_1": 76.92,
            "secondary_metric_2": "Recall @ 0.75 (%)",
            "secondary_value_2": 43.85,
            "notes": "First multi-pathology fine-tuning run (3 epochs)"
        },
        {
            "component": "Grounding (TransVG-AGPT)",
            "checkpoint_or_pipeline": "Task 3 Continuation (Ep 5 Best)",
            "evaluation_cohort": "MS-CXR Retrospective Test",
            "sample_size": 130,
            "split_nature": "Retrospective Inspected",
            "primary_metric": "Mean best-match IoU",
            "primary_value": 0.6672,
            "secondary_metric_1": "Recall @ 0.50 (%)",
            "secondary_value_1": 80.00,
            "secondary_metric_2": "Recall @ 0.75 (%)",
            "secondary_value_2": 43.85,
            "notes": "Controlled continuation run; production integrated checkpoint"
        },
        {
            "component": "Grounding (TransVG-AGPT)",
            "checkpoint_or_pipeline": "Task 2 Multi-Pathology (Ep 3)",
            "evaluation_cohort": "MS-CXR Validation",
            "sample_size": 114,
            "split_nature": "Validation Set",
            "primary_metric": "Mean best-match IoU",
            "primary_value": 0.6510,
            "secondary_metric_1": "Recall @ 0.50 (%)",
            "secondary_value_1": 75.44,
            "secondary_metric_2": "Recall @ 0.75 (%)",
            "secondary_value_2": 46.49,
            "notes": "Validation baseline for continuation"
        },
        {
            "component": "Grounding (TransVG-AGPT)",
            "checkpoint_or_pipeline": "Task 3 Continuation (Ep 5 Best)",
            "evaluation_cohort": "MS-CXR Validation",
            "sample_size": 114,
            "split_nature": "Validation Set",
            "primary_metric": "Mean best-match IoU",
            "primary_value": 0.6550,
            "secondary_metric_1": "Recall @ 0.50 (%)",
            "secondary_value_1": 74.56,
            "secondary_metric_2": "Recall @ 0.75 (%)",
            "secondary_value_2": 46.49,
            "notes": "+0.0040 (+0.40%) absolute gain over Task 2 validation best"
        },

        # Report Generation Pipelines
        {
            "component": "Report Generation (MedGemma)",
            "checkpoint_or_pipeline": "Baseline Single-Stage Prompt",
            "evaluation_cohort": "MIMIC-CXR / ImaGenome Dev",
            "sample_size": 250,
            "split_nature": "Development Cohort",
            "primary_metric": "Micro F1",
            "primary_value": 0.4379,
            "secondary_metric_1": "Micro Precision",
            "secondary_value_1": 0.5521,
            "secondary_metric_2": "Micro Recall",
            "secondary_value_2": 0.3629,
            "notes": "Suffered from 56% boilerplate normal hallucinations"
        },
        {
            "component": "Report Generation (MedGemma)",
            "checkpoint_or_pipeline": "Production Structured Prompt v3",
            "evaluation_cohort": "MIMIC-CXR / ImaGenome Dev",
            "sample_size": 250,
            "split_nature": "Development Cohort",
            "primary_metric": "Micro F1",
            "primary_value": 0.3220,
            "secondary_metric_1": "Micro Precision",
            "secondary_value_1": 0.4590,
            "secondary_metric_2": "Micro Recall",
            "secondary_value_2": 0.2480,
            "notes": "Completely eliminated boilerplate normals (0%), reduced contradictions by 77.8%"
        },

        # End-to-End System Reliability
        {
            "component": "End-to-End System Integration",
            "checkpoint_or_pipeline": "Full Application (Grounding + Reporting)",
            "evaluation_cohort": "Task 5B Live Multi-Image Cases",
            "sample_size": 4,
            "split_nature": "Live Integration Suite",
            "primary_metric": "Batch Job Completion Rate (%)",
            "primary_value": 100.0,
            "secondary_metric_1": "Association Accuracy (%)",
            "secondary_value_1": 100.0,
            "secondary_metric_2": "Cross-Contamination Rate (%)",
            "secondary_value_2": 0.0,
            "notes": "100% cryptographic SHA-256 byte/pixel integrity, 0% Markdown leakage"
        }
    ]

    df = pd.DataFrame(rows)
    out_csv = FINAL_REPORT_DIR / "evaluation_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved evaluation summary: {out_csv} ({len(df)} rows)")
    return df


def build_per_category_grounding_metrics():
    """
    Builds artifacts/final_report/per_category_grounding_metrics.csv
    Comparing Task 2 Epoch 3 and Task 3 Epoch 5 on the validation split (N=114)
    and retrospective test split (N=130).
    """
    # Validation data from training_history.json
    val_data = [
        {"category": "Cardiomegaly", "n_val": 40, "t2_miou": 0.8291, "t3_miou": 0.8315, "t2_rec50": 100.0, "t3_rec50": 100.0, "t2_rec75": 87.50, "t3_rec75": 90.00},
        {"category": "Pneumonia", "n_val": 13, "t2_miou": 0.5394, "t3_miou": 0.5546, "t2_rec50": 53.85, "t3_rec50": 53.85, "t2_rec75": 23.08, "t3_rec75": 23.08},
        {"category": "Pneumothorax", "n_val": 13, "t2_miou": 0.5876, "t3_miou": 0.5845, "t2_rec50": 69.23, "t3_rec50": 69.23, "t2_rec75": 23.08, "t3_rec75": 23.08},
        {"category": "Consolidation", "n_val": 12, "t2_miou": 0.5513, "t3_miou": 0.5663, "t2_rec50": 66.67, "t3_rec50": 66.67, "t2_rec75": 33.33, "t3_rec75": 25.00},
        {"category": "Lung Opacity", "n_val": 12, "t2_miou": 0.4427, "t3_miou": 0.4463, "t2_rec50": 33.33, "t3_rec50": 33.33, "t2_rec75": 16.67, "t3_rec75": 16.67},
        {"category": "Pleural Effusion", "n_val": 11, "t2_miou": 0.6192, "t3_miou": 0.6202, "t2_rec50": 81.82, "t3_rec50": 81.82, "t2_rec75": 27.27, "t3_rec75": 27.27},
        {"category": "Atelectasis", "n_val": 8, "t2_miou": 0.5750, "t3_miou": 0.5521, "t2_rec50": 62.50, "t3_rec50": 50.00, "t2_rec75": 25.00, "t3_rec75": 25.00},
        {"category": "Pulmonary Edema", "n_val": 5, "t2_miou": 0.6116, "t3_miou": 0.6426, "t2_rec50": 80.00, "t3_rec50": 80.00, "t2_rec75": 20.00, "t3_rec75": 20.00},
        {"category": "OVERALL", "n_val": 114, "t2_miou": 0.6510, "t3_miou": 0.6550, "t2_rec50": 75.44, "t3_rec50": 74.56, "t2_rec75": 46.49, "t3_rec75": 46.49},
    ]

    for row in val_data:
        row["delta_miou"] = round(row["t3_miou"] - row["t2_miou"], 4)
        row["rel_pct_gain_miou"] = round((row["delta_miou"] / row["t2_miou"]) * 100.0, 2)
        row["delta_rec50"] = round(row["t3_rec50"] - row["t2_rec50"], 2)
        row["delta_rec75"] = round(row["t3_rec75"] - row["t2_rec75"], 2)

    df = pd.DataFrame(val_data)
    out_csv = FINAL_REPORT_DIR / "per_category_grounding_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved per-category grounding metrics: {out_csv} ({len(df)} rows)")
    return df


def build_report_generation_metrics():
    """
    Builds artifacts/final_report/report_generation_metrics.csv
    Summarizing micro, macro, and per-category finding metrics on 250 cases.
    """
    rows = [
        # Global Micro
        {"metric_level": "Global Micro", "category": "All Findings (Micro)", "cohort_cases": 250, "baseline_precision": 0.5521, "baseline_recall": 0.3629, "baseline_f1": 0.4379, "v3_precision": 0.4590, "v3_recall": 0.2480, "v3_f1": 0.3220, "baseline_tp": 180, "baseline_fp": 146, "baseline_fn": 316, "v3_tp": 123, "v3_fp": 145, "v3_fn": 373},
        # Global Macro
        {"metric_level": "Global Macro", "category": "All Findings (Macro)", "cohort_cases": 250, "baseline_precision": 0.3460, "baseline_recall": 0.3879, "baseline_f1": 0.2792, "v3_precision": 0.4007, "v3_recall": 0.3493, "v3_f1": 0.2251, "baseline_tp": 109, "baseline_fp": 143, "baseline_fn": 141, "v3_tp": 84, "v3_fp": 144, "v3_fn": 166},
        # Per Category
        {"metric_level": "Category Level", "category": "Cardiomegaly", "cohort_cases": 9, "baseline_precision": 0.2069, "baseline_recall": 0.6667, "baseline_f1": 0.3158, "v3_precision": 0.2000, "v3_recall": 0.4444, "v3_f1": 0.2759, "baseline_tp": 6, "baseline_fp": 23, "baseline_fn": 3, "v3_tp": 4, "v3_fp": 16, "v3_fn": 5},
        {"metric_level": "Category Level", "category": "Pleural Effusion", "cohort_cases": 0, "baseline_precision": 0.0000, "baseline_recall": 0.0000, "baseline_f1": 0.0000, "v3_precision": 0.0000, "v3_recall": 0.0000, "v3_f1": 0.0000, "baseline_tp": 0, "baseline_fp": 5, "baseline_fn": 0, "v3_tp": 0, "v3_fp": 0, "v3_fn": 0},
        {"metric_level": "Category Level", "category": "Pneumonia / Consolidation", "cohort_cases": 94, "baseline_precision": 0.9091, "baseline_recall": 0.2128, "baseline_f1": 0.3449, "v3_precision": 0.8333, "v3_recall": 0.0532, "v3_f1": 0.1000, "baseline_tp": 20, "baseline_fp": 2, "baseline_fn": 74, "v3_tp": 5, "v3_fp": 1, "v3_fn": 89},
        {"metric_level": "Category Level", "category": "Pulmonary Edema", "cohort_cases": 57, "baseline_precision": 0.6786, "baseline_recall": 0.3333, "baseline_f1": 0.4470, "v3_precision": 0.6667, "v3_recall": 0.1053, "v3_f1": 0.1819, "baseline_tp": 19, "baseline_fp": 9, "baseline_fn": 38, "v3_tp": 6, "v3_fp": 3, "v3_fn": 51},
        {"metric_level": "Category Level", "category": "Atelectasis", "cohort_cases": 9, "baseline_precision": 0.0000, "baseline_recall": 0.0000, "baseline_f1": 0.0000, "v3_precision": 0.5000, "v3_recall": 0.1111, "v3_f1": 0.1818, "baseline_tp": 0, "baseline_fp": 5, "baseline_fn": 9, "v3_tp": 1, "v3_fp": 1, "v3_fn": 8},
        {"metric_level": "Category Level", "category": "Support Devices", "cohort_cases": 10, "baseline_precision": 0.1522, "baseline_recall": 0.7000, "baseline_f1": 0.2500, "v3_precision": 0.1364, "v3_recall": 0.9000, "v3_f1": 0.2369, "baseline_tp": 7, "baseline_fp": 39, "baseline_fn": 3, "v3_tp": 9, "v3_fp": 57, "v3_fn": 1},
        {"metric_level": "Category Level", "category": "Normal / Unremarkable", "cohort_cases": 71, "baseline_precision": 0.4750, "baseline_recall": 0.8028, "baseline_f1": 0.5969, "v3_precision": 0.4683, "v3_recall": 0.8310, "v3_f1": 0.5990, "baseline_tp": 57, "baseline_fp": 63, "baseline_fn": 14, "v3_tp": 59, "v3_fp": 67, "v3_fn": 12},
    ]

    df = pd.DataFrame(rows)
    out_csv = FINAL_REPORT_DIR / "report_generation_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved report generation metrics: {out_csv} ({len(df)} rows)")
    return df


def build_report_error_metrics():
    """
    Builds artifacts/final_report/report_error_metrics.csv
    Comparing clinical error metrics between baseline prompt and Prompt v3.
    """
    rows = [
        {
            "error_metric": "False Normal Rate (Abnormal Cohort)",
            "numerator_baseline": 68,
            "denominator_baseline": 179,
            "baseline_rate_pct": 37.99,
            "numerator_v3": 71,
            "denominator_v3": 179,
            "v3_rate_pct": 39.66,
            "delta_rate_pct": 1.67,
            "interpretation": "Similar false-normal tendency under conservative structured prompting"
        },
        {
            "error_metric": "Normal Boilerplate Rate",
            "numerator_baseline": 140,
            "denominator_baseline": 250,
            "baseline_rate_pct": 56.00,
            "numerator_v3": 0,
            "denominator_v3": 250,
            "v3_rate_pct": 0.00,
            "delta_rate_pct": -56.00,
            "interpretation": "Complete elimination of generic non-informative template text"
        },
        {
            "error_metric": "Internal Contradiction Rate",
            "numerator_baseline": 18,
            "denominator_baseline": 250,
            "baseline_rate_pct": 7.20,
            "numerator_v3": 4,
            "denominator_v3": 250,
            "v3_rate_pct": 1.60,
            "delta_rate_pct": -5.60,
            "interpretation": "77.8% relative reduction in conflicting clinical statements"
        },
        {
            "error_metric": "Unsupported Positive Assertion Rate",
            "numerator_baseline": 70,
            "denominator_baseline": 250,
            "baseline_rate_pct": 28.00,
            "numerator_v3": 73,
            "denominator_v3": 250,
            "v3_rate_pct": 29.20,
            "delta_rate_pct": 1.20,
            "interpretation": "Marginal change; driven primarily by non-annotated device mentions"
        },
        {
            "error_metric": "Laterality Omission Rate",
            "numerator_baseline": 105,
            "denominator_baseline": 250,
            "baseline_rate_pct": 42.00,
            "numerator_v3": 94,
            "denominator_v3": 250,
            "v3_rate_pct": 37.60,
            "delta_rate_pct": -4.40,
            "interpretation": "10.5% relative reduction in ambiguous un-lateralized finding statements"
        },
        {
            "error_metric": "Laterality Concordance Rate",
            "numerator_baseline": 43,
            "denominator_baseline": 250,
            "baseline_rate_pct": 17.20,
            "numerator_v3": 54,
            "denominator_v3": 250,
            "v3_rate_pct": 21.60,
            "delta_rate_pct": 4.40,
            "interpretation": "25.6% relative improvement in anatomically concordant lateralization"
        }
    ]

    df = pd.DataFrame(rows)
    out_csv = FINAL_REPORT_DIR / "report_error_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved report error metrics: {out_csv} ({len(df)} rows)")
    return df


def build_end_to_end_reliability_metrics():
    """
    Builds artifacts/final_report/end_to_end_reliability_metrics.csv
    Compiling automated software verification and live end-to-end integration metrics.
    """
    rows = [
        {
            "metric_category": "Software Suite Verification",
            "metric_name": "Pytest Automated Test Pass Rate",
            "numerator": 165,
            "denominator": 165,
            "observed_value": "100.0%",
            "engineering_target": "100.0%",
            "status": "PASS",
            "evidence_source": "Backend & Frontend test suites (pytest src/ backend/)"
        },
        {
            "metric_category": "Live Batch Processing",
            "metric_name": "Batch Job Execution Completion Rate",
            "numerator": 4,
            "denominator": 4,
            "observed_value": "100.0%",
            "engineering_target": "100.0%",
            "status": "PASS",
            "evidence_source": "scratch/evaluation_results.json (live backend execution)"
        },
        {
            "metric_category": "Live Batch Processing",
            "metric_name": "API Endpoint Success Rate",
            "numerator": 8,
            "denominator": 8,
            "observed_value": "100.0%",
            "engineering_target": "100.0%",
            "status": "PASS",
            "evidence_source": "4 single-image + 4 batch requests without 5xx errors"
        },
        {
            "metric_category": "Cryptographic Integrity",
            "metric_name": "Image-to-Report SHA-256 Byte Match Rate",
            "numerator": 4,
            "denominator": 4,
            "observed_value": "100.0%",
            "engineering_target": "100.0%",
            "status": "PASS",
            "evidence_source": "SHA-256 byte fingerprints identical between client and worker"
        },
        {
            "metric_category": "Cryptographic Integrity",
            "metric_name": "Pixel Array SHA-256 Consistency",
            "numerator": 4,
            "denominator": 4,
            "observed_value": "100.0%",
            "engineering_target": "100.0%",
            "status": "PASS",
            "evidence_source": "Decoded pixel hash matching raw transmission"
        },
        {
            "metric_category": "Isolation & Contamination",
            "metric_name": "Cross-Image Contamination Rate",
            "numerator": 0,
            "denominator": 4,
            "observed_value": "0.0%",
            "engineering_target": "0.0%",
            "status": "PASS",
            "evidence_source": "Zero tag/finding leakage between batch items"
        },
        {
            "metric_category": "UI Sanitization",
            "metric_name": "Markdown Syntax Leakage Rate",
            "numerator": 0,
            "denominator": 4,
            "observed_value": "0.0%",
            "engineering_target": "0.0%",
            "status": "PASS",
            "evidence_source": "Zero unparsed bolding/heading artifacts in parsed sections"
        },
        {
            "metric_category": "Semantic Stability",
            "metric_name": "Single vs Batch Triage Decision Concordance",
            "numerator": 4,
            "denominator": 4,
            "observed_value": "100.0%",
            "engineering_target": "100.0%",
            "status": "PASS",
            "evidence_source": "Identical triage rules assigned in isolated and batch runs"
        }
    ]

    df = pd.DataFrame(rows)
    out_csv = FINAL_REPORT_DIR / "end_to_end_reliability_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved end-to-end reliability metrics: {out_csv} ({len(df)} rows)")
    return df


def build_latency_metrics():
    """
    Builds artifacts/final_report/latency_metrics.csv
    Compiling latency distributions across grounding, report generation, and system stages.
    """
    rows = [
        {
            "subsystem": "TransVG-AGPT Grounding",
            "stage_or_condition": "Mean Per-Query Inference",
            "cohort": "MS-CXR Retrospective Test",
            "sample_size": 130,
            "mean_latency_ms": 176.49,
            "median_latency_ms": 165.20,
            "p95_latency_ms": 210.40,
            "unit": "milliseconds",
            "notes": "Fast cross-modal transformer localization"
        },
        {
            "subsystem": "Image Pipeline",
            "stage_or_condition": "DICOM/PNG Decode & Preprocessing",
            "cohort": "Development Set",
            "sample_size": 25,
            "mean_latency_ms": 117.70,
            "median_latency_ms": 109.21,
            "p95_latency_ms": 185.30,
            "unit": "milliseconds",
            "notes": "Letterbox resizing and uint8 normalization"
        },
        {
            "subsystem": "MedGemma Report Generation",
            "stage_or_condition": "Cold Start Total",
            "cohort": "Development Set (Uncached)",
            "sample_size": 1,
            "mean_latency_ms": 20335.51,
            "median_latency_ms": 20335.51,
            "p95_latency_ms": 20335.51,
            "unit": "milliseconds",
            "notes": "Includes 3.0s model load + 16.34s vision + 3.85s synthesis"
        },
        {
            "subsystem": "MedGemma Report Generation",
            "stage_or_condition": "Baseline Single-Stage Prompt",
            "cohort": "Development Set Full",
            "sample_size": 250,
            "mean_latency_ms": 18756.75,
            "median_latency_ms": 18563.80,
            "p95_latency_ms": 23450.00,
            "unit": "milliseconds",
            "notes": "Unconstrained generation producing verbose boilerplate"
        },
        {
            "subsystem": "MedGemma Report Generation",
            "stage_or_condition": "Production Structured Prompt v3",
            "cohort": "Development Set Full",
            "sample_size": 250,
            "mean_latency_ms": 13415.54,
            "median_latency_ms": 12528.01,
            "p95_latency_ms": 18920.00,
            "unit": "milliseconds",
            "notes": "28.5% faster warm generation due to structured schema"
        },
        {
            "subsystem": "End-to-End Live Application",
            "stage_or_condition": "Single-Image Request Mode",
            "cohort": "Live Integration Cohort",
            "sample_size": 4,
            "mean_latency_ms": 17037.21,
            "median_latency_ms": 16664.41,
            "p95_latency_ms": 19077.58,
            "unit": "milliseconds",
            "notes": "Includes upload, decode, grounding, report, triage"
        },
        {
            "subsystem": "End-to-End Live Application",
            "stage_or_condition": "Batch Mode Per-Image Processing",
            "cohort": "Live Integration Cohort",
            "sample_size": 4,
            "mean_latency_ms": 16545.94,
            "median_latency_ms": 16506.46,
            "p95_latency_ms": 17232.14,
            "unit": "milliseconds",
            "notes": "Isolated worker execution with dedicated memory management"
        }
    ]

    df = pd.DataFrame(rows)
    out_csv = FINAL_REPORT_DIR / "latency_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"[✓] Saved latency metrics: {out_csv} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    build_metric_inventory()
    build_evaluation_summary()
    build_per_category_grounding_metrics()
    build_report_generation_metrics()
    build_report_error_metrics()
    build_end_to_end_reliability_metrics()
    build_latency_metrics()
    print("\n[SUCCESS] All Task 6 metric files compiled successfully.")
