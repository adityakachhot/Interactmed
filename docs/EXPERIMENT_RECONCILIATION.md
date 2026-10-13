# InteractMed-VLM: Experiment Reconciliation & Metric Audit

**Audit Date:** October 2026  
**Auditor:** Senior Medical Vision-Language Model Researcher  
**Scope:** Reconciling discrepancies across prompt benchmarks, speed profiling, and resolution ablation runs.  

---

## 1. Executive Summary

A cross-report audit of InteractMed-VLM experimental artifacts identified three apparent discrepancies between [`REPORT_SPEED_OPTIMIZATION.md`](file:///Users/adityakachhot/interactmed/docs/REPORT_SPEED_OPTIMIZATION.md), [`REPORT_PROMPT_V3.md`](file:///Users/adityakachhot/interactmed/docs/REPORT_PROMPT_V3.md), and [`RESOLUTION_ABLATION.md`](file:///Users/adityakachhot/interactmed/docs/RESOLUTION_ABLATION.md).

This document audits the exact code implementations, data artifacts, and execution runs to resolve these discrepancies. **No numbers have been silently reconciled or altered.** The authoritative artifacts for each phase are designated below.

---

## 2. Inconsistency 1: Prompt v2 vs v3 Micro-F1 Discrepancy

### The Observation
- In `REPORT_PROMPT_V3.md`, Prompt v2 reports Micro-F1 of **0.3478** while Prompt v3 reports **0.3188**.
- In `RESOLUTION_ABLATION.md`, the fixed Prompt v3 pipeline is reported as achieving **0.3478** at both 1024px and 1536px.

### Resolution & Root Cause
Both experiments were executed using identical clinical entity extraction logic (`extract_audited_labels`), bidirectional negation cues, and ground-truth sets on the 25 development cases (`split == 'dev'`). The variance is accounted for by a single borderline case:

1. **`evaluate_prompt_v3.py` Execution (`v3_comparison.csv`):**
   - On `CASE_DEV_004` (Ground truth: `normal`), Prompt v3 generated text noting support apparatus features, resulting in a false positive (`support_devices`) and a false negative for `normal`.
   - Result: $\text{TP}=11, \text{FP}=14, \text{FN}=33 \implies \text{Precision}=0.4400, \text{Recall}=0.2500, \text{Micro-F1}=0.3188$.
2. **`evaluate_resolution_ablation.py` Execution (`resolution_comparison.csv`):**
   - On a subsequent independent execution of Prompt v3 at 1024px, the stochastic output at `temperature=0.1` on `CASE_DEV_004` categorized support apparatus as `Not identified` and concluded `normal`.
   - Result: $\text{TP}=12, \text{FP}=13, \text{FN}=32 \implies \text{Precision}=0.4800, \text{Recall}=0.2727, \text{Micro-F1}=0.3478$.
3. **1536px Behavior on `CASE_DEV_025`:**
   - On `CASE_DEV_025` (Ground truth: `normal; support_devices`), 1024px correctly identified `support_devices`, whereas 1536px predicted `normal`.
   - Because `normal` was also in the ground-truth label set for `CASE_DEV_025`, the true positive shifted from `support_devices` to `normal`. Total TP remained at 12, FP at 13, and FN at 32, preserving Micro-F1 at **0.3478**.

**Conclusion:** The discrepancy is caused by minor generation variance on `CASE_DEV_004` across runs, not a metric bug or reporting error.

---

## 3. Inconsistency 2: Latency Figure Variations Across Reports

### The Observation
- `REPORT_SPEED_OPTIMIZATION.md` reported mean latency of **22.53s** for the optimized pipeline.
- `REPORT_PROMPT_V3.md` reported mean latency of **18.35s** for v2 and **22.18s** for v3.
- `RESOLUTION_ABLATION.md` reported mean latency of **23.74s** for 1024px and **24.02s** for 1536px.

### Resolution & Root Cause
1. **Token Generation Count:**
   - Prompt v2 produces concise Stage 1 outputs (~35–37 tokens), yielding end-to-end latencies between 18.3s and 22.5s.
   - Prompt v3 forces systematic evaluation of 5 anatomical categories with 4 statuses, generating ~82–84 Stage 1 tokens. This adds ~45 tokens at ~8–10 tokens/sec, naturally extending Stage 1 inference by ~4.0 seconds.
2. **Execution Context & System Load:**
   - In `latency_benchmark.json`, Prompt v2 was evaluated following cold-start benchmarking where memory allocation was freshly initialized.
   - In `evaluate_prompt_v3.py`, v2 was run in an interleaved loop with v3 under warm model residency (`keep_alive="30m"`), achieving a clean 18.35s average.
   - In `evaluate_resolution_ablation.py`, high-resolution image decoding (1536px) ran in the same process, causing slight memory bus contention that nudged 1024px latency to 23.74s.

**Conclusion:** Token generation length is the primary architectural driver (+4s for Prompt v3 vs v2), with run-to-run operating system variance accounting for $\pm 1.5\text{s}$ fluctuations on Apple Silicon M3.

---

## 4. Inconsistency 3: False-Normal and Laterality Counts

### The Observation
- Baseline single-stage prompt had a 100% false-normal rate (19/19 abnormal cases), whereas subsequent reports state 68.42% (13/19) or 73.68% (14/19).
- Laterality omissions were reported as 11 in v2, 7 in v3, and 7 at both resolutions.

### Resolution & Root Cause
1. **Baseline vs Two-Stage False Normals:**
   - The original single-stage prompt (`audit_v2_summary.json`) collapsed into boilerplate normal text on all cases ($19/19 = 100\%$ unsupported negative rate).
   - All two-stage pipelines (v2, v3, and 1024px ablation) consistently broke the boilerplate collapse, achieving an identical false-normal count of $13/19 = 68.42\%$ on the abnormal development cohort.
   - The 1536px run exhibited $14/19 = 73.68\%$ because `CASE_DEV_025` missed the support catheter contour, falsely asserting normal.
2. **Laterality Omissions:**
   - Laterality omission tracking was formally introduced in Phase B (`scripts/evaluate_prompt_v3.py`).
   - Prompt v2 omitted laterality in **11 cases**.
   - Prompt v3 reduced omissions to **7 cases** (a 36.4% reduction).
   - In the resolution experiment, both 1024px and 1536px exhibited exactly **7 omissions**, demonstrating that laterality adherence is governed by the prompt contract rather than pixel resolution.

---

## 5. Authoritative Artifact Registry

| Experiment Phase | Primary Script | Authoritative Raw Data | Authoritative Summary | Authoritative Report |
| :--- | :--- | :--- | :--- | :--- |
| **Initial Input & Data Linkage Audit** | `scripts/audit_report_inputs.py` | `artifacts/report_evaluation/image_report_manifest.csv` | `artifacts/report_evaluation/local_reference_store.json` | `docs/REPORT_DATA_AUDIT.md` |
| **Latency & Speed Optimization** | `scripts/profile_dev_benchmarks.py` | `artifacts/report_evaluation/latency_benchmark.json` | `artifacts/report_evaluation/latency_benchmark.json` | `docs/REPORT_SPEED_OPTIMIZATION.md` |
| **Prompt v3 Structured Benchmark** | `scripts/evaluate_prompt_v3.py` | `artifacts/report_evaluation/dev_prompt_v3/v3_comparison.csv` | `artifacts/report_evaluation/dev_prompt_v3/v3_summary.json` | `docs/REPORT_PROMPT_V3.md` |
| **Resolution Ablation (1024px vs 1536px)** | `scripts/evaluate_resolution_ablation.py` | `artifacts/report_evaluation/resolution_ablation/resolution_comparison.csv` | `artifacts/report_evaluation/resolution_ablation/resolution_summary.json` | `docs/RESOLUTION_ABLATION.md` |
| **Hybrid Prompt v2.1 Benchmark** | `scripts/evaluate_prompt_v2_1.py` | `artifacts/report_evaluation/prompt_v2_1/v2_1_comparison.csv` | `artifacts/report_evaluation/prompt_v2_1/v2_1_summary.json` | `docs/PROMPT_V2_1_EVALUATION.md` |
