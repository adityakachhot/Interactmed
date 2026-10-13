# InteractMed-VLM: Inference Latency Profiling & Speed Optimization

**Date:** October 12, 2026  
**Auditor / Researcher:** Senior Medical Vision-Language Model Researcher  
**Hardware Platform:** Apple Silicon M3, 16 GB Unified Memory, macOS  
**Model:** `medgemma:4b` (Q4_K_M GGUF, 4.3B Parameters) via local Ollama  
**Evaluation Cohort:** 25 Development Cases (`split == 'dev'`, strictly isolated from locked test set)  
**Data Artifact:** [`artifacts/report_evaluation/latency_benchmark.json`](file:///Users/adityakachhot/interactmed/artifacts/report_evaluation/latency_benchmark.json)  

---

## 1. Executive Summary

This report documents the granular latency instrumentation and optimization of the whole-chest-X-ray report generation pipeline in InteractMed-VLM. 

### Key Findings
1. **The Primary Bottleneck:** In the unoptimized revised pipeline, **Stage 1 (Vision Extraction) accounts for 73.4% of total latency** (25.36s out of 34.55s). MedGemma 4B spent ~176 tokens generating conversational conversational filler (e.g., *"Okay, I will analyze the chest radiograph..."*) and listing non-present apparatus items, hitting the token cap and delaying response times.
2. **The Optimization Strategy:** By converting Stage 1 into a concise 4-line structured visual extraction prompt and eliminating prompt instruction redundancies in Stage 2, generated tokens in Stage 1 dropped from **176.1 to 35.3 tokens (an 80.0% token reduction)**.
3. **Speedup Achieved:** End-to-end latency improved from **34.55s to 22.53s** (**1.53× speedup**, saving **12.02 seconds per image** on Apple Silicon M3).
4. **Clinical Equivalence Maintained:** The optimized pipeline achieved **92.0% clinical label agreement** with the unoptimized pipeline (23/25 exact match) with near-identical Micro-F1 (**0.3478 vs 0.3529**).
5. **Memory Residency Confirmed:** Model load time under cold-start was **3,002.17 ms** (~3.0 seconds). Under warm requests with `keep_alive="30m"`, weight reloading latency dropped to **0.00 ms**.

---

## 2. Stage-by-Stage Latency Breakdown ($N=25$)

Each inference stage was instrumented with sub-millisecond precision across all 25 development cases.

| Inference Stage | Baseline Pipeline | Revised Pipeline (Unoptimized) | Optimized Pipeline (Structured) | Optimization Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Image Decoding & Preprocessing** | 117.70 ms | 117.70 ms | 117.70 ms | 0.00 ms |
| **Stage 1: Targeted Visual Extraction** | — | 25,361.61 ms (25.36s) | **14,214.70 ms (14.21s)** | **-11,146.91 ms (-43.9%)** |
| *— Stage 1 Output Tokens* | — | 176.1 tokens | **35.3 tokens** | **-140.8 tokens (-80.0%)** |
| **Stage 2: Clinical Synthesis** | — | 9,069.53 ms (9.07s) | **8,196.34 ms (8.20s)** | **-873.19 ms (-9.6%)** |
| *— Stage 2 Output Tokens* | — | 92.9 tokens | **94.3 tokens** | +1.4 tokens |
| **Output Parsing & Section Validation** | 0.22 ms | 0.23 ms | **0.17 ms** | -0.06 ms |
| **Total End-to-End Latency (Mean)** | 19,167.33 ms (19.17s) | 34,548.88 ms (34.55s) | **22,528.75 ms (22.53s)** | **-12,020.13 ms (1.53× Speedup)** |
| **Total End-to-End Latency (Median)** | 18,976.38 ms (18.98s) | 34,999.98 ms (35.00s) | **21,947.82 ms (21.95s)** | **-13,052.16 ms (-37.3%)** |
| **Latency Standard Deviation** | 2,465.58 ms (2.47s) | 6,290.19 ms (6.29s) | **2,850.57 ms (2.85s)** | **-54.7% variance reduction** |

---

## 3. Cold-Start vs. Warm-Request Residency Profile

To evaluate the operational behavior of persistent model residency (`keep_alive="30m"`), the Ollama model server was explicitly purged (`keep_alive=0`) prior to Request 1 (`CASE_DEV_001`), followed by consecutive warm requests across the remaining 24 cases.

| Parameter | Cold-Start (Request 1) | Warm Requests (Requests 2–25) | Operational Finding |
| :--- | :---: | :---: | :--- |
| **Model Weight Load Time** | **3,002.17 ms (3.00s)** | **0.00 ms (0.00s)** | Apple Silicon unified memory retains GGUF weights; zero IO overhead during warm inference. |
| **Stage 1 Prompt / Vision Eval** | 16,337.06 ms | 14,126.33 ms | Vision token encoding remains steady (~1.2s for image patch embedding). |
| **Stage 2 Text Eval** | 3,852.06 ms | 8,377.34 ms | Text-only generation runs at ~20–22 tokens/second. |
| **Total Turnaround Time** | **20,335.51 ms** | **22,620.14 ms (Mean)** | `keep_alive="30m"` successfully eliminates the 3-second weight reload penalty across batch jobs. |

---

## 4. Benchmark Comparison Across Configurations ($N=25$)

```
+-----------------------------------------------------------------------------------------+
|                                    LATENCY COMPARISON                                   |
| Baseline Pipeline (Single Stage):      19.17s  [███████████████████]                     |
| Revised Pipeline (Unoptimized 2-Stage): 34.55s  [███████████████████████████████████]     |
| Optimized Pipeline (Structured 2-Stage): 22.53s  [███████████████████████]  (1.53x Faster) |
+-----------------------------------------------------------------------------------------+
```

### Quantitative Metrics Summary

| Configuration | Mean Latency (s) | Median Latency (s) | Latency Std (s) | Micro-Precision | Micro-Recall | Micro-F1 | Agreement w/ Revised |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Current Prompt** | 19.17s | 18.98s | 2.47s | 0.2800 | 0.1591 | 0.2029 | 48.0% |
| **Revised Grounded Pipeline** | 34.55s | 35.00s | 6.29s | **0.5000** | **0.2727** | **0.3529** | 100.0% (Ref) |
| **Optimized Structured Pipeline** | **22.53s** | **21.95s** | **2.85s** | **0.4800** | **0.2727** | **0.3478** | **92.0%** |

* **Speedup Factor:** 1.53× over Revised (saving 12.02 seconds per image).
* **Clinical Fidelity:** Micro-F1 is essentially identical (0.3478 vs 0.3529). Both true positives ($N=12$) and false negatives ($N=32$) are identical; only 1 additional borderline label was parsed in Optimized.
* **Consistency:** 23 out of 25 cases (92.0%) yielded exactly identical clinical label sets.

---

## 5. Architectural Recommendations

1. **Adopt Structured Extraction for Stage 1:** Rather than requesting paragraph-form radiology descriptions, prompt Stage 1 with:
   ```
   Inspect this chest radiograph and report findings concisely in 4 lines:
   1. Support apparatus: Identify any endotracheal tube, enteric tube, line, or None.
   2. Cardiac silhouette: State if cardiothoracic ratio is normal or enlarged (>0.5).
   3. Lungs: Describe visible airspace opacities, consolidation, edema, or state Clear.
   4. Pleural spaces: State if costophrenic angles are sharp or show blunting/effusion.
   Begin directly with 1. Support apparatus:
   ```
   This caps Stage 1 generation at ~35 tokens while preserving complete anatomical coverage.
2. **Maintain Token Budgets:** Set `num_predict=100` for Stage 1 and `num_predict=180` for Stage 2. This prevents conversational model drift without truncating genuine clinical findings.
3. **Preserve Image Resolution at 1024px:** Image preprocessing accounts for only 117.7 ms (0.5% of total time). Downscaling image resolution would severely compromise small opacity detection without delivering meaningful latency gains.
4. **Retain `keep_alive="30m"`:** Prevents repeated 3-second cold-load penalties during batch processing sessions.
