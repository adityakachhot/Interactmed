# InteractMed-VLM: Expanded Evaluation Cohort Reconciliation

## 1. Executive Summary & Audit Context

A critical evaluation discrepancy was identified following the initial benchmark of the expanded report-generation cohort:
- **Reported Benchmark:** The initial evaluation comparison reported results for $N=30$ cases (19 abnormal, 11 normal), achieving a reported Baseline Micro-F1 of 0.5000 and Prompt v3 Micro-F1 of 0.3415.
- **Cohort Inventory:** The expanded development cohort partition (`expanded_dev`) actually contains **250 images across 248 unique patients**, and the expanded held-out test partition (`expanded_test`) contains **117 images across 117 unique patients**.
- **Audit Mandate:** Investigate whether the 30-case evaluation was caused by data corruption, filtering rules, missing reference reports, or a script-level execution cap, and establish an uncompromised evaluation on all eligible development cases.

---

## 2. Cohort Holdings & Patient Mapping Verification

An exhaustive programmatic audit was performed across local image files, Chest-ImaGenome Gold annotations, and MIMIC-CXR metadata.

### 2.1 Inventory Summary

| Repository Asset | Record Count | Unique Identifiers | Integrity Status |
| :--- | :---: | :---: | :--- |
| **Local MIMIC-CXR-JPG Files** | 1,174 | 1,174 unique DICOM IDs | 100% valid JPEG files on local disk |
| **Chest-ImaGenome Gold Sentences** | 10,769 | 1,000 studies, 500 patients | Valid clinical sentences with section tags |
| **Chest-ImaGenome Gold Attributes** | 24,534 | 1,000 studies | Bounding boxes + coordinate annotations |
| **Chest-ImaGenome View Catalog** | 1,000 | 1,000 studies | Frontal view positions (AP / PA) |
| **Local Chest-ImaGenome Matches** | **417** | **417 unique images** | **100% frontal: 264 AP, 153 PA; 0 corrupt** |

### 2.2 Quarantining Historical Benchmark

To prevent benchmark contamination and data leakage:
- **Historical 50-Case Benchmark:** 25 development cases (`CASE_DEV_001`–`025`) and 25 exploratory test cases (`CASE_TEST_001`–`025`) from earlier research were audited.
- **Patient Isolation:** All 50 historical cases belong to 50 distinct patients.
- **Quarantine Policy:** These 50 patients and their images were strictly excluded from the expanded candidate pool.

### 2.3 Partition Allocation & Distribution

The remaining 367 candidate images were partitioned with strict patient grouping (zero patient overlap between partitions):

| Partition | Total Images | Unique Patients | Unique Studies | Frontal Views (AP / PA) | Category Breakdown | Purpose & Governance |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **`expanded_dev`** | **250** | **248** | **250** | 154 AP / 96 PA | 94 Pneumonia, 71 Normal, 57 Edema, 10 Devices, 9 Atelectasis, 9 Cardiomegaly | Active development, prompt validation, baseline comparison |
| **`expanded_test`** | **117** | **117** | **117** | 75 AP / 42 PA | 44 Pneumonia, 33 Normal, 26 Edema, 6 Cardiomegaly, 4 Atelectasis, 4 Devices | **Strictly locked & untouched** final test cohort |
| **Total Expanded** | **367** | **365** | **367** | 229 AP / 138 PA | Full multi-category clinical balance | Patient-grouped, 0 patient overlap |

- **Patient Grouping:** In `expanded_dev`, 246 patients have exactly 1 frontal radiograph, and 2 patients have 2 frontal radiographs from separate studies. Both multi-image patients are completely contained within `expanded_dev`.
- **Zero Patient Leakage:** $\text{Patients}(\text{expanded\_dev}) \cap \text{Patients}(\text{expanded\_test}) = \emptyset$.
- **Zero Historical Overlap:** $\text{Patients}(\text{expanded\_cohort}) \cap \text{Patients}(\text{historical}) = \emptyset$.

---

## 3. Discrepancy Audit: Why Only 30 Cases Were Evaluated

### 3.1 Investigation Findings

1. **Unique IDs in Development Partition:**
   - Manifest: `artifacts/report_evaluation/expanded_cohort/expanded_manifest.csv` contains exactly **250 rows** with `split == "expanded_dev"`.
   - Reference Store: `artifacts/report_evaluation/expanded_cohort/expanded_reference_store.json` contains exactly **250 records** with `split == "expanded_dev"`.
   - All 250 cases have `safe_case_id` values from `CASE_EXP_DEV_001` to `CASE_EXP_DEV_250`.

2. **Cases Evaluated in Initial Benchmark:**
   - Exactly **30 cases** (`CASE_EXP_DEV_001` through `CASE_EXP_DEV_030`) were evaluated.
   - Denominator breakdown: 19 abnormal cases and 11 normal cases ($19 + 11 = 30$).

3. **Root Cause Analysis:**
   - Review of the process execution transcript (`task-3524`) revealed that `scripts/evaluate_expanded_reports.py` was invoked with an explicit command-line limit:
     ```bash
     python scripts/evaluate_expanded_reports.py --split expanded_dev --limit 30
     ```
   - In `scripts/evaluate_expanded_reports.py`, the CLI parser defines `--limit` as an integer cap:
     ```python
     if limit:
         cohort_cases = cohort_cases[:limit]
     ```
   - The `--limit 30` option was originally intended as a smoke/sub-cohort test during script development. However, the resulting outputs (`expanded_comparison.csv` and `expanded_summary.json`) were documented as the evaluation results without subsequently executing the remaining 220 cases of `expanded_dev`.
   - **Conclusion:** There was **no data corruption, no missing file, and no automated exclusion filter**. All 220 remaining cases were simply unexecuted due to the `--limit 30` CLI parameter.

4. **Pairing Integrity Check:**
   - Verified that both the Baseline and Prompt v3 pipelines evaluated the **exact same 30 cases** in lockstep.
   - Both pipelines received the identical decoded RGB array, identical 1024px maximum-side resize, and identical JPEG base64 payload.

5. **Reference Report Verification:**
   - 100% of the 250 cases in `expanded_dev` have verified local image files on disk (0 missing).
   - 100% of the 250 cases have non-empty gold reference reports assembled from Chest-ImaGenome sentence annotations (0 empty, word counts range from 24 to 252 words, median 78 words).
   - All 250 cases are valid, eligible image-report pairs.

---

## 4. Preservation & Versioning Strategy

To maintain strict scientific provenance:
1. **Preserve Historical Artifacts:** The original 30-case exploratory outputs are preserved in `artifacts/report_evaluation/expanded_cohort/`.
2. **Dedicated Versioned Directory:** The complete development evaluation is targeted to a new versioned directory:
   ```
   artifacts/report_evaluation/expanded_cohort_full/
   ```
3. **Resilience & Checkpointing:** The evaluation script is upgraded with:
   - `--output-dir` argument to direct outputs to `artifacts/report_evaluation/expanded_cohort_full/`.
   - Progressive per-case saving: `expanded_comparison.csv`, `expanded_case_results.json`, and `expanded_summary.json` are flushed to disk after every single case.
   - `--resume` support: Checks already evaluated `safe_case_id`s in `output_dir` so completed evaluations are never recomputed.

---

## 5. Audit Conclusion

The candidate pool and development partition are fully validated. All 250 cases in `expanded_dev` represent eligible, uncorrupted, frontal radiograph image-report pairs with zero patient overlap with the held-out test cohort or historical benchmarks. The 30-case evaluation was strictly an artifact of `--limit 30`, and the complete evaluation on all 250 development cases is fully actionable.
