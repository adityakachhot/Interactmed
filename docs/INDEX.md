# InteractMed-VLM Documentation Index

Welcome to the comprehensive technical documentation and architecture reference package for **InteractMed-VLM** — a Vision-Language System for Chest X-Ray Grounding, Report Generation, Patient-Friendly Explanation, Batch Processing, and Research-Only Triage.

---

## 🏛️ System Architecture Documents

The primary system documentation is organized into modular, audited reference files:

| Document | Primary Focus & Coverage | Target Audience |
| :--- | :--- | :--- |
| **[`docs/ARCHITECTURE.md`](file:///Users/adityakachhot/interactmed/docs/ARCHITECTURE.md)** | **Primary System Architecture Specification.** Complete overview, Mermaid context diagrams, component responsibility table, deployment architecture (local vs cloud), and data security. | Lead Architects, Systems Engineers, Academic Supervisors |
| **[`docs/architecture/MODELS_AND_METHODS.md`](file:///Users/adityakachhot/interactmed/docs/architecture/MODELS_AND_METHODS.md)** | **Authoritative Model & Methodology Reference.** Model inventory (TransVG-AGPT, MedGemma 4B, fallbacks), mathematical formulations, forward passes, two-stage grounded reasoning, and evaluation equations. | Machine Learning Engineers, CV/NLP Researchers |
| **[`docs/architecture/FINE_TUNING_AND_TRAINING.md`](file:///Users/adityakachhot/interactmed/docs/architecture/FINE_TUNING_AND_TRAINING.md)** | **Training & Fine-Tuning Runbook.** Complete epoch-by-epoch RSNA grounding training logs, MedGemma 4B 4-bit QLoRA preflight checks, gradient routing, and master experiment tables. | ML Training Engineers, Cloud GPU Operators |
| **[`docs/architecture/EVALUATION_AND_IMPROVEMENTS.md`](file:///Users/adityakachhot/interactmed/docs/architecture/EVALUATION_AND_IMPROVEMENTS.md)** | **Research Evaluation & Improvements Report.** IoU metrics on 901 RSNA test cases, full 250-case development report benchmarks, Wilson confidence intervals, and the master improvements matrix. | Research Scientists, Clinical Evaluators |
| **[`docs/architecture/DATASETS_AND_RESEARCH_GOVERNANCE.md`](file:///Users/adityakachhot/interactmed/docs/architecture/DATASETS_AND_RESEARCH_GOVERNANCE.md)** | **Datasets, Manifests & Ethics Reference.** Detailed audit of RSNA, Chest-ImaGenome Gold, MS-CXR, and MIMIC-CXR; patient-disjoint split governance; multi-view vs single-view audit. | Data Engineers, Compliance Officers |
| **[`docs/architecture/API_UI_AND_WORKFLOWS.md`](file:///Users/adityakachhot/interactmed/docs/architecture/API_UI_AND_WORKFLOWS.md)** | **Runtime Workflows & API Reference.** Step-by-step execution traces for grounding, reporting, batch processing, and research triage; double-blind annotation protocol; master REST API table. | Full-Stack Engineers, API Integrators |
| **[`docs/architecture/OPERATIONS_AND_REPRODUCIBILITY.md`](file:///Users/adityakachhot/interactmed/docs/architecture/OPERATIONS_AND_REPRODUCIBILITY.md)** | **Engineering Runbook & Operations Guide.** Installation instructions, environment requirements, Ollama setup, test suite execution (104 tests), smoke tests, and troubleshooting runbook. | DevOps Engineers, Reproducibility Auditors |
| **[`docs/architecture/ROADMAP_AND_LIMITATIONS.md`](file:///Users/adityakachhot/interactmed/docs/architecture/ROADMAP_AND_LIMITATIONS.md)** | **Limitations & Prioritized Technical Roadmap.** Objective failure mode analysis, opacity ceiling investigation, and 8 prioritized research milestones. | Project Managers, Principal Investigators |

---

## 📊 Standalone Mermaid Architecture Diagrams

Standalone Mermaid (`.mmd`) source files are available in `docs/architecture/diagrams/`:
- [`01_system_context.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/01_system_context.mmd) — High-level system context, users, and boundaries.
- [`02_component_architecture.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/02_component_architecture.mmd) — Class diagram and service interactions.
- [`03_grounding_sequence.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/03_grounding_sequence.mmd) — End-to-end phrase grounding sequence.
- [`04_report_generation_sequence.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/04_report_generation_sequence.mmd) — Two-stage grounded report synthesis sequence.
- [`05_batch_processing_sequence.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/05_batch_processing_sequence.mmd) — Bounded batch processing queue sequence.
- [`06_training_and_finetuning.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/06_training_and_finetuning.mmd) — Separate TransVG and MedGemma training tracks.
- [`07_evaluation_workflow.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/07_evaluation_workflow.mmd) — Development selection vs. locked final test gating.
- [`08_triage_workflow.mmd`](file:///Users/adityakachhot/interactmed/docs/architecture/diagrams/08_triage_workflow.mmd) — Deterministic triage rule engine flow.

---

## 🔍 Historical & Specialized Technical Reports

For historical lineage and specific ablation logs:
- [`FINAL_RSNA_EVALUATION.md`](file:///Users/adityakachhot/interactmed/docs/FINAL_RSNA_EVALUATION.md) — Original RSNA test evaluation report.
- [`EXPANDED_REPORT_EVALUATION.md`](file:///Users/adityakachhot/interactmed/docs/EXPANDED_REPORT_EVALUATION.md) — 250-case report benchmark audit.
- [`EXPANDED_EVALUATION_RECONCILIATION.md`](file:///Users/adityakachhot/interactmed/docs/EXPANDED_EVALUATION_RECONCILIATION.md) — Audit of the 30-case vs 250-case discrepancy.
- [`MEDGEMMA_FINE_TUNING_RESULTS.md`](file:///Users/adityakachhot/interactmed/docs/MEDGEMMA_FINE_TUNING_RESULTS.md) — MedGemma preflight safety checks and single-batch overfit logs.
- [`MEDGEMMA_FINETUNING_RUNBOOK.md`](file:///Users/adityakachhot/interactmed/docs/MEDGEMMA_FINETUNING_RUNBOOK.md) — Step-by-step cloud training instructions for Google Colab.
- [`RESOLUTION_ABLATION.md`](file:///Users/adityakachhot/interactmed/docs/RESOLUTION_ABLATION.md) — 1024px vs 1536px resolution ablation results.
- [`TRIAGE_CLASSIFICATION.md`](file:///Users/adityakachhot/interactmed/docs/TRIAGE_CLASSIFICATION.md) — Triage system design and clinical annotation protocol.
- [`BATCH_REPORTS.md`](file:///Users/adityakachhot/interactmed/docs/BATCH_REPORTS.md) — Batch processing speed profiling and memory optimization report.
