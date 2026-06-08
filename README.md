# Solana Rug-Pull Risk Detection

This repository studies Solana rug-pull risk detection from on-chain token and wallet behavior. The goal is not to claim a perfect rug-pull oracle, but to evaluate how weak labels, temporal shift, external RugCheck validation, probability calibration, alert ranking, and GraphSAGE-style graph features affect practical risk detection.

## Final Framing

The final report frames the project as **Solana rug-pull risk detection under weak labels and temporal shift**. Early labels are noisy behavioral proxies, so final conclusions are checked against a rebuilt RugCheck validation set. The final analysis emphasizes:

- Weak-label sensitivity: comparing strict, relaxed, union, intersection, and consensus label variants.
- Temporal shift: training or scoring across 2024 and evaluating on 2025-style token behavior.
- RugCheck validation: using `rugcheck_model_validation_master.csv` as the external validation source of truth.
- Calibration: checking threshold behavior, calibration bins, and predicted positive rates.
- Ranking: evaluating top-k enrichment and alert-budget usefulness rather than only fixed-threshold classification.
- GraphSAGE ablation: comparing token-only, graph-only/GraphSAGE, and combined feature views where final artifacts exist.

## Repository Structure

```text
.
|-- data/
|   |-- raw/                         # Original SolRPDS and Dune CSV inputs.
|   |-- bronze/                      # Dune raw-to-bronze parquet output.
|   |-- silver/                      # Cleaned Dune token events and token-wallet edges.
|   |-- gold/                        # Token features, PyG graph tables, and GraphSAGE scores/embeddings.
|   `-- results/                     # Intermediate experiment outputs; see warning below.
|-- figures/
|   `-- final/                       # Final report figures used for conclusions.
|-- outputs/
|   `-- final/                       # Final report CSV/Markdown artifacts used for conclusions.
|-- src/                             # Reproducible data, modeling, validation, and final assembly scripts.
|-- archive/                         # Old archived outputs, not used for final conclusions.
|-- README.md                        # GitHub/instructor-facing project guide.
`-- requirements.txt                 # Python dependencies inferred from project imports.
```

Important scripts in `src/`:

- `dune_polars_lakehouse_pipeline.py`: builds Dune bronze/silver/gold parquet tables from monthly raw Dune CSVs.
- `dune_2024_2025_eda_ml_gnn_pipeline.py`: earlier EDA, ML, and graph experiment pipeline.
- `dune_publishable_experiments_pipeline.py`: publishable 2024/2025 experiment layer, including weak-label and graph ablation artifacts.
- `build_rugcheck_ground_truth.py`, `run_rugcheck_batch_pipeline.py`, `rebuild_rugcheck_labels_from_cache.py`: RugCheck collection/cache/rebuild utilities.
- `evaluate_rugcheck_external_validation.py`: evaluates existing model scores against RugCheck validation.
- `run_final_rugcheck_graphsage_ablation.py`: final GraphSAGE/RugCheck ablation assembly.
- `generate_final_core_outputs.py`: final CSV summaries, metrics, audit, and run summary.
- `generate_final_report_figures.py`: final figures for the report.
- `generate_threshold_sweep.py`, `generate_feature_sanity_by_label.py`, `generate_cross_source_shift_summary.py`, `generate_disagreement_analysis.py`, `generate_alert_budget_simulation.py`: final supporting analyses.

## Final Artifacts

The GitHub submission should be reviewed from these folders:

- `outputs/final/`: final metrics, calibration, ranking, label sensitivity, disagreement, and audit tables.
- `figures/final/`: final figures for the submitted conclusions.

Final RugCheck source of truth:

- `data/results/dune_publishable_2024_2025/rugcheck_model_validation_master.csv`

Key final output files:

- `outputs/final/final_run_summary.md`
- `outputs/final/final_artifact_audit.md`
- `outputs/final/token_labels_all_versions.csv`
- `outputs/final/weak_label_rugcheck_confusion.csv`
- `outputs/final/temporal_model_metrics.csv`
- `outputs/final/threshold_calibration_metrics.csv`
- `outputs/final/calibration_bins.csv`
- `outputs/final/topk_ranking_metrics.csv`
- `outputs/final/alert_budget_simulation.csv`
- `outputs/final/label_sensitivity_summary.csv`
- `outputs/final/feature_sanity_by_label.csv`
- `outputs/final/cross_source_shift_summary.csv`
- `outputs/final/disagreement_summary.csv`
- `outputs/final/disagreement_cases.csv`
- `outputs/final/threshold_sweep_metrics.csv`
- `outputs/final/graphsage_ablation.csv`
- `outputs/final/graphsage_final_metrics.csv`
- `outputs/final/graphsage_final_topk.csv`

Key final figures:

- `figures/final/weak_label_vs_rugcheck_confusion.png`
- `figures/final/temporal_shift_2024_2025.png`
- `figures/final/calibration_curve.png`
- `figures/final/topk_enrichment.png`
- `figures/final/alert_budget_simulation.png`
- `figures/final/label_sensitivity.png`
- `figures/final/graphsage_final_ablation.png`
- `figures/final/feature_sanity_by_label.png`
- `figures/final/disagreement_quadrants.png`
- `figures/final/cross_source_shift_summary.png`

## Important Warning About Old Outputs

Only `outputs/final/`, `figures/final/`, and the explicitly listed RugCheck validation files should be used for final conclusions. Older artifacts under `data/results/`, `archive/`, notebooks, report drafts, and intermediate experiment folders are retained locally for transparency and development history, but they are not the final GitHub evidence layer.

In particular, the final run summary notes that older RugCheck counts and stale retraining artifacts are superseded by the rebuilt RugCheck master with 165,259 usable labels. Missing RugCheck coverage is not treated as safe.

## Requirements and Setup

Python 3.11+ is recommended. Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`torch` and `torch-geometric` are included for GraphSAGE reproduction, but PyTorch Geometric may require an installation command matched to the local CPU/CUDA/Python environment. If installation fails, use the official PyTorch Geometric install selector and then rerun `pip install -r requirements.txt` for the remaining packages.

## Reproduction Guide

The final submission can be read directly from the checked-in final artifacts. To reproduce the final artifact layer from existing project inputs and intermediate outputs, the detectable final script order is:

```bash
python src/generate_final_report_figures.py
python src/generate_final_core_outputs.py
python src/generate_disagreement_analysis.py
python src/generate_threshold_sweep.py
python src/generate_feature_sanity_by_label.py
python src/generate_cross_source_shift_summary.py
python src/generate_alert_budget_simulation.py
```

For deeper reproduction from raw Dune/RugCheck inputs, the broader pipeline appears to be:

```bash
python src/dune_polars_lakehouse_pipeline.py
python src/dune_publishable_experiments_pipeline.py
python src/run_rugcheck_batch_pipeline.py
python src/rebuild_rugcheck_labels_from_cache.py
python src/evaluate_rugcheck_external_validation.py
python src/run_final_rugcheck_graphsage_ablation.py
```

The broader pipeline may be time-consuming and depends on local raw data, cached RugCheck responses, and optional GraphSAGE dependencies. The final packaging pass did not rerun RugCheck crawling or retrain the main token/GraphSAGE models.

## Known Limitations

- Weak labels are noisy behavioral proxies and should not be interpreted as ground truth.
- RugCheck is used as external validation, not as a perfect oracle.
- Missing or failed RugCheck API coverage is not assumed safe.
- Temporal shift between 2024 and 2025 can change score calibration and operating thresholds.
- Some older retraining artifacts are stale relative to the final RugCheck master and are excluded from final claims.
- GraphSAGE results depend on graph construction choices, available dependencies, and existing score artifacts.
- Final claims are based on available Dune/SolRPDS/RugCheck data, not complete Solana market coverage.

## Suggested Instructor Review Path

1. Read `outputs/final/final_run_summary.md` to understand the final data source, label counts, reused models, skipped items, and exclusions.
2. Check `outputs/final/final_artifact_audit.md` for the final artifact inventory.
3. Review figures in `figures/final/` for the final visual evidence.
4. Inspect `outputs/final/weak_label_rugcheck_confusion.csv`, `temporal_model_metrics.csv`, `threshold_calibration_metrics.csv`, `topk_ranking_metrics.csv`, and `graphsage_ablation.csv` for the main quantitative evidence.

## External Data Archive

Large raw and intermediate data files are not stored in this GitHub repository to keep the submission lightweight. They are available here:

[Google Drive data archive](https://drive.google.com/drive/folders/1Yq4swgzZghV7vkGSjp2mD7TjD0xLTBpT?usp=sharing)

The final conclusions in this repository are based on the checked-in final artifacts under `outputs/final/` and `figures/final/`. The external archive is provided only for full raw-data reproduction and audit.

## Team Contributions

### Nguyen Thi Phuong Thao (V202401781)

* Data collection and crawling
* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Graph construction and validation
* Visualization and result interpretation
* Report writing: Introduction, Data Overview

### Le Thao Vy (V202401694)

* Data collection and crawling
* Model development and implementation
* GraphSAGE training and evaluation
* Baseline comparison
* Hyperparameter tuning
* RugCheck external validation
* Report writing: Implementation Details, Empirical Results, Limitations

### Shared Contributions

* Literature review
* Research design
* Result discussion
* Conclusion writing
* Presentation preparation
* Final report revision
* Repository maintenance

