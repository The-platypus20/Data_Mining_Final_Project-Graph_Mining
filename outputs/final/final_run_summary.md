# Final Run Summary

## RugCheck Source Of Truth
- Latest RugCheck source file: `data\results\dune_publishable_2024_2025\rugcheck_model_validation_master.csv`
- Total rows after 24h window de-duplication: 167,507
- Unique token/year rows: 167,507
- Usable RugCheck labels: 165,259
- RugCheck risky: 135,617
- RugCheck safe: 29,642
- No usable RugCheck/API error: 2,248
- Year coverage: {2024: 66096, 2025: 101411}
- Clarification: The older 6,277-label count is stale. It came from an earlier RugCheck cache/build before the full batch was rebuilt. The latest source-of-truth master has 165,259 usable RugCheck labels. The 98-row retraining artifact is also stale/mismatched relative to the latest master, so it is not used as the source of truth.

## Input Files Used
- `data\results\dune_publishable_2024_2025\rugcheck_model_validation_master.csv`
- `data\results\dune_publishable_2024_2025\rugcheck_coverage_summary.csv`
- `data\results\dune_publishable_2024_2025\rugcheck_external_validation_summary.json`
- `data\results\dune_publishable_2024_2025\xgboost_window_ablation.csv`

## Label Counts
- weak_strict: positives=739, negatives=166,768, missing/uncertain=0
- weak_relaxed: positives=163,836, negatives=3,671, missing/uncertain=0
- rugcheck_binary: positives=135,617, negatives=29,642, missing/uncertain=2,248
- label_union: positives=135,866, negatives=29,399, missing/uncertain=2,242
- label_intersection: positives=490, negatives=164,769, missing/uncertain=2,248
- label_consensus: positives=490, negatives=29,399, missing/uncertain=137,618

## Train/Test Split Used
- Main final validation metrics use Dune 2025 rows with RugCheck labels as the test set and existing Dune weak-label model score columns.
- RugCheck-supervised temporal retraining artifact is marked skipped/stale because the current retraining distribution says temporal split `not_run` and only 98 rows.

## Leakage Columns Excluded For RugCheck Target
- rugcheck_score, rugcheck_score_normalised, risk_count, danger_count, warn_count, risk_names, risk_levels, risk_scores, label_reason, api_status, api_ok, raw_json_path

## Models Actually Run Or Reused
- No new model training in this final-output run.
- Reused existing score column: rule_baseline / silver_label_score.
- Reused existing score column: old_weak_token_logistic / token_logistic_score.
- Reused existing score column: old_weak_xgboost_token / token_model_score.
- Reused existing score column: graphsage_direct / graphsage_score.
- Reused existing score column: old_weak_xgboost_token_graphsage / combined_model_score.
- Existing artifact reports score generation completed: {'experiment': 'token_only_logistic', 'model': 'logistic_regression', 'status': 'completed', 'feature_count': 13}
- Existing artifact reports score generation completed: {'experiment': 'token_only', 'model': 'xgboost', 'status': 'completed', 'feature_count': 13}
- Existing artifact reports score generation completed: {'experiment': 'token_graphsage_combined', 'model': 'xgboost', 'status': 'completed', 'feature_count': 29}

## Skipped Items And Reasons
- No RugCheck crawl was run; latest 165,259-label master was reused.
- RugCheck-supervised temporal retraining was not rerun; existing retraining artifact has only 98 rows and temporal split not_run.
- graph_feature_ablation.csv not created; no separate final temporal graph feature pipeline was run in this pass. Existing GraphSAGE ablation was copied to graphsage_ablation.csv.

## Final Output Files
- `outputs\final\calibration_bins.csv`
- `outputs\final\figure_token_source_year_labels.csv`
- `outputs\final\final_artifact_audit.csv`
- `outputs\final\final_artifact_audit.md`
- `outputs\final\final_run_summary.md`
- `outputs\final\graphsage_ablation.csv`
- `outputs\final\label_sensitivity_summary.csv`
- `outputs\final\lifespan_cumulative_curve_summary.csv`
- `outputs\final\temporal_model_metrics.csv`
- `outputs\final\threshold_calibration_metrics.csv`
- `outputs\final\token_distribution_by_source_year.csv`
- `outputs\final\token_labels_all_versions.csv`
- `outputs\final\topk_ranking_metrics.csv`
- `outputs\final\weak_label_rugcheck_confusion.csv`

## Final Figures
- `figures\final\behavior_separation.png`
- `figures\final\calibration_curve.png`
- `figures\final\feature_importance.png`
- `figures\final\label_sensitivity.png`
- `figures\final\lifespan_cumulative_curve.png`
- `figures\final\predicted_positive_rate_by_model.png`
- `figures\final\temporal_shift_2024_2025.png`
- `figures\final\token_distribution_by_source_year.png`
- `figures\final\topk_enrichment.png`
- `figures\final\weak_label_vs_rugcheck_confusion.png`


## Feature Sanity Analysis
- Output: `outputs/final/feature_sanity_by_label.csv`
- Figure: `figures/final/feature_sanity_by_label.png`
- Skipped missing feature columns: none
