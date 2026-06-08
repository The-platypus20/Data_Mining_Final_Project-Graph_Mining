# RugCheck-Aware GraphSAGE Final Ablation

Status: completed

Run date: 2026-06-08

## Source of Truth

- RugCheck usable labels: 165,259
- Labels/features: `outputs\final\token_labels_all_versions.csv`
- Existing GraphSAGE routine: `src/dune_publishable_experiments_pipeline.py::try_run_graphsage`
- Existing events: `data\silver\dune_token_events_2024_2025.parquet`
- Existing early-window inputs: `data\results\dune_publishable_2024_2025\early_window_silver_features.csv`

## Run Configuration

- Target label: `rugcheck_binary`
- Split: temporal_2024_train_2025_test
- Window: 1h, the smallest existing GraphSAGE early-window evaluation
- Threshold for precision/recall/F1: 0.5
- RugCheck-derived fields were not used as node features.

## Metrics

- n_train: 65,009
- n_test: 100,250
- ROC-AUC: 0.506942
- PR-AUC: 0.828242
- Precision: 0.826334
- Recall: 1.000000
- F1: 0.904910
- Predicted positive rate: 1.000000
- Brier score: 0.143704
- ECE: 0.010651

## Top-K

| target_label    | split                         |   window_hours |   top_k_percent |     k |   positives_in_top_k |   base_positive_rate |   precision_at_k |   enrichment_at_k |
|:----------------|:------------------------------|---------------:|----------------:|------:|---------------------:|---------------------:|-----------------:|------------------:|
| rugcheck_binary | temporal_2024_train_2025_test |              1 |               1 |  1003 |                  814 |             0.826334 |         0.811565 |          0.982127 |
| rugcheck_binary | temporal_2024_train_2025_test |              1 |               5 |  5013 |                 4218 |             0.826334 |         0.841412 |          1.01825  |
| rugcheck_binary | temporal_2024_train_2025_test |              1 |              10 | 10025 |                 8372 |             0.826334 |         0.835112 |          1.01062  |

## Existing Routine Metrics

|   precision |   recall |      f1 |   average_precision |   roc_auc |   tn |    fp |   fn |    tp |   window_hours | model     |   train_year |   test_year | status    |   train_token_labels |   train_positive_labels |
|------------:|---------:|--------:|--------------------:|----------:|-----:|------:|-----:|------:|---------------:|:----------|-------------:|------------:|:----------|---------------------:|------------------------:|
|    0.826334 |        1 | 0.90491 |            0.828242 |  0.506942 |    0 | 17410 |    0 | 82840 |              1 | graphsage |         2024 |        2025 | completed |                65009 |                   52777 |

## Outputs

- `outputs\final\graphsage_final_metrics.csv`
- `outputs\final\graphsage_final_topk.csv`
- `outputs\final\graphsage_final_run_summary.md`
- `outputs\final\graphsage_ablation.csv`
- `figures\final\graphsage_final_ablation.png`
