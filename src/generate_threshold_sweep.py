from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"

LABELS_PATH = OUTPUT_DIR / "token_labels_all_versions.csv"
MASTER_PATH = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025" / "rugcheck_model_validation_master.csv"
OUT_PATH = OUTPUT_DIR / "threshold_sweep_metrics.csv"
FIGURE_PATH = FIGURE_DIR / "threshold_sweep.png"

THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
TARGET_COLUMNS = ["rugcheck_binary", "weak_strict", "weak_relaxed"]
SCORE_COLUMNS = {
    "rule_baseline": "silver_label_score",
    "old_weak_token_logistic": "token_logistic_score",
    "old_weak_xgboost_token": "token_model_score",
    "graphsage_direct": "graphsage_score",
    "old_weak_xgboost_token_graphsage": "combined_model_score",
}


def log(message: str) -> None:
    print(f"[threshold-sweep] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def load_frame() -> pd.DataFrame:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH, low_memory=False)
    labels["token_address"] = labels["token_address"].astype(str)
    labels["year"] = pd.to_numeric(labels["year"], errors="coerce").astype("Int64")

    if MASTER_PATH.exists():
        header = pd.read_csv(MASTER_PATH, nrows=0).columns
        usecols = [column for column in ["token_address", "year", "window_hours", *SCORE_COLUMNS.values()] if column in header]
        scores = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
        if "window_hours" in scores.columns:
            scores = scores[scores["window_hours"].eq(24)].copy()
        scores["token_address"] = scores["token_address"].astype(str)
        scores["year"] = pd.to_numeric(scores["year"], errors="coerce").astype("Int64")
        scores = scores.drop_duplicates(["year", "token_address"], keep="last")
        merge_cols = [column for column in scores.columns if column not in labels.columns or column in ["token_address", "year"]]
        labels = labels.merge(scores[merge_cols], on=["year", "token_address"], how="left")

    for column in [*TARGET_COLUMNS, *SCORE_COLUMNS.values()]:
        if column in labels.columns:
            labels[column] = pd.to_numeric(labels[column], errors="coerce")
    return labels


def sweep_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_scores = {model: column for model, column in SCORE_COLUMNS.items() if column in frame.columns}
    for target in TARGET_COLUMNS:
        if target not in frame.columns:
            continue
        y_all = pd.to_numeric(frame[target], errors="coerce")
        target_mask = y_all.isin([0, 1])
        if target == "rugcheck_binary":
            evaluation_target = "rugcheck_binary"
            split_name = "all_usable_rugcheck"
        else:
            missing_rugcheck = ~pd.to_numeric(frame.get("rugcheck_binary"), errors="coerce").isin([0, 1])
            if not missing_rugcheck.any():
                continue
            target_mask = target_mask & missing_rugcheck
            evaluation_target = target
            split_name = "rugcheck_missing_fallback"
        if not target_mask.any():
            continue

        for model, score_column in available_scores.items():
            scores = pd.to_numeric(frame.loc[target_mask, score_column], errors="coerce").clip(0, 1)
            y = y_all.loc[target_mask]
            valid = scores.notna() & y.isin([0, 1])
            if not valid.any():
                continue
            y_valid = y.loc[valid].astype(int)
            s_valid = scores.loc[valid].astype(float)
            true_positive_rate = float(y_valid.mean())
            for threshold in THRESHOLDS:
                pred = (s_valid >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()
                rows.append(
                    {
                        "model": model,
                        "score_column": score_column,
                        "evaluation_target": evaluation_target,
                        "split/test_set": split_name,
                        "threshold": threshold,
                        "n_samples": int(len(y_valid)),
                        "predicted_positive_rate": float(pred.mean()),
                        "precision": float(precision_score(y_valid, pred, zero_division=0)),
                        "recall": float(recall_score(y_valid, pred, zero_division=0)),
                        "f1": float(f1_score(y_valid, pred, zero_division=0)),
                        "true_positive_rate": true_positive_rate,
                        "false_positive_count": int(fp),
                        "false_negative_count": int(fn),
                    }
                )
    return pd.DataFrame(rows)


def plot_threshold_sweep(metrics: pd.DataFrame) -> None:
    plt = require_matplotlib()
    plot_data = metrics[metrics["evaluation_target"].eq("rugcheck_binary")].copy()
    if plot_data.empty:
        plot_data = metrics.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
    for model, group in plot_data.groupby("model"):
        group = group.sort_values("threshold")
        axes[0].plot(group["threshold"], group["precision"], marker="o", label=model)
        axes[1].plot(group["threshold"], group["recall"], marker="o", label=model)
    axes[0].set_title("Precision by Threshold")
    axes[1].set_title("Recall by Threshold")
    for ax in axes:
        ax.set_xlabel("Threshold")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Metric value")
    axes[1].legend(fontsize=8, frameon=True, loc="best")
    fig.suptitle("Threshold Sweep on Existing Model Scores")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    metrics = sweep_metrics(frame)
    if metrics.empty:
        raise RuntimeError("No valid score/target combinations found for threshold sweep.")
    metrics.to_csv(OUT_PATH, index=False)
    plot_threshold_sweep(metrics)
    log(f"Saved {OUT_PATH}")
    log(f"Saved {FIGURE_PATH}")
    log(f"Rows: {len(metrics):,}")


if __name__ == "__main__":
    main()
