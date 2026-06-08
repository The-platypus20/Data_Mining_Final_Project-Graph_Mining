from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DUNE_DIR = RESULTS_DIR / "dune_publishable_2024_2025"
CROSS_TIME_DIR = RESULTS_DIR / "cross_time"

HISTORICAL_FEATURES_PATH = CROSS_TIME_DIR / "aligned_historical_features.csv"
VALIDATION_PREDICTIONS_2024_PATH = CROSS_TIME_DIR / "validation_predictions_2024.csv"
DUNE_MASTER_PATH = DUNE_DIR / "rugcheck_model_validation_master.csv"

OUT_PATH = OUTPUT_DIR / "cross_source_shift_summary.csv"
FIGURE_PATH = FIGURE_DIR / "cross_source_shift_summary.png"

FEATURES = [
    "lifespan_hours",
    "activity_count",
    "total_volume",
    "imbalance_ratio",
    "entity_concentration_ratio",
]

DUNE_SCORE_COLUMNS = {
    "rule_baseline": "silver_label_score",
    "old_weak_token_logistic": "token_logistic_score",
    "old_weak_xgboost_token": "token_model_score",
    "graphsage_direct": "graphsage_score",
    "old_weak_xgboost_token_graphsage": "combined_model_score",
}


def log(message: str) -> None:
    print(f"[cross-source-shift] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def summarize_feature_frame(
    frame: pd.DataFrame,
    source: str,
    label_source: str,
    label_column: str,
    score_columns: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    rows = []
    score_columns = score_columns or {}
    for year, group in frame.groupby("year"):
        row: dict[str, object] = {
            "source": source,
            "year": int(year),
            "n_tokens": int(group["token_id"].nunique() if "token_id" in group.columns else len(group)),
            "label_source": label_source,
        }
        labels = pd.to_numeric(group[label_column], errors="coerce") if label_column in group.columns else pd.Series(np.nan, index=group.index)
        known_labels = labels[labels.isin([0, 1])]
        row["positive_rate"] = float(known_labels.mean()) if len(known_labels) else np.nan
        row["label_coverage_rate"] = float(len(known_labels) / len(group)) if len(group) else np.nan
        for feature in FEATURES:
            if feature in group.columns:
                row[f"median_{feature}"] = float(pd.to_numeric(group[feature], errors="coerce").median())
            else:
                row[f"median_{feature}"] = np.nan
        for model, column in score_columns.items():
            if column in group.columns:
                scores = pd.to_numeric(group[column], errors="coerce")
                row[f"predicted_positive_rate_{model}"] = float((scores >= 0.5).mean()) if scores.notna().any() else np.nan
        rows.append(row)
    return rows


def load_solrpds_rows() -> list[dict[str, object]]:
    if not HISTORICAL_FEATURES_PATH.exists():
        return []
    hist = pd.read_csv(HISTORICAL_FEATURES_PATH, low_memory=False)
    hist = hist.rename(columns={"mint": "token_id"})
    hist["source"] = "SolRPDS"
    rows = summarize_feature_frame(hist, "SolRPDS", "SolRPDS inactivity/rug_label", "rug_label")

    if VALIDATION_PREDICTIONS_2024_PATH.exists():
        pred = pd.read_csv(VALIDATION_PREDICTIONS_2024_PATH, low_memory=False)
        if {"year", "model", "predicted_rug_label"}.issubset(pred.columns):
            pred_rates = (
                pred.groupby(["year", "model"])["predicted_rug_label"]
                .mean()
                .reset_index()
            )
            for row in rows:
                if row["year"] != 2024:
                    continue
                for pred_row in pred_rates[pred_rates["year"].eq(2024)].itertuples(index=False):
                    row[f"predicted_positive_rate_solrpds_{pred_row.model}"] = float(pred_row.predicted_rug_label)
    return rows


def load_dune_rows() -> list[dict[str, object]]:
    if not DUNE_MASTER_PATH.exists():
        return []
    header = pd.read_csv(DUNE_MASTER_PATH, nrows=0).columns
    usecols = [
        column
        for column in [
            "token_address",
            "year",
            "window_hours",
            "rugcheck_label",
            *FEATURES,
            *DUNE_SCORE_COLUMNS.values(),
        ]
        if column in header
    ]
    dune = pd.read_csv(DUNE_MASTER_PATH, usecols=usecols, low_memory=False)
    if "window_hours" in dune.columns:
        dune = dune[dune["window_hours"].eq(24)].copy()
    dune = dune.rename(columns={"token_address": "token_id"})
    dune["rugcheck_binary"] = pd.to_numeric(dune["rugcheck_label"], errors="coerce").where(
        pd.to_numeric(dune["rugcheck_label"], errors="coerce").isin([0, 1])
    )
    return summarize_feature_frame(
        dune,
        "Dune",
        "RugCheck binary external labels",
        "rugcheck_binary",
        DUNE_SCORE_COLUMNS,
    )


def build_summary() -> pd.DataFrame:
    rows = [*load_solrpds_rows(), *load_dune_rows()]
    summary = pd.DataFrame(rows)
    source_order = {"SolRPDS": 0, "Dune": 1}
    summary["_source_order"] = summary["source"].map(source_order).fillna(9)
    summary = summary.sort_values(["_source_order", "year"]).drop(columns=["_source_order"])
    summary.to_csv(OUT_PATH, index=False)
    return summary


def plot_summary(summary: pd.DataFrame) -> None:
    plt = require_matplotlib()
    plot = summary.copy()
    plot["source_year"] = plot["source"] + " " + plot["year"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.flatten()

    specs = [
        ("positive_rate", "Positive label rate", "Class-Prior Shift"),
        ("median_lifespan_hours", "Median lifespan hours", "Lifespan Shift"),
        ("median_total_volume", "Median total volume", "Volume Shift"),
        ("median_imbalance_ratio", "Median imbalance ratio", "Imbalance Shift"),
    ]
    colors = np.where(plot["source"].eq("SolRPDS"), "#4c78a8", "#b8403a")
    for ax, (column, ylabel, title) in zip(axes, specs):
        values = pd.to_numeric(plot[column], errors="coerce")
        ax.bar(plot["source_year"], values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        if column != "positive_rate":
            ax.set_yscale("symlog", linthresh=1)
    fig.suptitle("Cross-Source Shift: SolRPDS vs Dune/RugCheck", y=0.995)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    if summary.empty:
        raise RuntimeError("No SolRPDS or Dune rows available for cross-source shift summary.")
    plot_summary(summary)
    log(f"Saved {OUT_PATH}")
    log(f"Saved {FIGURE_PATH}")
    log(f"Rows: {len(summary):,}")


if __name__ == "__main__":
    main()
