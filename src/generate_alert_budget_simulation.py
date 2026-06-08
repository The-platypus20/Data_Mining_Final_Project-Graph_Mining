from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"
MASTER_PATH = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025" / "rugcheck_model_validation_master.csv"

OUT_PATH = OUTPUT_DIR / "alert_budget_simulation.csv"
FIGURE_PATH = FIGURE_DIR / "alert_budget_simulation.png"

SCORE_COLUMNS = {
    "rule_baseline": "silver_label_score",
    "old_weak_token_logistic": "token_logistic_score",
    "old_weak_xgboost_token": "token_model_score",
    "graphsage_direct": "graphsage_score",
    "old_weak_xgboost_token_graphsage": "combined_model_score",
}

BUDGETS = [
    ("top_100", "absolute", 100),
    ("top_500", "absolute", 500),
    ("top_1000", "absolute", 1000),
    ("top_1pct", "fraction", 0.01),
    ("top_5pct", "fraction", 0.05),
    ("top_10pct", "fraction", 0.10),
]


def log(message: str) -> None:
    print(f"[alert-budget] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def load_scored_rugcheck() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing RugCheck master: {MASTER_PATH}")
    header = pd.read_csv(MASTER_PATH, nrows=0).columns
    usecols = [column for column in ["token_address", "year", "window_hours", "rugcheck_label", *SCORE_COLUMNS.values()] if column in header]
    frame = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
    if "window_hours" in frame.columns:
        frame = frame[frame["window_hours"].eq(24)].copy()
    frame = frame.drop_duplicates(["year", "token_address"], keep="last")
    frame["rugcheck_binary"] = pd.to_numeric(frame["rugcheck_label"], errors="coerce")
    frame = frame[frame["rugcheck_binary"].isin([0, 1])].copy()
    frame["rugcheck_binary"] = frame["rugcheck_binary"].astype(int)
    for column in SCORE_COLUMNS.values():
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def budget_size(n: int, budget_type: str, value: float | int) -> int:
    if budget_type == "absolute":
        return min(n, int(value))
    return max(1, int(np.ceil(n * float(value))))


def simulate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_risky = int(frame["rugcheck_binary"].sum())
    baseline = float(frame["rugcheck_binary"].mean()) if len(frame) else np.nan
    for model, score_column in SCORE_COLUMNS.items():
        if score_column not in frame.columns:
            continue
        scored = frame.dropna(subset=[score_column]).sort_values(score_column, ascending=False).copy()
        if scored.empty:
            continue
        for budget_label, budget_type, value in BUDGETS:
            k = budget_size(len(scored), budget_type, value)
            reviewed = scored.head(k)
            risky_found = int(reviewed["rugcheck_binary"].sum())
            precision = risky_found / k if k else np.nan
            rows.append(
                {
                    "model": model,
                    "score_column": score_column,
                    "target_label": "rugcheck_binary",
                    "budget": budget_label,
                    "reviewed_tokens": int(k),
                    "risky_found": risky_found,
                    "precision": float(precision),
                    "recall_capture_rate": risky_found / total_risky if total_risky else np.nan,
                    "enrichment_vs_baseline": precision / baseline if baseline else np.nan,
                    "baseline_positive_rate": baseline,
                    "total_risky_tokens": total_risky,
                    "n_scored_tokens": int(len(scored)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    return out


def plot_simulation(metrics: pd.DataFrame) -> None:
    plt = require_matplotlib()
    order = [label for label, _, _ in BUDGETS]
    plot = metrics[metrics["budget"].isin(order)].copy()
    plot["budget"] = pd.Categorical(plot["budget"], categories=order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True)
    for model, group in plot.groupby("model"):
        group = group.sort_values("budget")
        axes[0].plot(group["budget"].astype(str), group["precision"], marker="o", label=model)
        axes[1].plot(group["budget"].astype(str), group["recall_capture_rate"], marker="o", label=model)
    axes[0].set_title("Precision by Alert Budget")
    axes[1].set_title("Risky Tokens Captured by Alert Budget")
    axes[0].set_ylabel("Precision")
    axes[1].set_ylabel("Recall capture rate")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8, frameon=True, loc="best")
    fig.suptitle("Alert-Budget Simulation on RugCheck-Labeled Tokens")
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_scored_rugcheck()
    metrics = simulate(frame)
    if metrics.empty:
        raise RuntimeError("No valid model scores found for alert-budget simulation.")
    plot_simulation(metrics)
    log(f"Saved {OUT_PATH}")
    log(f"Saved {FIGURE_PATH}")
    log(f"Rows: {len(metrics):,}")


if __name__ == "__main__":
    main()
