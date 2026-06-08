from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"

LABELS_PATH = OUTPUT_DIR / "token_labels_all_versions.csv"
MASTER_PATH = RESULTS_DIR / "rugcheck_model_validation_master.csv"
SUMMARY_PATH = OUTPUT_DIR / "feature_sanity_by_label.csv"
FIGURE_PATH = FIGURE_DIR / "feature_sanity_by_label.png"
FINAL_SUMMARY_PATH = OUTPUT_DIR / "final_run_summary.md"

KEY_FEATURES = [
    "lifespan_hours",
    "activity_count",
    "total_volume",
    "sell_pressure",
    "imbalance_ratio",
    "entity_concentration_ratio",
    "unique_wallets",
    "graph_degree",
    "connected_entities",
]


def log(message: str) -> None:
    print(f"[feature-sanity] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def load_frame() -> pd.DataFrame:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    frame = pd.read_csv(LABELS_PATH, low_memory=False)
    frame["token_address"] = frame["token_address"].astype(str)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")

    missing_features = [column for column in KEY_FEATURES if column not in frame.columns]
    if missing_features and MASTER_PATH.exists():
        header = pd.read_csv(MASTER_PATH, nrows=0).columns
        usecols = [
            column
            for column in ["token_address", "year", "window_hours", *missing_features]
            if column in header
        ]
        if len(usecols) > 2:
            master = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
            if "window_hours" in master.columns:
                master = master[master["window_hours"].eq(24)].copy()
            master["token_address"] = master["token_address"].astype(str)
            master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("Int64")
            master = master.drop_duplicates(["year", "token_address"], keep="last")
            merge_cols = [
                column
                for column in master.columns
                if column in ["token_address", "year"] or column not in frame.columns
            ]
            frame = frame.merge(master[merge_cols], on=["year", "token_address"], how="left")

    for column in [*KEY_FEATURES, "weak_strict", "weak_relaxed", "rugcheck_binary", "label_consensus"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def iqr(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def summarize_group(frame: pd.DataFrame, comparison: str, label_name: str, mask: pd.Series, features: list[str]) -> list[dict[str, object]]:
    rows = []
    group = frame[mask].copy()
    for feature in features:
        values = pd.to_numeric(group[feature], errors="coerce")
        rows.append(
            {
                "comparison": comparison,
                "label_group": label_name,
                "feature": feature,
                "n_tokens": int(values.notna().sum()),
                "median": float(values.median()) if values.notna().any() else np.nan,
                "iqr": iqr(values),
                "q1": float(values.quantile(0.25)) if values.notna().any() else np.nan,
                "q3": float(values.quantile(0.75)) if values.notna().any() else np.nan,
            }
        )
    return rows


def build_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available_features = [column for column in KEY_FEATURES if column in frame.columns]
    skipped_features = [column for column in KEY_FEATURES if column not in frame.columns]
    rows: list[dict[str, object]] = []

    comparisons = [
        ("weak_strict", "weak_strict", "positive", "negative"),
        ("weak_relaxed", "weak_relaxed", "positive", "negative"),
        ("rugcheck_binary", "RugCheck", "risky", "safe"),
        ("label_consensus", "weak/RugCheck consensus", "positive", "negative"),
    ]
    for column, comparison, positive_label, negative_label in comparisons:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.extend(summarize_group(frame, comparison, positive_label, values.eq(1), available_features))
        rows.extend(summarize_group(frame, comparison, negative_label, values.eq(0), available_features))

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary, skipped_features


def plot_summary(summary: pd.DataFrame) -> None:
    plt = require_matplotlib()
    plot_features = [
        feature
        for feature in ["lifespan_hours", "activity_count", "total_volume", "sell_pressure", "imbalance_ratio", "entity_concentration_ratio"]
        if feature in set(summary["feature"])
    ]
    plot_data = summary[
        summary["feature"].isin(plot_features)
        & summary["comparison"].isin(["weak_strict", "weak_relaxed", "RugCheck", "weak/RugCheck consensus"])
    ].copy()
    plot_data["series"] = plot_data["comparison"] + ": " + plot_data["label_group"]
    n_features = len(plot_features)
    if n_features == 0:
        return
    fig, axes = plt.subplots(n_features, 1, figsize=(11, max(4, 2.8 * n_features)), sharex=False)
    if n_features == 1:
        axes = [axes]
    for ax, feature in zip(axes, plot_features):
        subset = plot_data[plot_data["feature"].eq(feature)].copy()
        y = np.arange(len(subset))
        ax.barh(y, subset["median"], color="#4c78a8", alpha=0.85)
        ax.errorbar(
            subset["median"],
            y,
            xerr=[subset["median"] - subset["q1"], subset["q3"] - subset["median"]],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            linewidth=1,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(subset["series"], fontsize=8)
        ax.set_title(feature)
        if feature in {"lifespan_hours", "activity_count", "total_volume", "imbalance_ratio"}:
            ax.set_xscale("symlog", linthresh=1)
    fig.suptitle("Feature Sanity by Label Group: Median with IQR", y=0.995)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def append_summary_note(skipped_features: list[str]) -> None:
    if not FINAL_SUMMARY_PATH.exists():
        return
    note = [
        "",
        "## Feature Sanity Analysis",
        f"- Output: `outputs/final/{SUMMARY_PATH.name}`",
        f"- Figure: `figures/final/{FIGURE_PATH.name}`",
    ]
    if skipped_features:
        note.append(f"- Skipped missing feature columns: {', '.join(skipped_features)}")
    else:
        note.append("- Skipped missing feature columns: none")
    text = FINAL_SUMMARY_PATH.read_text(encoding="utf-8")
    marker = "## Feature Sanity Analysis"
    if marker in text:
        text = text.split(marker)[0].rstrip()
    FINAL_SUMMARY_PATH.write_text(text + "\n" + "\n".join(note) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    summary, skipped_features = build_summary(frame)
    if summary.empty:
        raise RuntimeError("No feature sanity rows could be generated.")
    plot_summary(summary)
    append_summary_note(skipped_features)
    log(f"Saved {SUMMARY_PATH}")
    log(f"Saved {FIGURE_PATH}")
    log(f"Skipped feature columns: {', '.join(skipped_features) if skipped_features else 'none'}")


if __name__ == "__main__":
    main()
