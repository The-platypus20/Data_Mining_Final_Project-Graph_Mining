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
SUMMARY_PATH = OUTPUT_DIR / "disagreement_summary.csv"
CASES_PATH = OUTPUT_DIR / "disagreement_cases.csv"
FIGURE_PATH = FIGURE_DIR / "disagreement_quadrants.png"

LABEL_VERSIONS = ["weak_strict", "weak_relaxed", "weak_3class"]
FEATURE_COLUMNS = [
    "lifespan_hours",
    "sell_pressure",
    "imbalance_ratio",
    "entity_concentration_ratio",
    "activity_count",
    "total_volume",
]
SCORE_COLUMNS = [
    "silver_label_score",
    "token_logistic_score",
    "token_model_score",
    "graphsage_score",
    "combined_model_score",
]


def log(message: str) -> None:
    print(f"[disagreement] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def load_analysis_frame() -> pd.DataFrame:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    labels = pd.read_csv(LABELS_PATH, low_memory=False)
    labels["token_address"] = labels["token_address"].astype(str)
    labels["year"] = pd.to_numeric(labels["year"], errors="coerce").astype("Int64")

    if MASTER_PATH.exists():
        header = pd.read_csv(MASTER_PATH, nrows=0).columns
        usecols = [
            column
            for column in ["token_address", "year", "window_hours", *FEATURE_COLUMNS, *SCORE_COLUMNS, "evidence_tier"]
            if column in header
        ]
        master = pd.read_csv(MASTER_PATH, usecols=usecols, low_memory=False)
        if "window_hours" in master.columns:
            master = master[master["window_hours"].eq(24)].copy()
        master["token_address"] = master["token_address"].astype(str)
        master["year"] = pd.to_numeric(master["year"], errors="coerce").astype("Int64")
        master = master.drop_duplicates(["year", "token_address"], keep="last")
        merge_cols = [column for column in master.columns if column not in labels.columns or column in ["token_address", "year"]]
        labels = labels.merge(master[merge_cols], on=["year", "token_address"], how="left")

    for column in [*LABEL_VERSIONS, "rugcheck_binary", *FEATURE_COLUMNS, *SCORE_COLUMNS]:
        if column in labels.columns:
            labels[column] = pd.to_numeric(labels[column], errors="coerce")
    return labels


def comparison_subset(frame: pd.DataFrame, label_version: str) -> pd.DataFrame:
    compared = frame[frame["rugcheck_binary"].isin([0, 1])].copy()
    if label_version == "weak_3class":
        compared = compared[compared[label_version].isin([-1, 1])].copy()
        compared["weak_binary_for_compare"] = compared[label_version].map({-1: 0, 1: 1})
    else:
        compared = compared[compared[label_version].isin([0, 1])].copy()
        compared["weak_binary_for_compare"] = compared[label_version]
    compared["rugcheck_binary"] = compared["rugcheck_binary"].astype(int)
    compared["weak_binary_for_compare"] = compared["weak_binary_for_compare"].astype(int)
    return compared


def group_name(weak_value: int, rugcheck_value: int) -> str:
    if weak_value == 1 and rugcheck_value == 1:
        return "weak_rug__rugcheck_risky"
    if weak_value == 0 and rugcheck_value == 0:
        return "weak_nonrug__rugcheck_safe"
    if weak_value == 1 and rugcheck_value == 0:
        return "weak_rug__rugcheck_safe"
    return "weak_nonrug__rugcheck_risky"


def make_cases_and_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_frames = []
    summary_rows = []
    median_columns = [column for column in [*FEATURE_COLUMNS, *SCORE_COLUMNS] if column in frame.columns]

    for label_version in LABEL_VERSIONS:
        compared = comparison_subset(frame, label_version)
        total = len(compared)
        if total == 0:
            summary_rows.append(
                {
                    "label_version": label_version,
                    "disagreement_group": "not_available",
                    "count": 0,
                    "share": np.nan,
                    "reason": "No comparable rows with non-missing RugCheck and confident/valid weak label.",
                }
            )
            continue

        compared["label_version"] = label_version
        compared["disagreement_group"] = [
            group_name(weak, rugcheck)
            for weak, rugcheck in zip(compared["weak_binary_for_compare"], compared["rugcheck_binary"])
        ]
        compared["is_disagreement"] = compared["weak_binary_for_compare"].ne(compared["rugcheck_binary"])
        keep_columns = [
            "label_version",
            "disagreement_group",
            "is_disagreement",
            "token_address",
            "year",
            label_version,
            "weak_binary_for_compare",
            "rugcheck_binary",
            *median_columns,
        ]
        case_frames.append(compared[[column for column in keep_columns if column in compared.columns]].copy())

        for group, group_frame in compared.groupby("disagreement_group"):
            row = {
                "label_version": label_version,
                "disagreement_group": group,
                "count": int(len(group_frame)),
                "share": float(len(group_frame) / total),
                "reason": "",
            }
            for column in median_columns:
                row[f"median_{column}"] = float(pd.to_numeric(group_frame[column], errors="coerce").median())
            summary_rows.append(row)

    cases = pd.concat(case_frames, ignore_index=True) if case_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    cases.to_csv(CASES_PATH, index=False)
    return summary, cases


def plot_quadrants(summary: pd.DataFrame) -> None:
    plt = require_matplotlib()
    plot_data = summary[summary["disagreement_group"].ne("not_available")].copy()
    order = [
        "weak_rug__rugcheck_risky",
        "weak_nonrug__rugcheck_safe",
        "weak_rug__rugcheck_safe",
        "weak_nonrug__rugcheck_risky",
    ]
    pivot = (
        plot_data.pivot_table(
            index="label_version",
            columns="disagreement_group",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=LABEL_VERSIONS, columns=order, fill_value=0)
    )
    colors = ["#7b2d26", "#2f7d59", "#d6a23f", "#8d96a8"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors)
    ax.set_ylabel("Compared tokens")
    ax.set_title("Weak Label vs RugCheck Agreement and Disagreement Quadrants")
    ax.legend(
        [
            "weak rug / RugCheck risky",
            "weak non-rug / RugCheck safe",
            "weak rug / RugCheck safe",
            "weak non-rug / RugCheck risky",
        ],
        frameon=True,
        fontsize=8,
    )
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_analysis_frame()
    summary, cases = make_cases_and_summary(frame)
    plot_quadrants(summary)
    log(f"Saved {SUMMARY_PATH}")
    log(f"Saved {CASES_PATH}")
    log(f"Saved {FIGURE_PATH}")
    log(f"Compared case rows: {len(cases):,}")


if __name__ == "__main__":
    main()
