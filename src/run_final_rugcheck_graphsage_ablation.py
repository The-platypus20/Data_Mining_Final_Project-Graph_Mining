from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import dune_publishable_experiments_pipeline as publishable  # noqa: E402


FINAL_DIR = PROJECT_ROOT / "outputs" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"
TMP_DIR = FINAL_DIR / "_graphsage_final_tmp"

EVENTS_PATH = PROJECT_ROOT / "data" / "silver" / "dune_token_events_2024_2025.parquet"
WINDOWS_PATH = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025" / "early_window_silver_features.csv"
LABELS_PATH = FINAL_DIR / "token_labels_all_versions.csv"

METRICS_PATH = FINAL_DIR / "graphsage_final_metrics.csv"
TOPK_PATH = FINAL_DIR / "graphsage_final_topk.csv"
SUMMARY_PATH = FINAL_DIR / "graphsage_final_run_summary.md"
ABLATION_PATH = FINAL_DIR / "graphsage_ablation.csv"
FIGURE_PATH = FIGURE_DIR / "graphsage_final_ablation.png"

USABLE_RUGCHECK_LABELS = 165_259
WINDOW_HOURS = 1
TARGET_LABEL = "rugcheck_binary"
SPLIT = "temporal_2024_train_2025_test"
THRESHOLD = 0.5


def require_inputs() -> None:
    missing = [path for path in [EVENTS_PATH, WINDOWS_PATH, LABELS_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(path) for path in missing))


def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    events = pd.read_parquet(EVENTS_PATH)
    if "trader_id" not in events.columns and "wallet_address" in events.columns:
        events = events.rename(columns={"wallet_address": "trader_id"})
    required_event_cols = {"year", "token_address", "trader_id", "amount_usd", "tx_id", "side", "block_time"}
    missing_event_cols = sorted(required_event_cols.difference(events.columns))
    if missing_event_cols:
        raise ValueError(f"Events file is missing GraphSAGE columns: {missing_event_cols}")

    windows = pd.read_csv(WINDOWS_PATH)
    windows = windows[windows["window_hours"].eq(WINDOW_HOURS)].copy()
    if windows.empty:
        raise ValueError(f"No existing early-window GraphSAGE rows for window_hours={WINDOW_HOURS}")

    labels = pd.read_csv(LABELS_PATH)
    required_label_cols = {"year", "token_address", TARGET_LABEL, "api_ok"}
    missing_label_cols = sorted(required_label_cols.difference(labels.columns))
    if missing_label_cols:
        raise ValueError(f"Final label table is missing columns: {missing_label_cols}")

    labels[TARGET_LABEL] = pd.to_numeric(labels[TARGET_LABEL], errors="coerce")
    usable = labels[labels["api_ok"].eq(True) & labels[TARGET_LABEL].isin([0, 1])].copy()
    usable_count = int(len(usable))
    if usable_count != USABLE_RUGCHECK_LABELS:
        raise ValueError(f"Expected {USABLE_RUGCHECK_LABELS:,} usable RugCheck labels, found {usable_count:,}")

    label_view = usable[["year", "token_address", TARGET_LABEL]].drop_duplicates(["year", "token_address"])
    windows = windows.drop(columns=["silver_label", "weak_label"], errors="ignore").merge(
        label_view,
        on=["year", "token_address"],
        how="left",
    )
    windows["silver_label"] = windows[TARGET_LABEL]
    windows["weak_label"] = windows[TARGET_LABEL]

    n_train = int(windows[windows["year"].eq(2024)]["silver_label"].notna().sum())
    n_test = int(windows[windows["year"].eq(2025)]["silver_label"].notna().sum())
    if n_train < 20 or n_test < 20:
        raise ValueError(f"Insufficient RugCheck temporal labels after merge: n_train={n_train}, n_test={n_test}")

    class_counts = {
        "usable_rugcheck_labels": usable_count,
        "n_train": n_train,
        "n_test": n_test,
        "train_positive": int(windows[windows["year"].eq(2024)]["silver_label"].sum()),
        "test_positive": int(windows[windows["year"].eq(2025)]["silver_label"].sum()),
    }
    return events, windows, class_counts


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        if end == 1.0:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob >= start) & (y_prob < end)
        if not mask.any():
            continue
        ece += float(mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean()))
    return ece


def precision_at_k_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(-y_prob)
    base_rate = float(y_true.mean()) if len(y_true) else np.nan
    rows = []
    for pct in [1, 5, 10]:
        k = max(1, int(np.ceil(len(y_true) * pct / 100.0)))
        selected = y_true[order[:k]]
        precision = float(selected.mean()) if k else np.nan
        rows.append(
            {
                "target_label": TARGET_LABEL,
                "split": SPLIT,
                "window_hours": WINDOW_HOURS,
                "top_k_percent": pct,
                "k": int(k),
                "positives_in_top_k": int(selected.sum()),
                "base_positive_rate": base_rate,
                "precision_at_k": precision,
                "enrichment_at_k": float(precision / base_rate) if base_rate > 0 else np.nan,
            }
        )
    return rows


def compute_outputs(embeddings: pd.DataFrame, class_counts: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = embeddings[
        embeddings["year"].eq(2025)
        & embeddings["window_hours"].eq(WINDOW_HOURS)
        & embeddings["silver_label"].notna()
    ].copy()
    if test.empty:
        raise ValueError("GraphSAGE run produced no labeled 2025 test scores")

    y_true = pd.to_numeric(test["silver_label"], errors="coerce").astype(int).to_numpy()
    y_prob = pd.to_numeric(test["graphsage_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    topk_rows = precision_at_k_rows(y_true, y_prob)
    topk = pd.DataFrame(topk_rows)
    topk_lookup = {int(row["top_k_percent"]): row for row in topk_rows}

    metric = {
        "n_train": class_counts["n_train"],
        "n_test": class_counts["n_test"],
        "train_positive": class_counts["train_positive"],
        "test_positive": class_counts["test_positive"],
        "target_label": TARGET_LABEL,
        "split": SPLIT,
        "window_hours": WINDOW_HOURS,
        "threshold": THRESHOLD,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": ece_score(y_true, y_prob),
        "precision_at_1pct": topk_lookup[1]["precision_at_k"],
        "precision_at_5pct": topk_lookup[5]["precision_at_k"],
        "precision_at_10pct": topk_lookup[10]["precision_at_k"],
        "enrichment_at_1pct": topk_lookup[1]["enrichment_at_k"],
        "enrichment_at_5pct": topk_lookup[5]["enrichment_at_k"],
        "enrichment_at_10pct": topk_lookup[10]["enrichment_at_k"],
        "model": "graphsage",
        "status": "completed",
    }
    return pd.DataFrame([metric]), topk


def run_existing_graphsage(events: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    publishable.WINDOW_HOURS = [WINDOW_HOURS]
    publishable.GRAPHSAGE_EMBEDDINGS_PATH = TMP_DIR / "graphsage_inductive_embeddings.csv"
    publishable.GRAPHSAGE_METRICS_PATH = TMP_DIR / "graphsage_inductive_metrics.csv"
    publishable.GRAPHSAGE_TRAINING_LOG_PATH = TMP_DIR / "graphsage_training_log.csv"
    embeddings, existing_metrics, status = publishable.try_run_graphsage(events, windows)
    if status != "completed" or embeddings is None or embeddings.empty:
        raise RuntimeError(f"Existing GraphSAGE routine did not complete: {status}")
    return embeddings, existing_metrics


def write_summary(metrics: pd.DataFrame, topk: pd.DataFrame, existing_metrics: pd.DataFrame, class_counts: dict[str, int]) -> Path:
    metric = metrics.iloc[0].to_dict()
    summary_path = TMP_DIR / "graphsage_final_run_summary.md"
    lines = [
        "# RugCheck-Aware GraphSAGE Final Ablation",
        "",
        "Status: completed",
        "",
        "Run date: 2026-06-08",
        "",
        "## Source of Truth",
        "",
        f"- RugCheck usable labels: {class_counts['usable_rugcheck_labels']:,}",
        f"- Labels/features: `{LABELS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Existing GraphSAGE routine: `src/dune_publishable_experiments_pipeline.py::try_run_graphsage`",
        f"- Existing events: `{EVENTS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Existing early-window inputs: `{WINDOWS_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Run Configuration",
        "",
        f"- Target label: `{TARGET_LABEL}`",
        f"- Split: {SPLIT}",
        f"- Window: {WINDOW_HOURS}h, the smallest existing GraphSAGE early-window evaluation",
        "- Threshold for precision/recall/F1: 0.5",
        "- RugCheck-derived fields were not used as node features.",
        "",
        "## Metrics",
        "",
        f"- n_train: {int(metric['n_train']):,}",
        f"- n_test: {int(metric['n_test']):,}",
        f"- ROC-AUC: {metric['roc_auc']:.6f}",
        f"- PR-AUC: {metric['pr_auc']:.6f}",
        f"- Precision: {metric['precision']:.6f}",
        f"- Recall: {metric['recall']:.6f}",
        f"- F1: {metric['f1']:.6f}",
        f"- Predicted positive rate: {metric['predicted_positive_rate']:.6f}",
        f"- Brier score: {metric['brier_score']:.6f}",
        f"- ECE: {metric['ece']:.6f}",
        "",
        "## Top-K",
        "",
        topk.to_markdown(index=False),
        "",
        "## Existing Routine Metrics",
        "",
        existing_metrics.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{TOPK_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{SUMMARY_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{ABLATION_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{FIGURE_PATH.relative_to(PROJECT_ROOT)}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def update_ablation(metrics: pd.DataFrame) -> Path:
    row = metrics.copy()
    row["experiment"] = "rugcheck_graphsage_final_1h"
    row["source_artifact"] = str(METRICS_PATH)
    if ABLATION_PATH.exists():
        existing = pd.read_csv(ABLATION_PATH)
        existing = existing[existing.get("experiment", pd.Series(dtype=str)).ne("rugcheck_graphsage_final_1h")]
        updated = pd.concat([existing, row], ignore_index=True, sort=False)
    else:
        updated = row
    path = TMP_DIR / "graphsage_ablation.csv"
    updated.to_csv(path, index=False)
    return path


def write_figure(metrics: pd.DataFrame, topk: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    metric = metrics.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["ROC-AUC", "PR-AUC", "F1"], [metric["roc_auc"], metric["pr_auc"], metric["f1"]], color=["#2f6f8f", "#d08c3f", "#5b8c5a"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("GraphSAGE RugCheck Metrics")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([f"{int(v)}%" for v in topk["top_k_percent"]], topk["precision_at_k"], color="#7a4e9d")
    axes[1].axhline(float(topk["base_positive_rate"].iloc[0]), color="#333333", linestyle="--", linewidth=1, label="Base rate")
    axes[1].set_ylim(0, max(1.0, float(topk["precision_at_k"].max()) * 1.1))
    axes[1].set_title("Precision at Top K")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Final RugCheck-Aware GraphSAGE Ablation, 2024 -> 2025")
    fig.tight_layout()
    path = TMP_DIR / "graphsage_final_ablation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def publish_outputs(paths: dict[str, Path]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["metrics"], METRICS_PATH)
    shutil.copy2(paths["topk"], TOPK_PATH)
    shutil.copy2(paths["summary"], SUMMARY_PATH)
    shutil.copy2(paths["ablation"], ABLATION_PATH)
    shutil.copy2(paths["figure"], FIGURE_PATH)


def main() -> None:
    require_inputs()
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    events, windows, class_counts = prepare_inputs()
    embeddings, existing_metrics = run_existing_graphsage(events, windows)
    metrics, topk = compute_outputs(embeddings, class_counts)

    metrics_path = TMP_DIR / "graphsage_final_metrics.csv"
    topk_path = TMP_DIR / "graphsage_final_topk.csv"
    metrics.to_csv(metrics_path, index=False)
    topk.to_csv(topk_path, index=False)
    summary_path = write_summary(metrics, topk, existing_metrics, class_counts)
    ablation_path = update_ablation(metrics)
    figure_path = write_figure(metrics, topk)

    publish_outputs(
        {
            "metrics": metrics_path,
            "topk": topk_path,
            "summary": summary_path,
            "ablation": ablation_path,
            "figure": figure_path,
        }
    )

    print(json.dumps({"status": "completed", "metrics": metrics.iloc[0].to_dict(), "topk": topk.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
