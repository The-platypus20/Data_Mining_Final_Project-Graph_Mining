from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"

MASTER_PATH = RESULTS_DIR / "rugcheck_model_validation_master.csv"
COVERAGE_PATH = RESULTS_DIR / "rugcheck_coverage_summary.csv"
EXTERNAL_SUMMARY_PATH = RESULTS_DIR / "rugcheck_external_validation_summary.json"
EXTERNAL_EVAL_PATH = RESULTS_DIR / "external_eval_existing_models.csv"
EXISTING_CALIBRATION_PATH = RESULTS_DIR / "threshold_calibration_existing_models.csv"
RETRAINED_RESULTS_PATH = RESULTS_DIR / "rugcheck_retrained_results.csv"
RETRAINED_DISTRIBUTION_PATH = RESULTS_DIR / "rugcheck_train_test_distribution.csv"
XGBOOST_ABLATION_PATH = RESULTS_DIR / "xgboost_window_ablation.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "rugcheck_feature_importance.csv"

LEAKAGE_COLUMNS = [
    "rugcheck_score",
    "rugcheck_score_normalised",
    "risk_count",
    "danger_count",
    "warn_count",
    "risk_names",
    "risk_levels",
    "risk_scores",
    "label_reason",
    "api_status",
    "api_ok",
    "raw_json_path",
]

SCORE_COLUMNS = {
    "rule_baseline": "silver_label_score",
    "old_weak_token_logistic": "token_logistic_score",
    "old_weak_xgboost_token": "token_model_score",
    "graphsage_direct": "graphsage_score",
    "old_weak_xgboost_token_graphsage": "combined_model_score",
}

EXPECTED_OUTPUTS = [
    "final_run_summary.md",
    "token_labels_all_versions.csv",
    "label_sensitivity_summary.csv",
    "weak_label_rugcheck_confusion.csv",
    "temporal_model_metrics.csv",
    "threshold_calibration_metrics.csv",
    "calibration_bins.csv",
    "topk_ranking_metrics.csv",
    "graphsage_ablation.csv",
]

EXPECTED_FIGURES = [
    "label_sensitivity.png",
    "weak_label_vs_rugcheck_confusion.png",
    "behavior_separation.png",
    "temporal_shift_2024_2025.png",
    "predicted_positive_rate_by_model.png",
    "topk_enrichment.png",
    "calibration_curve.png",
    "feature_importance.png",
    "token_distribution_by_source_year.png",
    "lifespan_cumulative_curve.png",
]


def log(message: str) -> None:
    print(f"[final-core] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def read_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing RugCheck master file: {MASTER_PATH}")
    frame = pd.read_csv(MASTER_PATH, low_memory=False)
    if "window_hours" in frame.columns:
        frame = frame[frame["window_hours"].eq(24)].copy()
    frame["token_address"] = frame["token_address"].astype(str)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["rugcheck_label"] = pd.to_numeric(frame["rugcheck_label"], errors="coerce")
    for column in ["weak_label", "silver_label"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in SCORE_COLUMNS.values():
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "api_ok" in frame.columns:
        frame["api_ok"] = frame["api_ok"].astype(str).str.lower().isin(["true", "1"])
    else:
        frame["api_ok"] = frame["rugcheck_label"].isin([0, 1])
    return frame.drop_duplicates(["year", "token_address"], keep="last")


def valid_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.isin([0, 1]))


def make_token_labels(master: pd.DataFrame) -> pd.DataFrame:
    labels = master[
        [
            column
            for column in [
                "token_address",
                "year",
                "weak_label",
                "silver_label",
                "silver_label_score",
                "rugcheck_label",
                "api_ok",
                "lifespan_hours",
                "sell_pressure",
                "entity_concentration_ratio",
                "total_volume",
                "activity_count",
            ]
            if column in master.columns
        ]
    ].copy()
    weak_base = valid_binary(labels.get("weak_label", labels.get("silver_label", pd.Series(np.nan, index=labels.index))))
    score = pd.to_numeric(labels.get("silver_label_score"), errors="coerce")
    labels["weak_strict"] = weak_base
    labels["weak_relaxed"] = np.where(score.notna(), (score >= 0.30).astype(int), weak_base)
    labels["weak_3class"] = np.select(
        [labels["weak_strict"].eq(1), score.le(0.20)],
        [1, -1],
        default=0,
    )
    labels.loc[score.isna() & weak_base.isna(), "weak_3class"] = np.nan
    labels["rugcheck_binary"] = valid_binary(labels["rugcheck_label"])

    weak_known = labels["weak_strict"].isin([0, 1])
    rug_known = labels["rugcheck_binary"].isin([0, 1])
    labels["label_union"] = np.nan
    labels.loc[(labels["weak_strict"].eq(1)) | (labels["rugcheck_binary"].eq(1)), "label_union"] = 1
    labels.loc[weak_known & rug_known & labels["weak_strict"].eq(0) & labels["rugcheck_binary"].eq(0), "label_union"] = 0

    labels["label_intersection"] = np.nan
    labels.loc[weak_known & rug_known, "label_intersection"] = (
        labels.loc[weak_known & rug_known, "weak_strict"].eq(1)
        & labels.loc[weak_known & rug_known, "rugcheck_binary"].eq(1)
    ).astype(int)

    labels["label_consensus"] = np.nan
    agree = weak_known & rug_known & labels["weak_strict"].eq(labels["rugcheck_binary"])
    labels.loc[agree, "label_consensus"] = labels.loc[agree, "weak_strict"]
    labels.to_csv(OUTPUT_DIR / "token_labels_all_versions.csv", index=False)
    return labels


def label_sensitivity(labels: pd.DataFrame) -> pd.DataFrame:
    versions = ["weak_strict", "weak_relaxed", "weak_3class", "rugcheck_binary", "label_union", "label_intersection", "label_consensus"]
    rows = []
    for version in versions:
        if version not in labels.columns:
            continue
        for year, frame in labels.groupby("year", dropna=False):
            values = pd.to_numeric(frame[version], errors="coerce")
            total = int(len(frame))
            uncertain = int(values.eq(0).sum()) if version == "weak_3class" else int(values.isna().sum())
            positive = int(values.eq(1).sum())
            coverage = int(frame["rugcheck_binary"].isin([0, 1]).sum())
            rows.append(
                {
                    "label_version": version,
                    "year": int(year) if not pd.isna(year) else np.nan,
                    "total_tokens": total,
                    "positive_count": positive,
                    "positive_rate": positive / total if total else np.nan,
                    "uncertain_count": uncertain,
                    "uncertain_rate": uncertain / total if total else np.nan,
                    "rugcheck_coverage_count": coverage,
                    "rugcheck_coverage_rate": coverage / total if total else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "label_sensitivity_summary.csv", index=False)
    return out


def compare_to_rugcheck(labels: pd.DataFrame) -> pd.DataFrame:
    versions = ["weak_strict", "weak_relaxed", "label_union", "label_intersection", "label_consensus"]
    rows = []
    for version in versions:
        values = pd.to_numeric(labels[version], errors="coerce")
        target = pd.to_numeric(labels["rugcheck_binary"], errors="coerce")
        mask = values.isin([0, 1]) & target.isin([0, 1])
        if not mask.any():
            continue
        y_true = target[mask].astype(int)
        y_pred = values[mask].astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "label_version": version,
                "n_compared": int(mask.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "agreement_rate": float((y_true.to_numpy() == y_pred.to_numpy()).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "weak_label_rugcheck_confusion.csv", index=False)
    return out


def score_metrics(y_true: pd.Series, score: pd.Series, threshold: float = 0.5) -> dict[str, Any]:
    mask = y_true.isin([0, 1]) & score.notna()
    y = y_true[mask].astype(int)
    s = score[mask].astype(float).clip(0, 1)
    pred = (s >= threshold).astype(int)
    if len(y) == 0:
        return {}
    return {
        "n_samples": int(len(y)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, s)) if y.nunique() == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, s)) if y.nunique() == 2 else np.nan,
        "predicted_positive_rate": float(pred.mean()),
        "true_positive_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, s)) if y.nunique() == 2 else np.nan,
        "ece": float(expected_calibration_error(y, s)) if y.nunique() == 2 else np.nan,
    }


def topk_metrics(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target = valid_binary(master["rugcheck_label"])
    for split_name, split_frame in [("all_usable_rugcheck", master), ("dune_2025_rugcheck", master[master["year"].eq(2025)])]:
        split_target = valid_binary(split_frame["rugcheck_label"])
        usable = split_target.isin([0, 1])
        if not usable.any():
            continue
        baseline = float(split_target[usable].mean())
        for model, column in SCORE_COLUMNS.items():
            if column not in split_frame.columns:
                continue
            scored = split_frame.loc[usable, [column]].copy()
            scored["target"] = split_target[usable].astype(int)
            scored[column] = pd.to_numeric(scored[column], errors="coerce")
            scored = scored.dropna(subset=[column]).sort_values(column, ascending=False)
            if scored.empty:
                continue
            row = {
                "model": model,
                "target_label": "rugcheck_binary",
                "split/test_set": split_name,
                "baseline_positive_rate": baseline,
                "predicted_positive_rate": float((scored[column] >= 0.5).mean()),
                "auprc": float(average_precision_score(scored["target"], scored[column])) if scored["target"].nunique() == 2 else np.nan,
                "roc_auc": float(roc_auc_score(scored["target"], scored[column])) if scored["target"].nunique() == 2 else np.nan,
            }
            for pct in [0.01, 0.05, 0.10]:
                k = max(1, int(math.ceil(len(scored) * pct)))
                precision_at_k = float(scored.head(k)["target"].mean())
                key = f"{int(pct * 100)}pct"
                row[f"precision_at_{key}"] = precision_at_k
                row[f"enrichment_at_{key}"] = precision_at_k / baseline if baseline > 0 else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "topk_ranking_metrics.csv", index=False)
    return out


def expected_calibration_error(y_true: pd.Series, scores: pd.Series, bins: int = 10) -> float:
    y = y_true.to_numpy(dtype=float)
    s = scores.to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (s >= low) & (s <= high)
        else:
            mask = (s >= low) & (s < high)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return ece


def calibration_outputs(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    bin_rows = []
    for split_name, split_frame in [("all_usable_rugcheck", master), ("dune_2025_rugcheck", master[master["year"].eq(2025)])]:
        target = valid_binary(split_frame["rugcheck_label"])
        for model, column in SCORE_COLUMNS.items():
            if column not in split_frame.columns:
                continue
            score = pd.to_numeric(split_frame[column], errors="coerce").clip(0, 1)
            mask = target.isin([0, 1]) & score.notna()
            if not mask.any():
                continue
            y = target[mask].astype(int)
            s = score[mask].astype(float)
            metric_rows.append(
                {
                    "model": model,
                    "target_label": "rugcheck_binary",
                    "split/test_set": split_name,
                    "brier_score": float(brier_score_loss(y, s)),
                    "ece": float(expected_calibration_error(y, s)),
                    "predicted_positive_rate_at_0_5": float((s >= 0.5).mean()),
                    "true_positive_rate": float(y.mean()),
                    "n_samples": int(len(y)),
                }
            )
            bins = pd.cut(s, bins=np.linspace(0, 1, 11), include_lowest=True)
            grouped = pd.DataFrame({"score": s, "target": y, "bin": bins}).groupby("bin", observed=False)
            for interval, group in grouped:
                if group.empty:
                    continue
                bin_rows.append(
                    {
                        "model": model,
                        "target_label": "rugcheck_binary",
                        "split/test_set": split_name,
                        "bin": str(interval),
                        "n_samples": int(len(group)),
                        "mean_predicted_probability": float(group["score"].mean()),
                        "observed_positive_rate": float(group["target"].mean()),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    bins = pd.DataFrame(bin_rows)
    metrics.to_csv(OUTPUT_DIR / "threshold_calibration_metrics.csv", index=False)
    bins.to_csv(OUTPUT_DIR / "calibration_bins.csv", index=False)
    return metrics, bins


def temporal_metrics(master: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_frame = master[master["year"].eq(2025)].copy()
    target = valid_binary(split_frame["rugcheck_label"])
    for model, column in SCORE_COLUMNS.items():
        if column not in split_frame.columns:
            continue
        metrics = score_metrics(target, pd.to_numeric(split_frame[column], errors="coerce"))
        if not metrics:
            continue
        metrics.update(
            {
                "train_source/year": "Dune 2024 weak labels",
                "test_source/year": "Dune 2025 RugCheck labels",
                "label_source": "RugCheck external validation target; model scores trained from weak Dune labels",
                "model": model,
                "n_train": int(master[master["year"].eq(2024)]["weak_label"].isin([0, 1]).sum()) if "weak_label" in master.columns else np.nan,
                "n_test": metrics.pop("n_samples"),
                "xgboost_status": "existing score; see rugcheck_external_validation_summary.json",
            }
        )
        rows.append(metrics)

    if RETRAINED_DISTRIBUTION_PATH.exists():
        dist = pd.read_csv(RETRAINED_DISTRIBUTION_PATH)
        skipped = dist[dist["role"].eq("not_run")]
        for _, row in skipped.iterrows():
            rows.append(
                {
                    "train_source/year": "Dune 2024 RugCheck labels",
                    "test_source/year": "Dune 2025 RugCheck labels",
                    "label_source": "RugCheck supervised retraining",
                    "model": "rugcheck_supervised_temporal",
                    "n_train": np.nan,
                    "n_test": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "predicted_positive_rate": np.nan,
                    "true_positive_rate": np.nan,
                    "brier_score": np.nan,
                    "ece": np.nan,
                    "xgboost_status": f"skipped: {row.get('status', 'not_run')}",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "temporal_model_metrics.csv", index=False)
    return out


def copy_graphsage_ablation() -> pd.DataFrame:
    if XGBOOST_ABLATION_PATH.exists():
        frame = pd.read_csv(XGBOOST_ABLATION_PATH)
        frame["source_artifact"] = str(XGBOOST_ABLATION_PATH)
        frame.to_csv(OUTPUT_DIR / "graphsage_ablation.csv", index=False)
        return frame
    out = pd.DataFrame([{"status": "skipped", "reason": "xgboost_window_ablation.csv not found"}])
    out.to_csv(OUTPUT_DIR / "graphsage_ablation.csv", index=False)
    return out


def make_figures(label_summary: pd.DataFrame, confusion: pd.DataFrame, topk: pd.DataFrame, calibration_bins: pd.DataFrame, temporal: pd.DataFrame, master: pd.DataFrame) -> list[Path]:
    plt = require_matplotlib()
    created: list[Path] = []

    pivot = label_summary[label_summary["label_version"].isin(["weak_strict", "weak_relaxed", "rugcheck_binary"])].pivot(index="year", columns="label_version", values="positive_rate")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Positive rate")
    ax.set_title("Label Sensitivity by Year")
    fig.tight_layout()
    out = FIGURE_DIR / "label_sensitivity.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_conf = confusion.set_index("label_version")[["tp", "fp", "tn", "fn"]]
    plot_conf.plot(kind="bar", stacked=True, ax=ax, color=["#b8403a", "#d6a23f", "#2f7d59", "#8d96a8"])
    ax.set_ylabel("Compared tokens")
    ax.set_title("Weak Label vs RugCheck Confusion")
    fig.tight_layout()
    out = FIGURE_DIR / "weak_label_vs_rugcheck_confusion.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    behavior = master[master["rugcheck_label"].isin([0, 1])].copy()
    behavior["label"] = np.where(behavior["rugcheck_label"].eq(1), "RugCheck risky", "RugCheck safe")
    data = [np.log1p(pd.to_numeric(behavior.loc[behavior["label"].eq(label), "lifespan_hours"], errors="coerce").fillna(0).clip(lower=0)) for label in ["RugCheck safe", "RugCheck risky"]]
    ax.boxplot(data, labels=["RugCheck safe", "RugCheck risky"], showfliers=False)
    ax.set_ylabel("log1p(lifespan_hours)")
    ax.set_title("Behavior Separation by RugCheck Label")
    fig.tight_layout()
    out = FIGURE_DIR / "behavior_separation.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    shift = master.groupby("year").agg(tokens=("token_address", "nunique"), rugcheck_risky_rate=("rugcheck_label", lambda x: pd.to_numeric(x, errors="coerce").eq(1).mean()), median_volume=("total_volume", "median")).reset_index()
    ax2 = ax.twinx()
    ax.bar(shift["year"].astype(str), shift["tokens"], color="#8d96a8", alpha=0.65, label="tokens")
    ax2.plot(shift["year"].astype(str), shift["rugcheck_risky_rate"], color="#b8403a", marker="o", label="RugCheck risky rate")
    ax.set_ylabel("Unique tokens")
    ax2.set_ylabel("RugCheck risky rate")
    ax.set_title("Temporal Shift: Dune 2024 vs 2025")
    fig.tight_layout()
    out = FIGURE_DIR / "temporal_shift_2024_2025.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    temporal_plot = temporal[temporal["predicted_positive_rate"].notna()].copy()
    ax.bar(temporal_plot["model"], temporal_plot["predicted_positive_rate"], color="#4c78a8")
    ax.axhline(float(temporal_plot["true_positive_rate"].dropna().iloc[0]), color="#b8403a", linestyle="--", label="true positive rate")
    ax.set_ylabel("Rate")
    ax.set_title("Predicted Positive Rate by Model")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / "predicted_positive_rate_by_model.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(out)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    topk_plot = topk[topk["split/test_set"].eq("dune_2025_rugcheck")].copy()
    ax.bar(topk_plot["model"], topk_plot["enrichment_at_1pct"], color="#7b2d26")
    ax.set_ylabel("Enrichment at top 1%")
    ax.set_title("Top-k Enrichment")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = FIGURE_DIR / "topk_enrichment.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    created.append(out)

    if not calibration_bins.empty:
        fig, ax = plt.subplots(figsize=(6.5, 6))
        for model, group in calibration_bins[calibration_bins["split/test_set"].eq("dune_2025_rugcheck")].groupby("model"):
            ax.plot(group["mean_predicted_probability"], group["observed_positive_rate"], marker="o", label=model)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive rate")
        ax.set_title("Calibration Curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = FIGURE_DIR / "calibration_curve.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        created.append(out)

    if FEATURE_IMPORTANCE_PATH.exists():
        imp = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        if {"feature", "importance"}.issubset(imp.columns):
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            plot_imp = imp.sort_values("importance", ascending=False).head(12)
            ax.barh(plot_imp["feature"][::-1], plot_imp["importance"][::-1], color="#2f7d59")
            ax.set_title("Feature Importance")
            fig.tight_layout()
            out = FIGURE_DIR / "feature_importance.png"
            fig.savefig(out, dpi=220)
            plt.close(fig)
            created.append(out)
    return created


def audit_artifacts() -> pd.DataFrame:
    rows = []
    for name in EXPECTED_OUTPUTS:
        path = OUTPUT_DIR / name
        row: dict[str, Any] = {"artifact": str(path.relative_to(PROJECT_ROOT)), "exists": path.exists(), "report_ready": False}
        if path.exists() and path.suffix.lower() == ".csv":
            try:
                sample = pd.read_csv(path, nrows=5)
                row["row_count"] = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
                row["main_columns"] = ", ".join(sample.columns[:12])
                row["report_ready"] = row["row_count"] > 0
            except Exception as exc:
                row["reason"] = str(exc)
        elif path.exists() and path.suffix.lower() == ".md":
            row["row_count"] = np.nan
            row["main_columns"] = ""
            row["report_ready"] = path.stat().st_size > 0
        else:
            row["reason"] = "missing"
        rows.append(row)
    for name in EXPECTED_FIGURES:
        path = FIGURE_DIR / name
        rows.append(
            {
                "artifact": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.exists(),
                "row_count": np.nan,
                "main_columns": "",
                "report_ready": path.exists() and path.stat().st_size > 1000,
                "reason": "" if path.exists() else "missing",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUTPUT_DIR / "final_artifact_audit.csv", index=False)
    lines = ["# Final Artifact Audit", ""]
    for _, row in audit.iterrows():
        status = "exists" if row["exists"] else "missing"
        ready = "report-ready" if row["report_ready"] else "not report-ready"
        detail = f"; rows={row['row_count']}" if pd.notna(row.get("row_count")) else ""
        columns = f"; columns={row['main_columns']}" if row.get("main_columns") else ""
        reason = f"; reason={row['reason']}" if row.get("reason") else ""
        lines.append(f"- `{row['artifact']}`: {status}, {ready}{detail}{columns}{reason}")
    (OUTPUT_DIR / "final_artifact_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit


def source_summary(master: pd.DataFrame, labels: pd.DataFrame, models_run: list[str], skipped: list[str], figures: list[Path]) -> None:
    usable = master["rugcheck_label"].isin([0, 1])
    year_counts = {int(year): int(count) for year, count in master.groupby("year")["token_address"].nunique().items()}
    coverage = {}
    if COVERAGE_PATH.exists():
        coverage = pd.read_csv(COVERAGE_PATH).set_index("metric")["value"].to_dict()
    old_usable_note = (
        "The older 6,277-label count is stale. It came from an earlier RugCheck cache/build before the full batch "
        "was rebuilt. The latest source-of-truth master has 165,259 usable RugCheck labels. The 98-row retraining "
        "artifact is also stale/mismatched relative to the latest master, so it is not used as the source of truth."
    )
    lines = [
        "# Final Run Summary",
        "",
        "## RugCheck Source Of Truth",
        f"- Latest RugCheck source file: `{MASTER_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Total rows after 24h window de-duplication: {len(master):,}",
        f"- Unique token/year rows: {master[['year', 'token_address']].drop_duplicates().shape[0]:,}",
        f"- Usable RugCheck labels: {int(usable.sum()):,}",
        f"- RugCheck risky: {int(master['rugcheck_label'].eq(1).sum()):,}",
        f"- RugCheck safe: {int(master['rugcheck_label'].eq(0).sum()):,}",
        f"- No usable RugCheck/API error: {int((~usable).sum()):,}",
        f"- Year coverage: {year_counts}",
        f"- Clarification: {old_usable_note}",
        "",
        "## Input Files Used",
        f"- `{MASTER_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{COVERAGE_PATH.relative_to(PROJECT_ROOT)}`" if COVERAGE_PATH.exists() else "- RugCheck coverage summary missing",
        f"- `{EXTERNAL_SUMMARY_PATH.relative_to(PROJECT_ROOT)}`" if EXTERNAL_SUMMARY_PATH.exists() else "- External validation summary missing",
        f"- `{XGBOOST_ABLATION_PATH.relative_to(PROJECT_ROOT)}`" if XGBOOST_ABLATION_PATH.exists() else "- GraphSAGE ablation source missing",
        "",
        "## Label Counts",
    ]
    for column in ["weak_strict", "weak_relaxed", "rugcheck_binary", "label_union", "label_intersection", "label_consensus"]:
        values = pd.to_numeric(labels[column], errors="coerce")
        lines.append(f"- {column}: positives={int(values.eq(1).sum()):,}, negatives={int(values.eq(0).sum()):,}, missing/uncertain={int(values.isna().sum()):,}")
    lines.extend(
        [
            "",
            "## Train/Test Split Used",
            "- Main final validation metrics use Dune 2025 rows with RugCheck labels as the test set and existing Dune weak-label model score columns.",
            "- RugCheck-supervised temporal retraining artifact is marked skipped/stale because the current retraining distribution says temporal split `not_run` and only 98 rows.",
            "",
            "## Leakage Columns Excluded For RugCheck Target",
            "- " + ", ".join(LEAKAGE_COLUMNS),
            "",
            "## Models Actually Run Or Reused",
        ]
    )
    lines.extend(f"- {model}" for model in models_run)
    lines.extend(["", "## Skipped Items And Reasons"])
    lines.extend(f"- {item}" for item in skipped)
    lines.extend(["", "## Final Output Files"])
    for path in sorted(OUTPUT_DIR.glob("*")):
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    lines.extend(["", "## Final Figures"])
    for path in sorted(FIGURE_DIR.glob("*")):
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    (OUTPUT_DIR / "final_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    master = read_master()
    labels = make_token_labels(master)
    sensitivity = label_sensitivity(labels)
    confusion = compare_to_rugcheck(labels)
    topk = topk_metrics(master)
    calibration, calibration_bins = calibration_outputs(master)
    temporal = temporal_metrics(master, calibration)
    copy_graphsage_ablation()
    figures = make_figures(sensitivity, confusion, topk, calibration_bins, temporal, master)

    models_run = [
        "No new model training in this final-output run.",
        "Reused existing score column: rule_baseline / silver_label_score.",
        "Reused existing score column: old_weak_token_logistic / token_logistic_score.",
        "Reused existing score column: old_weak_xgboost_token / token_model_score.",
        "Reused existing score column: graphsage_direct / graphsage_score.",
        "Reused existing score column: old_weak_xgboost_token_graphsage / combined_model_score.",
    ]
    if EXTERNAL_SUMMARY_PATH.exists():
        summary = json.loads(EXTERNAL_SUMMARY_PATH.read_text(encoding="utf-8"))
        for row in summary.get("model_score_generation", []):
            models_run.append(f"Existing artifact reports score generation completed: {row}")
    skipped = [
        "No RugCheck crawl was run; latest 165,259-label master was reused.",
        "RugCheck-supervised temporal retraining was not rerun; existing retraining artifact has only 98 rows and temporal split not_run.",
        "graph_feature_ablation.csv not created; no separate final temporal graph feature pipeline was run in this pass. Existing GraphSAGE ablation was copied to graphsage_ablation.csv.",
    ]
    audit = audit_artifacts()
    source_summary(master, labels, models_run, skipped, figures)
    log(f"RugCheck usable labels: {int(master['rugcheck_label'].isin([0, 1]).sum()):,}")
    log(f"Outputs under {OUTPUT_DIR}: {len(list(OUTPUT_DIR.glob('*')))} files")
    log(f"Figures under {FIGURE_DIR}: {len(list(FIGURE_DIR.glob('*')))} files")
    log(f"Audit rows: {len(audit)}")


if __name__ == "__main__":
    main()
