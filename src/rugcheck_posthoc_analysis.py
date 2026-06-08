from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rugcheck_supervised_retraining_experiment import (  # noqa: E402
    build_splits,
    evaluate_binary,
    feature_sets,
    load_usable,
    make_xgboost_or_fallback,
    predict_scores,
)


RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"
DEFAULT_MASTER = RESULTS_DIR / "rugcheck_model_validation_master.csv"
DEFAULT_CALIBRATION = RESULTS_DIR / "rugcheck_retrained_threshold_calibration.csv"
DEFAULT_EVIDENCE_OUT = RESULTS_DIR / "rugcheck_evidence_tier_eval.csv"
DEFAULT_OPERATING_POINTS_OUT = RESULTS_DIR / "rugcheck_operating_points.csv"
DEFAULT_DISAGREEMENTS_OUT = RESULTS_DIR / "rugcheck_disagreement_cases.csv"

MODEL_SCORE_COLUMNS = [
    "silver_label_score",
    "token_logistic_score",
    "token_model_score",
    "graphsage_score",
    "combined_model_score",
]

DISAGREEMENT_COLUMNS = [
    "token_address",
    "year",
    "window_hours",
    "evidence_tier",
    "disagreement_group",
    "disagreement_rank",
    "disagreement_score",
    "weak_label",
    "rugcheck_label",
    "label_reason",
    "rugcheck_score",
    "rugcheck_score_normalised",
    "risk_names",
    "risk_levels",
    "activity_count",
    "total_volume",
    "imbalance_ratio",
    "sell_pressure",
    "lifespan_hours",
    "entity_concentration_ratio",
    "graph_degree",
    "connected_entities",
    *MODEL_SCORE_COLUMNS,
]


def log(message: str) -> None:
    print(f"[rugcheck-posthoc] {message}", flush=True)


def add_evidence_tier(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ["activity_count", "connected_entities", "total_volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    high = frame["activity_count"].ge(20) & frame["connected_entities"].ge(10) & frame["total_volume"].gt(0)
    medium = frame["activity_count"].ge(5) & frame["connected_entities"].ge(3) & frame["total_volume"].gt(0)
    frame["evidence_tier"] = np.select([high, medium], ["high", "medium"], default="low")
    return frame


def evaluate_evidence_tiers(frame: pd.DataFrame, min_class_per_year: int = 10) -> pd.DataFrame:
    usable = add_evidence_tier(frame)
    token_columns, _, combined_columns = feature_sets(usable)
    splits, _ = build_splits(usable, min_class_per_year)
    experiments = {
        "xgboost_token": token_columns,
        "xgboost_token_graphsage": combined_columns,
    }
    rows: list[dict[str, Any]] = []

    for split_name, (train, test) in splits.items():
        test = add_evidence_tier(test)
        for model_name, columns in experiments.items():
            if not columns:
                continue
            model, model_type, status = make_xgboost_or_fallback()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train[columns], train["rugcheck_label"].astype(int))
            scored = test.copy()
            scored["model_score"] = predict_scores(model, scored[columns])

            for tier in ["high", "medium", "low"]:
                subset = scored[scored["evidence_tier"].eq(tier)].copy()
                if subset.empty:
                    rows.append(
                        {
                            "split": split_name,
                            "model": model_name,
                            "model_type": model_type,
                            "status": status,
                            "evidence_tier": tier,
                            "n": 0,
                            "rugcheck_risky_count": 0,
                            "rugcheck_safe_count": 0,
                            "weak_risky_rate": np.nan,
                            "precision": np.nan,
                            "recall": np.nan,
                            "f1": np.nan,
                            "pr_auc": np.nan,
                        }
                    )
                    continue
                metrics = evaluate_binary(subset["rugcheck_label"], subset["model_score"])
                rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "model_type": model_type,
                        "status": status,
                        "evidence_tier": tier,
                        "n": int(len(subset)),
                        "rugcheck_risky_count": int(subset["rugcheck_label"].eq(1).sum()),
                        "rugcheck_safe_count": int(subset["rugcheck_label"].eq(0).sum()),
                        "weak_risky_rate": float(pd.to_numeric(subset["weak_label"], errors="coerce").fillna(0).mean()),
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "pr_auc": metrics["pr_auc"],
                        "tn": metrics["tn"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "tp": metrics["tp"],
                        "feature_count": len(columns),
                    }
                )
    return pd.DataFrame(rows)


def pick_operating_points(calibration: pd.DataFrame) -> pd.DataFrame:
    frame = calibration.copy()
    for column in ["precision", "recall", "f1", "threshold", "tn", "fp", "fn", "tp"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["false_positive_rate"] = np.divide(
        frame["fp"],
        frame["fp"] + frame["tn"],
        out=np.zeros(len(frame), dtype=float),
        where=(frame["fp"] + frame["tn"]).to_numpy() != 0,
    )

    rows: list[pd.Series] = []
    selectors = {
        "best_f1": lambda group: group.sort_values(["f1", "precision", "recall", "threshold"], ascending=[False, False, False, True]).head(1),
        "high_recall": lambda group: group[group["recall"].ge(0.90)].sort_values(["precision", "recall", "threshold"], ascending=[False, False, True]).head(1),
        "precision_alert": lambda group: group[group["precision"].ge(0.80)].sort_values(["recall", "precision", "threshold"], ascending=[False, False, True]).head(1),
        "low_false_positive": lambda group: group[group["recall"].ge(0.50)].sort_values(["false_positive_rate", "recall", "precision", "threshold"], ascending=[True, False, False, True]).head(1),
    }
    for (split, model), group in frame.groupby(["split", "model"], dropna=False):
        for point_name, selector in selectors.items():
            selected = selector(group)
            if selected.empty:
                unavailable = group.iloc[[0]].copy()
                unavailable.loc[:, "operating_point"] = point_name
                unavailable.loc[:, "available"] = False
                rows.append(unavailable.iloc[0])
                continue
            selected = selected.copy()
            selected.loc[:, "operating_point"] = point_name
            selected.loc[:, "available"] = True
            rows.append(selected.iloc[0])
    columns = [
        "split",
        "model",
        "operating_point",
        "available",
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
        "tn",
        "fp",
        "fn",
        "tp",
        "n",
        "safe_count",
        "risky_count",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["split", "model", "operating_point"])


def score_max_model(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    available = [column for column in MODEL_SCORE_COLUMNS if column in frame.columns]
    for column in available:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if available:
        frame["max_model_score"] = frame[available].max(axis=1, skipna=True)
        frame["min_model_score"] = frame[available].min(axis=1, skipna=True)
    else:
        frame["max_model_score"] = np.nan
        frame["min_model_score"] = np.nan
    return frame


def select_top(frame: pd.DataFrame, group_name: str, mask: pd.Series, sort_columns: list[str], ascending: list[bool], score_column: str) -> pd.DataFrame:
    subset = frame[mask].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(sort_columns, ascending=ascending).head(25).copy()
    subset["disagreement_group"] = group_name
    subset["disagreement_rank"] = range(1, len(subset) + 1)
    subset["disagreement_score"] = pd.to_numeric(subset[score_column], errors="coerce")
    return subset


def disagreement_cases(master_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(master_path, low_memory=False)
    frame["rugcheck_label"] = pd.to_numeric(frame["rugcheck_label"], errors="coerce")
    frame["weak_label"] = pd.to_numeric(frame["weak_label"], errors="coerce")
    for column in [
        "rugcheck_score",
        "rugcheck_score_normalised",
        "activity_count",
        "total_volume",
        "imbalance_ratio",
        "sell_pressure",
        "lifespan_hours",
        "entity_concentration_ratio",
        "graph_degree",
        "connected_entities",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = add_evidence_tier(score_max_model(frame))
    usable = frame[frame["rugcheck_label"].isin([0, 1])].copy()
    usable["rugcheck_risk_strength"] = (
        usable["rugcheck_score_normalised"].fillna(-1)
        + 0.001 * usable["rugcheck_score"].fillna(0)
    )
    usable["weak_strength"] = usable["silver_label_score"].fillna(-1)

    groups = [
        select_top(
            usable,
            "weak_safe_rugcheck_risky",
            usable["weak_label"].eq(0) & usable["rugcheck_label"].eq(1),
            ["rugcheck_risk_strength", "risk_count", "danger_count", "weak_strength"],
            [False, False, False, True],
            "rugcheck_risk_strength",
        ),
        select_top(
            usable,
            "weak_risky_rugcheck_safe",
            usable["weak_label"].eq(1) & usable["rugcheck_label"].eq(0),
            ["weak_strength", "rugcheck_risk_strength"],
            [False, True],
            "weak_strength",
        ),
        select_top(
            usable,
            "model_high_rugcheck_safe",
            usable["rugcheck_label"].eq(0) & usable["max_model_score"].notna(),
            ["max_model_score", "rugcheck_risk_strength"],
            [False, True],
            "max_model_score",
        ),
        select_top(
            usable,
            "model_low_rugcheck_risky",
            usable["rugcheck_label"].eq(1) & usable["min_model_score"].notna(),
            ["min_model_score", "rugcheck_risk_strength"],
            [True, False],
            "min_model_score",
        ),
    ]
    out = pd.concat([group for group in groups if not group.empty], ignore_index=True) if any(not group.empty for group in groups) else pd.DataFrame()
    for column in DISAGREEMENT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[DISAGREEMENT_COLUMNS]


def print_operating_interpretation(operating: pd.DataFrame) -> None:
    for (split, model), group in operating.groupby(["split", "model"], sort=True):
        available = group[group["available"].astype(bool)]
        if available.empty:
            print(f"{split} / {model}: no requested operating points available.")
            continue
        best = available[available["operating_point"].eq("best_f1")]
        high_recall = available[available["operating_point"].eq("high_recall")]
        precision_alert = available[available["operating_point"].eq("precision_alert")]
        low_fp = available[available["operating_point"].eq("low_false_positive")]
        parts = []
        if not best.empty:
            row = best.iloc[0]
            parts.append(f"best F1 {row.f1:.3f} at threshold {row.threshold:.2f}")
        if not high_recall.empty:
            row = high_recall.iloc[0]
            parts.append(f"high-recall option precision {row.precision:.3f}, recall {row.recall:.3f} at {row.threshold:.2f}")
        if not precision_alert.empty:
            row = precision_alert.iloc[0]
            parts.append(f"precision-alert option precision {row.precision:.3f}, recall {row.recall:.3f} at {row.threshold:.2f}")
        if not low_fp.empty:
            row = low_fp.iloc[0]
            parts.append(f"low-FP option FPR {row.false_positive_rate:.3f}, recall {row.recall:.3f} at {row.threshold:.2f}")
        print(f"{split} / {model}: " + "; ".join(parts) + ".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc RugCheck evidence-tier, operating-point, and disagreement analysis.")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--evidence-out", type=Path, default=DEFAULT_EVIDENCE_OUT)
    parser.add_argument("--operating-points-out", type=Path, default=DEFAULT_OPERATING_POINTS_OUT)
    parser.add_argument("--disagreements-out", type=Path, default=DEFAULT_DISAGREEMENTS_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.evidence_out, args.operating_points_out, args.disagreements_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    usable = load_usable(args.master)
    evidence_eval = evaluate_evidence_tiers(usable)
    evidence_eval.to_csv(args.evidence_out, index=False)
    log(f"saved evidence-tier eval: {args.evidence_out}")

    calibration = pd.read_csv(args.calibration)
    operating = pick_operating_points(calibration)
    operating.to_csv(args.operating_points_out, index=False)
    log(f"saved operating points: {args.operating_points_out}")

    disagreements = disagreement_cases(args.master)
    disagreements.to_csv(args.disagreements_out, index=False)
    log(f"saved disagreement cases: {args.disagreements_out}")

    print_operating_interpretation(operating)


if __name__ == "__main__":
    main()
