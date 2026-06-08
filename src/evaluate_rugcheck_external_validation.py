from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"

DEFAULT_FEATURES = RESULTS_DIR / "early_window_silver_features.csv"
DEFAULT_RUGCHECK = RESULTS_DIR / "rugcheck_external_labels.csv"
DEFAULT_GRAPHSAGE = RESULTS_DIR / "graphsage_inductive_embeddings.csv"
DEFAULT_MASTER = RESULTS_DIR / "rugcheck_model_validation_master.csv"
DEFAULT_EXISTING_EVAL = RESULTS_DIR / "external_eval_existing_models.csv"
DEFAULT_CALIBRATION = RESULTS_DIR / "threshold_calibration_existing_models.csv"
DEFAULT_COVERAGE = RESULTS_DIR / "rugcheck_coverage_summary.csv"
DEFAULT_WEAK_CROSSTAB = RESULTS_DIR / "weak_vs_rugcheck_crosstab.csv"
DEFAULT_SPLIT_DIR = RESULTS_DIR / "rugcheck_benchmark_splits"
DEFAULT_SUMMARY = RESULTS_DIR / "rugcheck_external_validation_summary.json"

TOKEN_FEATURE_COLUMNS = [
    "activity_count",
    "buy_count",
    "sell_count",
    "total_volume",
    "buy_volume_usd",
    "sell_volume_usd",
    "imbalance_ratio",
    "unique_wallets",
    "lifespan_hours",
    "entity_concentration_ratio",
    "sell_pressure",
    "log_total_volume",
    "log_activity_count",
]

THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def make_xgboost_or_fallback() -> tuple[Pipeline, str, str]:
    try:
        from xgboost import XGBClassifier

        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=250,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=4,
                        ),
                    ),
                ]
            ),
            "xgboost",
            "completed",
        )
    except Exception as exc:
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", HistGradientBoostingClassifier(random_state=42, max_iter=160, learning_rate=0.06)),
                ]
            ),
            "hist_gradient_boosting_fallback",
            f"xgboost_unavailable: {exc}",
        )


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def predict_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    decision = model.decision_function(frame)
    return (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)


def evaluate_binary(y_true: pd.Series, scores: pd.Series | np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = y_true.astype(int).to_numpy()
    score_array = np.asarray(scores, dtype=float)
    pred = (score_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, score_array)) if len(np.unique(y)) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, score_array)) if len(np.unique(y)) == 2 else np.nan,
        "flagged_rate": float(pred.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score_old_weak_models(frame: pd.DataFrame, embedding_columns: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scored = frame.copy()
    model_rows: list[dict[str, Any]] = []

    token_columns = [column for column in TOKEN_FEATURE_COLUMNS if column in scored.columns]
    combined_columns = token_columns + embedding_columns
    experiments = [
        ("token_logistic_score", "token_only_logistic", make_logistic(), token_columns, "logistic_regression"),
        ("token_model_score", "token_only", None, token_columns, None),
        ("combined_model_score", "token_graphsage_combined", None, combined_columns, None),
    ]

    train_mask = scored["year"].eq(2024) & scored["weak_label"].isin([0, 1])
    if not train_mask.any():
        raise ValueError("No 2024 weak-label training rows found.")

    for score_column, experiment, model, columns, fixed_model_name in experiments:
        if not columns:
            scored[score_column] = np.nan
            model_rows.append({"experiment": experiment, "model": "not_run", "status": "no_features", "feature_count": 0})
            continue

        if model is None:
            model, model_name, status = make_xgboost_or_fallback()
        else:
            model_name, status = fixed_model_name, "completed"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(scored.loc[train_mask, columns], scored.loc[train_mask, "weak_label"].astype(int))
        scored[score_column] = predict_scores(model, scored[columns])
        model_rows.append({"experiment": experiment, "model": model_name, "status": status, "feature_count": len(columns)})

    return scored, model_rows


def build_master(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    features = pd.read_csv(args.features)
    features = features[features["window_hours"].eq(args.window_hours)].copy()
    features["token_address"] = features["token_address"].astype(str)

    rugcheck = pd.read_csv(args.rugcheck)
    rugcheck["token_address"] = rugcheck["token_address"].astype(str)
    rugcheck = rugcheck.drop_duplicates("token_address", keep="last")

    embeddings = pd.read_csv(args.graphsage)
    embeddings = embeddings[embeddings["window_hours"].eq(args.window_hours)].copy()
    embeddings["token_address"] = embeddings["token_address"].astype(str)
    embedding_columns = [column for column in embeddings.columns if column.startswith("graphsage_emb_")]
    embedding_keep = ["year", "window_hours", "token_address", "graphsage_score", *embedding_columns]

    master = features.merge(embeddings[embedding_keep], on=["year", "window_hours", "token_address"], how="left")
    for column in ["graphsage_score", *embedding_columns]:
        if column in master.columns:
            master[column] = pd.to_numeric(master[column], errors="coerce").fillna(0.0)

    master, model_rows = score_old_weak_models(master, embedding_columns)
    master = master.merge(rugcheck, on="token_address", how="left", suffixes=("", "_rugcheck"))

    master["rugcheck_label"] = pd.to_numeric(master["rugcheck_label"], errors="coerce").fillna(-1).astype(int)
    master["api_ok"] = master["api_ok"].fillna(False)
    master["evidence_tier"] = np.select(
        [
            (master["activity_count"] >= 10) & (master["connected_entities"] >= 5) & (master["total_volume"] > 0),
            (master["activity_count"] >= 3) & (master["total_volume"] > 0),
        ],
        ["high_evidence", "medium_evidence"],
        default="low_evidence",
    )

    ordered_columns = [
        "token_address",
        "year",
        "window_hours",
        "weak_label",
        "silver_label",
        "silver_label_score",
        "rugcheck_label",
        "api_status",
        "api_ok",
        "label_reason",
        "rugcheck_score",
        "rugcheck_score_normalised",
        "risk_count",
        "danger_count",
        "warn_count",
        "token_logistic_score",
        "token_model_score",
        "graphsage_score",
        "combined_model_score",
        "activity_count",
        "total_volume",
        "imbalance_ratio",
        "lifespan_hours",
        "connected_entities",
        "graph_degree",
        "entity_concentration_ratio",
        "sell_pressure",
        "unique_wallets",
        "evidence_tier",
        *embedding_columns,
    ]
    ordered_columns = [column for column in ordered_columns if column in master.columns]
    return master[ordered_columns], model_rows


def coverage_table(master: pd.DataFrame) -> pd.DataFrame:
    total = len(master)
    api_ok_count = int(master["api_ok"].fillna(False).astype(bool).sum())
    usable = master["rugcheck_label"].isin([0, 1])
    rows = [
        {"metric": "total_tokens", "value": total},
        {"metric": "api_ok_count", "value": api_ok_count},
        {"metric": "api_error_count", "value": total - api_ok_count},
        {"metric": "usable_label_count", "value": int(usable.sum())},
        {"metric": "rugcheck_risky_count", "value": int(master["rugcheck_label"].eq(1).sum())},
        {"metric": "rugcheck_safe_count", "value": int(master["rugcheck_label"].eq(0).sum())},
        {"metric": "unknown_count", "value": int(master["rugcheck_label"].eq(-1).sum())},
        {"metric": "usable_coverage_rate", "value": float(usable.mean()) if total else 0.0},
        {"metric": "unknown_rate", "value": float(master["rugcheck_label"].eq(-1).mean()) if total else 0.0},
    ]
    return pd.DataFrame(rows)


def evaluate_models(master: pd.DataFrame) -> pd.DataFrame:
    score_columns = {
        "rule_baseline": "silver_label_score",
        "token_logistic": "token_logistic_score",
        "token_only_model": "token_model_score",
        "graphsage_direct": "graphsage_score",
        "token_graphsage_combined": "combined_model_score",
    }
    subsets = build_subset_masks(master)
    rows = []
    for subset_name, mask in subsets.items():
        subset = master[mask & master["rugcheck_label"].isin([0, 1])].copy()
        for model_name, score_column in score_columns.items():
            if score_column not in subset.columns:
                continue
            scored = subset.dropna(subset=[score_column])
            if scored.empty:
                continue
            metrics = evaluate_binary(scored["rugcheck_label"], scored[score_column])
            metrics.update({"subset": subset_name, "model": model_name, "score_column": score_column, "threshold": 0.5})
            rows.append(metrics)
    return pd.DataFrame(rows)


def calibrate_thresholds(master: pd.DataFrame) -> pd.DataFrame:
    score_columns = {
        "rule_baseline": "silver_label_score",
        "token_logistic": "token_logistic_score",
        "token_only_model": "token_model_score",
        "graphsage_direct": "graphsage_score",
        "token_graphsage_combined": "combined_model_score",
    }
    usable = master[master["rugcheck_label"].isin([0, 1])].copy()
    rows = []
    for model_name, score_column in score_columns.items():
        if score_column not in usable.columns:
            continue
        scored = usable.dropna(subset=[score_column])
        for threshold in THRESHOLDS:
            metrics = evaluate_binary(scored["rugcheck_label"], scored[score_column], threshold=threshold)
            metrics.update({"model": model_name, "score_column": score_column, "threshold": threshold})
            rows.append(metrics)
    return pd.DataFrame(rows)


def build_subset_masks(master: pd.DataFrame) -> dict[str, pd.Series]:
    usable = master["rugcheck_label"].isin([0, 1])
    high_evidence = usable & master["evidence_tier"].eq("high_evidence")
    disagreement = usable & master["weak_label"].astype(float).ne(master["rugcheck_label"].astype(float))
    return {
        "all_usable_rugcheck": usable,
        "high_evidence_rugcheck": high_evidence,
        "disagreement_cases": disagreement,
    }


def write_benchmark_splits(master: pd.DataFrame, split_dir: Path) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    usable = master[master["rugcheck_label"].isin([0, 1])].copy()

    splits: dict[str, pd.DataFrame] = {
        "all_usable_rugcheck": usable,
        "high_evidence_rugcheck": usable[usable["evidence_tier"].eq("high_evidence")].copy(),
        "disagreement_cases": usable[usable["weak_label"].astype(float).ne(usable["rugcheck_label"].astype(float))].copy(),
    }

    positives = usable[usable["rugcheck_label"].eq(1)]
    negatives = usable[usable["rugcheck_label"].eq(0)]
    n_balanced = min(len(positives), len(negatives))
    if n_balanced:
        balanced = pd.concat(
            [
                positives.sample(n=n_balanced, random_state=42),
                negatives.sample(n=n_balanced, random_state=42),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=42)
    else:
        balanced = usable.iloc[0:0].copy()
    splits["balanced_rugcheck"] = balanced

    counts = {}
    for name, frame in splits.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)
        counts[name] = int(len(frame))
    return counts


def weak_crosstab(master: pd.DataFrame) -> pd.DataFrame:
    usable = master[master["rugcheck_label"].isin([0, 1])].copy()
    table = pd.crosstab(usable["weak_label"], usable["rugcheck_label"], margins=True)
    return table.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate old weak-label models against external RugCheck labels.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--rugcheck", type=Path, default=DEFAULT_RUGCHECK)
    parser.add_argument("--graphsage", type=Path, default=DEFAULT_GRAPHSAGE)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--master-out", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--existing-eval-out", type=Path, default=DEFAULT_EXISTING_EVAL)
    parser.add_argument("--calibration-out", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--coverage-out", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--weak-crosstab-out", type=Path, default=DEFAULT_WEAK_CROSSTAB)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [
        args.master_out,
        args.existing_eval_out,
        args.calibration_out,
        args.coverage_out,
        args.weak_crosstab_out,
        args.summary_out,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    master, model_rows = build_master(args)
    master.to_csv(args.master_out, index=False)

    coverage = coverage_table(master)
    coverage.to_csv(args.coverage_out, index=False)
    weak = weak_crosstab(master)
    weak.to_csv(args.weak_crosstab_out, index=False)
    existing_eval = evaluate_models(master)
    existing_eval.to_csv(args.existing_eval_out, index=False)
    calibration = calibrate_thresholds(master)
    calibration.to_csv(args.calibration_out, index=False)
    split_counts = write_benchmark_splits(master, args.split_dir)

    usable = master[master["rugcheck_label"].isin([0, 1])]
    safe_count = int(usable["rugcheck_label"].eq(0).sum())
    risky_count = int(usable["rugcheck_label"].eq(1).sum())
    retrain_decision = {
        "usable_label_count": int(len(usable)),
        "rugcheck_safe_count": safe_count,
        "rugcheck_risky_count": risky_count,
        "meets_minimum_retrain_rule": bool(len(usable) >= 500 and safe_count >= 100 and risky_count >= 100),
        "recommendation": "retrain_allowed" if len(usable) >= 500 and safe_count >= 100 and risky_count >= 100 else "external_evaluation_only",
    }

    summary = {
        "window_hours": args.window_hours,
        "model_score_generation": model_rows,
        "coverage": coverage.to_dict("records"),
        "benchmark_split_counts": split_counts,
        "retrain_decision": retrain_decision,
        "outputs": {
            "master": str(args.master_out),
            "existing_eval": str(args.existing_eval_out),
            "threshold_calibration": str(args.calibration_out),
            "coverage": str(args.coverage_out),
            "weak_vs_rugcheck": str(args.weak_crosstab_out),
            "split_dir": str(args.split_dir),
        },
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved master: {args.master_out}")
    print(f"Saved external eval: {args.existing_eval_out}")
    print(f"Saved threshold calibration: {args.calibration_out}")
    print(f"Retrain recommendation: {retrain_decision['recommendation']}")


if __name__ == "__main__":
    main()
