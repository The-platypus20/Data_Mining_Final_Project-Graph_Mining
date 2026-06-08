from __future__ import annotations

import argparse
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"

DEFAULT_INPUT = RESULTS_DIR / "rugcheck_model_validation_master.csv"
DEFAULT_RESULTS_OUT = RESULTS_DIR / "rugcheck_retrained_results.csv"
DEFAULT_CALIBRATION_OUT = RESULTS_DIR / "rugcheck_retrained_threshold_calibration.csv"
DEFAULT_IMPORTANCE_OUT = RESULTS_DIR / "rugcheck_feature_importance.csv"
DEFAULT_DISTRIBUTION_OUT = RESULTS_DIR / "rugcheck_train_test_distribution.csv"

RUGCHECK_DERIVED_COLUMNS = {
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
}

NON_FEATURE_COLUMNS = {
    "token_address",
    "year",
    "window_hours",
    "rugcheck_label",
    "weak_label",
    "silver_label",
    "silver_label_score",
    "token_logistic_score",
    "token_model_score",
    "combined_model_score",
    "evidence_tier",
}

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
    "graph_degree",
    "connected_entities",
    "entity_concentration_ratio",
    "active_days",
    "buy_sell_count_ratio",
    "sell_pressure",
    "volume_per_wallet",
    "activity_per_wallet",
    "log_total_volume",
    "log_activity_count",
]

BASELINE_SCORE_COLUMNS = {
    "rule_baseline": "silver_label_score",
    "old_weak_token_logistic": "token_logistic_score",
    "old_weak_xgboost_token": "token_model_score",
    "old_weak_xgboost_token_graphsage": "combined_model_score",
}

THRESHOLDS = [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]


def log(message: str) -> None:
    print(f"[rugcheck-retrain] {message}", flush=True)


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


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
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.04,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            min_child_weight=2,
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
                    ("model", HistGradientBoostingClassifier(random_state=42, max_iter=180, learning_rate=0.05)),
                ]
            ),
            "hist_gradient_boosting_fallback",
            f"xgboost_unavailable: {exc}",
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
        "safe_count": int((y == 0).sum()),
        "risky_count": int((y == 1).sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, score_array)) if len(np.unique(y)) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, score_array)) if len(np.unique(y)) == 2 else np.nan,
        "flagged_rate": float(pred.mean()) if len(pred) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_usable(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, low_memory=False)
    frame["rugcheck_label"] = pd.to_numeric(frame["rugcheck_label"], errors="coerce")
    usable = frame[frame["rugcheck_label"].isin([0, 1])].copy()
    usable["rugcheck_label"] = usable["rugcheck_label"].astype(int)
    return usable


def feature_sets(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    forbidden = RUGCHECK_DERIVED_COLUMNS | NON_FEATURE_COLUMNS
    numeric_columns = [
        column
        for column in frame.columns
        if column not in forbidden and pd.api.types.is_numeric_dtype(frame[column])
    ]
    token_columns = [column for column in TOKEN_FEATURE_COLUMNS if column in numeric_columns]
    graphsage_columns = [
        column
        for column in numeric_columns
        if column == "graphsage_score" or column.startswith("graphsage_emb_")
    ]
    combined_columns = token_columns + [column for column in graphsage_columns if column not in token_columns]
    return token_columns, graphsage_columns, combined_columns


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "n": int(len(frame)),
        "safe_count": int(frame["rugcheck_label"].eq(0).sum()),
        "risky_count": int(frame["rugcheck_label"].eq(1).sum()),
    }


def split_distribution(split_name: str, train: pd.DataFrame, test: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for role, frame in [("train", train), ("test", test)]:
        counts = class_counts(frame)
        counts.update(
            {
                "split": split_name,
                "role": role,
                "year_min": int(frame["year"].min()) if len(frame) else np.nan,
                "year_max": int(frame["year"].max()) if len(frame) else np.nan,
            }
        )
        rows.append(counts)
    return rows


def build_splits(frame: pd.DataFrame, min_class_per_year: int) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], list[dict[str, Any]]]:
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    distribution_rows: list[dict[str, Any]] = []

    train_2024 = frame[frame["year"].eq(2024)].copy()
    test_2025 = frame[frame["year"].eq(2025)].copy()
    temporal_ok = all(
        value >= min_class_per_year
        for value in [
            train_2024["rugcheck_label"].eq(0).sum(),
            train_2024["rugcheck_label"].eq(1).sum(),
            test_2025["rugcheck_label"].eq(0).sum(),
            test_2025["rugcheck_label"].eq(1).sum(),
        ]
    )
    if temporal_ok:
        splits["temporal_2024_train_2025_test"] = (train_2024, test_2025)
        distribution_rows.extend(split_distribution("temporal_2024_train_2025_test", train_2024, test_2025))
    else:
        reason = {
            "split": "temporal_2024_train_2025_test",
            "role": "not_run",
            "n": int(len(train_2024) + len(test_2025)),
            "safe_count": int(frame[frame["year"].isin([2024, 2025])]["rugcheck_label"].eq(0).sum()),
            "risky_count": int(frame[frame["year"].isin([2024, 2025])]["rugcheck_label"].eq(1).sum()),
            "year_min": 2024,
            "year_max": 2025,
            "status": f"too_few_class_examples_min_per_year_{min_class_per_year}",
        }
        distribution_rows.append(reason)

    train_idx, test_idx = train_test_split(
        frame.index,
        test_size=0.3,
        random_state=42,
        stratify=frame["rugcheck_label"],
    )
    strat_train = frame.loc[train_idx].copy()
    strat_test = frame.loc[test_idx].copy()
    splits["stratified_70_30"] = (strat_train, strat_test)
    distribution_rows.extend(split_distribution("stratified_70_30", strat_train, strat_test))
    return splits, distribution_rows


def model_importance(model: Pipeline, model_label: str, split: str, feature_columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        importance_type = "gain_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        importance_type = "abs_coefficient"
    else:
        values = np.full(len(feature_columns), np.nan)
        importance_type = "not_available"
    return pd.DataFrame(
        {
            "split": split,
            "model": model_label,
            "importance_type": importance_type,
            "feature": feature_columns,
            "importance": values,
        }
    ).sort_values(["split", "model", "importance"], ascending=[True, True, False])


def train_and_score(
    split_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame]]:
    result_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []

    experiments = [
        ("logistic_regression_token", make_logistic(), "logistic_regression", "completed", feature_groups["token"]),
        ("xgboost_token", None, None, None, feature_groups["token"]),
        ("xgboost_token_graphsage", None, None, None, feature_groups["token_graphsage"]),
    ]

    for model_label, model, model_type, status, columns in experiments:
        if not columns:
            result_rows.append(
                {
                    "split": split_name,
                    "model": model_label,
                    "model_type": "not_run",
                    "status": "no_features",
                    "feature_count": 0,
                }
            )
            continue

        if model is None:
            model, model_type, status = make_xgboost_or_fallback()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[columns], train["rugcheck_label"].astype(int))
        scores = predict_scores(model, test[columns])
        metrics = evaluate_binary(test["rugcheck_label"], scores)
        metrics.update(
            {
                "split": split_name,
                "model": model_label,
                "model_type": model_type,
                "status": status,
                "score_column": "predicted_rugcheck_risky_probability",
                "threshold": 0.5,
                "feature_count": len(columns),
            }
        )
        result_rows.append(metrics)
        importance_frames.append(model_importance(model, model_label, split_name, columns))

        for threshold in THRESHOLDS:
            threshold_metrics = evaluate_binary(test["rugcheck_label"], scores, threshold=threshold)
            threshold_metrics.update(
                {
                    "split": split_name,
                    "model": model_label,
                    "score_column": "predicted_rugcheck_risky_probability",
                    "threshold": threshold,
                }
            )
            calibration_rows.append(threshold_metrics)

    return result_rows, calibration_rows, importance_frames


def evaluate_baselines(split_name: str, test: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for model_label, score_column in BASELINE_SCORE_COLUMNS.items():
        if score_column not in test.columns:
            continue
        scored = test.dropna(subset=[score_column]).copy()
        if scored.empty:
            continue
        metrics = evaluate_binary(scored["rugcheck_label"], scored[score_column])
        metrics.update(
            {
                "split": split_name,
                "model": model_label,
                "model_type": "existing_score",
                "status": "completed",
                "score_column": score_column,
                "threshold": 0.5,
                "feature_count": np.nan,
            }
        )
        result_rows.append(metrics)
        for threshold in THRESHOLDS:
            threshold_metrics = evaluate_binary(scored["rugcheck_label"], scored[score_column], threshold=threshold)
            threshold_metrics.update(
                {
                    "split": split_name,
                    "model": model_label,
                    "score_column": score_column,
                    "threshold": threshold,
                }
            )
            calibration_rows.append(threshold_metrics)
    return result_rows, calibration_rows


def graphsage_improvement_message(results: pd.DataFrame) -> str:
    scored = results[results["model"].isin(["xgboost_token", "xgboost_token_graphsage"])].copy()
    if scored.empty or not scored["model"].eq("xgboost_token_graphsage").any():
        return "GraphSAGE comparison not available."

    split_order = ["temporal_2024_train_2025_test", "stratified_70_30"]
    for split in split_order:
        split_rows = scored[scored["split"].eq(split)]
        token = split_rows[split_rows["model"].eq("xgboost_token")]
        graph = split_rows[split_rows["model"].eq("xgboost_token_graphsage")]
        if token.empty or graph.empty:
            continue
        metric = "pr_auc" if token["pr_auc"].notna().any() and graph["pr_auc"].notna().any() else "f1"
        token_value = float(token.iloc[0][metric])
        graph_value = float(graph.iloc[0][metric])
        delta = graph_value - token_value
        verb = "improved" if delta > 0 else "did not improve"
        return (
            f"GraphSAGE features {verb} over token-only on {split}: "
            f"{metric} {graph_value:.4f} vs {token_value:.4f} (delta {delta:+.4f})."
        )
    return "GraphSAGE comparison not available."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain RugCheck-supervised token risk models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-out", type=Path, default=DEFAULT_RESULTS_OUT)
    parser.add_argument("--calibration-out", type=Path, default=DEFAULT_CALIBRATION_OUT)
    parser.add_argument("--importance-out", type=Path, default=DEFAULT_IMPORTANCE_OUT)
    parser.add_argument("--distribution-out", type=Path, default=DEFAULT_DISTRIBUTION_OUT)
    parser.add_argument("--min-class-per-year", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.results_out, args.calibration_out, args.importance_out, args.distribution_out]:
        path.parent.mkdir(parents=True, exist_ok=True)

    usable = load_usable(args.input)
    token_columns, graphsage_columns, combined_columns = feature_sets(usable)
    log(
        "usable RugCheck labels: "
        f"{len(usable):,} total, {usable['rugcheck_label'].eq(1).sum():,} risky, "
        f"{usable['rugcheck_label'].eq(0).sum():,} safe"
    )
    log(f"token features: {len(token_columns)}; GraphSAGE features: {len(graphsage_columns)}")

    splits, distribution_rows = build_splits(usable, args.min_class_per_year)
    feature_groups = {
        "token": token_columns,
        "token_graphsage": combined_columns,
    }

    all_results: list[dict[str, Any]] = []
    all_calibration: list[dict[str, Any]] = []
    all_importance: list[pd.DataFrame] = []
    for split_name, (train, test) in splits.items():
        baseline_results, baseline_calibration = evaluate_baselines(split_name, test)
        train_results, train_calibration, importance = train_and_score(split_name, train, test, feature_groups)
        all_results.extend(baseline_results)
        all_results.extend(train_results)
        all_calibration.extend(baseline_calibration)
        all_calibration.extend(train_calibration)
        all_importance.extend(importance)

    results = pd.DataFrame(all_results)
    calibration = pd.DataFrame(all_calibration)
    importance = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    distribution = pd.DataFrame(distribution_rows)

    results.to_csv(args.results_out, index=False)
    calibration.to_csv(args.calibration_out, index=False)
    importance.to_csv(args.importance_out, index=False)
    distribution.to_csv(args.distribution_out, index=False)

    log(f"saved results: {args.results_out}")
    log(f"saved threshold calibration: {args.calibration_out}")
    log(f"saved feature importance: {args.importance_out}")
    log(f"saved train/test distribution: {args.distribution_out}")
    print(graphsage_improvement_message(results))


if __name__ == "__main__":
    main()
