from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dune_2024_2025_eda_ml_gnn_pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    add_heuristic_rug_labels,
    clean_model_frame,
    create_token_events,
    load_or_build_feature_data,
    load_raw_dune_swaps,
    safe_divide,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "dune_paper_roadmap_2024_2025"
WEAK_LABEL_PATH = OUTPUT_DIR / "weak_label_diagnostics.csv"
TEMPORAL_METRICS_PATH = OUTPUT_DIR / "temporal_2024_to_2025_metrics.csv"
EARLY_WINDOW_METRICS_PATH = OUTPUT_DIR / "early_window_metrics.csv"
ABLATION_METRICS_PATH = OUTPUT_DIR / "ablation_metrics.csv"
PAPER_SUMMARY_PATH = OUTPUT_DIR / "paper_roadmap_summary.json"
EARLY_FEATURES_PATH = OUTPUT_DIR / "early_window_features.csv"


WINDOW_HOURS = [1, 6, 24]

FEATURE_GROUPS = {
    "token_activity_only": [
        "activity_count",
        "buy_count",
        "sell_count",
        "active_days",
        "log_activity_count",
    ],
    "volume_pressure_only": [
        "total_volume",
        "buy_volume_usd",
        "sell_volume_usd",
        "imbalance_ratio",
        "sell_pressure",
        "log_total_volume",
    ],
    "wallet_graph_only": [
        "unique_wallets",
        "graph_degree",
        "connected_entities",
        "entity_concentration_ratio",
        "volume_per_wallet",
        "activity_per_wallet",
    ],
    "lifetime_only": [
        "lifespan_hours",
        "active_days",
    ],
    "all_token_features": FEATURE_COLUMNS,
}


def log(message: str) -> None:
    print(f"[dune-paper-roadmap] {message}", flush=True)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true_np = y_true.astype(int).to_numpy()
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true_np, scores)) if len(np.unique(y_true_np)) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y_true_np, scores)) if len(np.unique(y_true_np)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_hgb_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(random_state=42, max_iter=160, learning_rate=0.06)),
        ]
    )


def make_logistic_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )


def make_rf_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=180, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]
    )


def predict_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    raw = model.decision_function(frame)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


def add_weak_label_diagnostics(labeled: pd.DataFrame) -> pd.DataFrame:
    diagnostics = labeled.copy()
    evidence_columns = ["high_sell_imbalance", "concentrated_wallet_flow", "short_lived"]
    diagnostics["weak_label_vote_count"] = diagnostics[evidence_columns].sum(axis=1)
    diagnostics["weak_label_consensus"] = np.select(
        [
            diagnostics["weak_label_vote_count"].ge(3),
            diagnostics["weak_label_vote_count"].eq(2),
            diagnostics["weak_label_vote_count"].eq(1),
        ],
        ["strong", "medium", "weak"],
        default="none",
    )
    summary = (
        diagnostics.groupby(["year", "heuristic_rug_label", "weak_label_consensus"], as_index=False)
        .agg(tokens=("token_address", "nunique"), median_score=("heuristic_rug_score", "median"))
        .sort_values(["year", "heuristic_rug_label", "weak_label_consensus"])
    )
    summary.to_csv(WEAK_LABEL_PATH, index=False)
    return diagnostics


def run_temporal_transfer(labeled: pd.DataFrame) -> pd.DataFrame:
    log("Running 2024-to-2025 temporal transfer experiments")
    data = clean_model_frame(labeled)
    train = data[data["year"].eq(2024)].copy()
    test = data[data["year"].eq(2025)].copy()
    models = {
        "logistic_regression": make_logistic_model(),
        "hist_gradient_boosting": make_hgb_model(),
        "random_forest": make_rf_model(),
    }
    rows = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[FEATURE_COLUMNS], train["heuristic_rug_label"])
        scores = predict_scores(model, test[FEATURE_COLUMNS])
        metric = evaluate(test["heuristic_rug_label"], scores)
        metric.update({"experiment": "temporal_transfer", "model": name, "train_year": 2024, "test_year": 2025, "features": "all_token_features"})
        rows.append(metric)
    metrics = pd.DataFrame(rows).sort_values(["f1", "average_precision"], ascending=False)
    metrics.to_csv(TEMPORAL_METRICS_PATH, index=False)
    return metrics


def engineer_window_features(events: pd.DataFrame, full_labels: pd.DataFrame, window_hours: int) -> pd.DataFrame:
    first_seen = events.groupby(["year", "token_address"], as_index=False).agg(first_seen_window=("block_time", "min"))
    windowed = events.merge(first_seen, on=["year", "token_address"], how="inner")
    max_time = windowed["first_seen_window"] + pd.to_timedelta(window_hours, unit="h")
    windowed = windowed[windowed["block_time"].le(max_time)].copy()

    side_counts = (
        windowed.pivot_table(index=["year", "token_address"], columns="side", values="tx_id", aggfunc="count", fill_value=0)
        .rename(columns={"buy": "buy_count", "sell": "sell_count"})
        .reset_index()
    )
    side_volume = (
        windowed.pivot_table(index=["year", "token_address"], columns="side", values="amount_usd", aggfunc="sum", fill_value=0)
        .rename(columns={"buy": "buy_volume_usd", "sell": "sell_volume_usd"})
        .reset_index()
    )
    base = (
        windowed.groupby(["year", "token_address"], as_index=False)
        .agg(
            activity_count=("tx_id", "count"),
            total_volume=("amount_usd", "sum"),
            unique_wallets=("trader_id", "nunique"),
            first_seen=("block_time", "min"),
            last_seen=("block_time", "max"),
            active_days=("block_date", "nunique"),
        )
    )
    wallet_token = (
        windowed.groupby(["year", "token_address", "trader_id"], as_index=False)
        .agg(wallet_volume=("amount_usd", "sum"))
    )
    concentration = (
        wallet_token.groupby(["year", "token_address"], as_index=False)
        .agg(max_wallet_volume=("wallet_volume", "max"), graph_degree=("trader_id", "nunique"))
    )
    features = base.merge(side_counts, on=["year", "token_address"], how="left").merge(side_volume, on=["year", "token_address"], how="left").merge(concentration, on=["year", "token_address"], how="left")
    for column in ["buy_count", "sell_count", "buy_volume_usd", "sell_volume_usd", "max_wallet_volume", "graph_degree"]:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    features["lifespan_hours"] = (features["last_seen"] - features["first_seen"]).dt.total_seconds().div(3600).fillna(0.0)
    features["connected_entities"] = features["graph_degree"]
    features["imbalance_ratio"] = safe_divide(features["sell_volume_usd"], features["buy_volume_usd"])
    features.loc[(features["buy_volume_usd"].eq(0)) & (features["sell_volume_usd"].gt(0)), "imbalance_ratio"] = features["sell_volume_usd"]
    features["entity_concentration_ratio"] = safe_divide(features["max_wallet_volume"], features["total_volume"])
    features["buy_sell_count_ratio"] = safe_divide(features["buy_count"], features["sell_count"])
    features["sell_pressure"] = safe_divide(features["sell_volume_usd"], features["total_volume"])
    features["volume_per_wallet"] = safe_divide(features["total_volume"], features["unique_wallets"])
    features["activity_per_wallet"] = safe_divide(features["activity_count"], features["unique_wallets"])
    features["log_total_volume"] = np.log1p(features["total_volume"].clip(lower=0))
    features["log_activity_count"] = np.log1p(features["activity_count"].clip(lower=0))
    labels = full_labels[["year", "token_address", "heuristic_rug_label", "heuristic_rug_score"]]
    features = features.drop(columns=["max_wallet_volume"]).merge(labels, on=["year", "token_address"], how="inner")
    features["window_hours"] = window_hours
    return features


def run_early_window_experiments(events: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    log("Running early-warning feature-window experiments")
    feature_frames = [engineer_window_features(events, labeled, hours) for hours in WINDOW_HOURS]
    all_windows = pd.concat(feature_frames, ignore_index=True)
    all_windows.to_csv(EARLY_FEATURES_PATH, index=False)

    rows = []
    for hours, frame in all_windows.groupby("window_hours"):
        frame = clean_model_frame(frame)
        train = frame[frame["year"].eq(2024)].copy()
        test = frame[frame["year"].eq(2025)].copy()
        if train["heuristic_rug_label"].nunique() < 2 or test["heuristic_rug_label"].nunique() < 2:
            continue
        model = make_hgb_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[FEATURE_COLUMNS], train["heuristic_rug_label"])
        scores = predict_scores(model, test[FEATURE_COLUMNS])
        metric = evaluate(test["heuristic_rug_label"], scores)
        metric.update({"experiment": "early_window_temporal_transfer", "model": "hist_gradient_boosting", "window_hours": int(hours), "train_year": 2024, "test_year": 2025})
        rows.append(metric)
    metrics = pd.DataFrame(rows).sort_values("window_hours")
    metrics.to_csv(EARLY_WINDOW_METRICS_PATH, index=False)
    return metrics


def run_ablation_experiments(labeled: pd.DataFrame) -> pd.DataFrame:
    log("Running feature-family ablation experiments")
    data = clean_model_frame(labeled)
    train = data[data["year"].eq(2024)].copy()
    test = data[data["year"].eq(2025)].copy()
    rows = []
    for group_name, columns in FEATURE_GROUPS.items():
        valid_columns = [column for column in columns if column in data.columns]
        model = make_hgb_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[valid_columns], train["heuristic_rug_label"])
        scores = predict_scores(model, test[valid_columns])
        metric = evaluate(test["heuristic_rug_label"], scores)
        metric.update({"experiment": "feature_ablation_temporal_transfer", "model": "hist_gradient_boosting", "feature_group": group_name, "feature_count": len(valid_columns)})
        rows.append(metric)
    metrics = pd.DataFrame(rows).sort_values(["f1", "average_precision"], ascending=False)
    metrics.to_csv(ABLATION_METRICS_PATH, index=False)
    return metrics


def main() -> None:
    ensure_output_dir()
    labeled, _events_from_gold = load_or_build_feature_data()
    labeled, thresholds = add_heuristic_rug_labels(labeled)
    labeled = add_weak_label_diagnostics(labeled)

    swaps = load_raw_dune_swaps()
    events = create_token_events(swaps)

    temporal_metrics = run_temporal_transfer(labeled)
    early_metrics = run_early_window_experiments(events, labeled)
    ablation_metrics = run_ablation_experiments(labeled)

    summary = {
        "research_stage": "paper-oriented roadmap experiments",
        "labeling": "weak/silver heuristic labels, not verified ground truth",
        "thresholds": thresholds,
        "label_counts": labeled.groupby("year")["heuristic_rug_label"].value_counts().unstack(fill_value=0).to_dict(),
        "best_temporal_model": temporal_metrics.iloc[0].to_dict() if not temporal_metrics.empty else {},
        "best_early_window": early_metrics.sort_values(["f1", "average_precision"], ascending=False).iloc[0].to_dict() if not early_metrics.empty else {},
        "best_ablation": ablation_metrics.iloc[0].to_dict() if not ablation_metrics.empty else {},
        "output_files": {
            "weak_label_diagnostics": str(WEAK_LABEL_PATH),
            "temporal_metrics": str(TEMPORAL_METRICS_PATH),
            "early_window_metrics": str(EARLY_WINDOW_METRICS_PATH),
            "ablation_metrics": str(ABLATION_METRICS_PATH),
            "early_features": str(EARLY_FEATURES_PATH),
        },
    }
    PAPER_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Paper roadmap experiments completed.")
    print(f"Temporal transfer metrics: {TEMPORAL_METRICS_PATH}")
    print(f"Early-window metrics: {EARLY_WINDOW_METRICS_PATH}")
    print(f"Ablation metrics: {ABLATION_METRICS_PATH}")
    print(f"Weak-label diagnostics: {WEAK_LABEL_PATH}")
    print(f"Summary: {PAPER_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
