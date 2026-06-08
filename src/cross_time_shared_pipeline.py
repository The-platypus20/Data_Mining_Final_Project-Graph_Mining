from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "cross_time"

HISTORICAL_FILES = {
    2021: "2021.csv",
    2022: "2022.csv",
    2023: "2023.csv",
    2024: "Jan_2024-Nov_2024.csv",
}
RECENT_PATTERN = "2025_*.csv"

# Do not model base routing assets as risky tokens. They appear on one side of
# many swaps and would dominate the 2025 aggregates.
BASE_MINTS = {
    "So11111111111111111111111111111111111111112",  # wrapped SOL
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4QGLXGdEmC8v4HtFf4gJ",  # USDT
}

COMMON_FEATURES = [
    "lifespan_hours",
    "activity_count",
    "total_volume",
    "imbalance_ratio",
    "entity_concentration_ratio",
    "graph_degree",
    "connected_entities",
]


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=float),
        where=denominator_array != 0,
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["", "NULL", "nil", "<nil>"])


def parse_timestamp_series(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    parsed = pd.to_datetime(text, utc=True, errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")

    # The supplied 2024 file has time fragments such as "41:11.0". Treat those
    # as offsets within a day so we can still compute a relative lifespan.
    seconds = text.str.extract(r"^(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)$")
    minutes = pd.to_numeric(seconds["minutes"], errors="coerce")
    secs = pd.to_numeric(seconds["seconds"], errors="coerce")
    offsets = pd.to_timedelta(minutes * 60 + secs, unit="s")
    fragment_ts = pd.Timestamp("2024-01-01", tz="UTC") + offsets
    return parsed.fillna(fragment_ts)


def hours_between(start: pd.Series, end: pd.Series) -> pd.Series:
    start_ts = parse_timestamp_series(start)
    end_ts = parse_timestamp_series(end)
    delta_hours = (end_ts - start_ts).dt.total_seconds() / 3600

    # Time fragments can wrap around midnight/hour boundaries. A single-day
    # modulo keeps the feature non-negative without creating synthetic dates.
    delta_hours = delta_hours.where(delta_hours >= 0, delta_hours + 24)
    return delta_hours


def load_historical_features(raw_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year, filename in HISTORICAL_FILES.items():
        path = raw_dir / filename
        frame = read_csv(path)
        required = {
            "LIQUIDITY_POOL_ADDRESS",
            "MINT",
            "TOTAL_ADDED_LIQUIDITY",
            "TOTAL_REMOVED_LIQUIDITY",
            "NUM_LIQUIDITY_ADDS",
            "NUM_LIQUIDITY_REMOVES",
            "FIRST_POOL_ACTIVITY_TIMESTAMP",
            "LAST_POOL_ACTIVITY_TIMESTAMP",
            "INACTIVITY_STATUS",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required historical columns: {missing}")

        for column in [
            "TOTAL_ADDED_LIQUIDITY",
            "TOTAL_REMOVED_LIQUIDITY",
            "NUM_LIQUIDITY_ADDS",
            "NUM_LIQUIDITY_REMOVES",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["year"] = year
        frame["mint"] = frame["MINT"].astype(str)
        frame["pool"] = frame["LIQUIDITY_POOL_ADDRESS"].astype(str)
        frame["row_volume"] = (
            frame["TOTAL_ADDED_LIQUIDITY"].fillna(0) + frame["TOTAL_REMOVED_LIQUIDITY"].fillna(0)
        )
        frame["row_activity"] = (
            frame["NUM_LIQUIDITY_ADDS"].fillna(0) + frame["NUM_LIQUIDITY_REMOVES"].fillna(0)
        )
        frame["row_lifespan_hours"] = hours_between(
            frame["FIRST_POOL_ACTIVITY_TIMESTAMP"], frame["LAST_POOL_ACTIVITY_TIMESTAMP"]
        )
        frame["rug_label"] = frame["INACTIVITY_STATUS"].astype(str).str.lower().eq("inactive").astype(int)
        frames.append(frame)

    rows = pd.concat(frames, ignore_index=True)
    rows = rows[~rows["mint"].isin(BASE_MINTS)].copy()
    rows = rows.dropna(subset=["mint", "rug_label"])

    pool_token = (
        rows.groupby(["year", "mint", "pool"], as_index=False)
        .agg(pool_volume=("row_volume", "sum"), pool_activity=("row_activity", "sum"))
    )
    pool_token["pool_rank"] = pool_token.groupby(["year", "mint"])["pool_volume"].rank(
        method="first", ascending=False
    )
    token_pool_totals = (
        pool_token.groupby(["year", "mint"], as_index=False)
        .agg(token_pool_volume=("pool_volume", "sum"))
    )
    top_pool = (
        pool_token[pool_token["pool_rank"] == 1]
        .groupby(["year", "mint"], as_index=False)
        .agg(top_pool_volume=("pool_volume", "sum"))
    )

    graph_features = compute_bipartite_graph_features(
        edges=pool_token.rename(columns={"pool": "entity", "pool_volume": "weight"}),
        token_column="mint",
        entity_column="entity",
        year_column="year",
    )

    features = (
        rows.groupby(["year", "mint"], as_index=False)
        .agg(
            lifespan_hours=("row_lifespan_hours", "max"),
            activity_count=("row_activity", "sum"),
            total_added=("TOTAL_ADDED_LIQUIDITY", "sum"),
            total_removed=("TOTAL_REMOVED_LIQUIDITY", "sum"),
            total_volume=("row_volume", "sum"),
            rug_label=("rug_label", "max"),
        )
        .merge(token_pool_totals, on=["year", "mint"], how="left")
        .merge(top_pool, on=["year", "mint"], how="left")
        .merge(graph_features, on=["year", "mint"], how="left")
    )
    features["imbalance_ratio"] = safe_divide(features["total_removed"], features["total_added"])
    features["entity_concentration_ratio"] = safe_divide(
        features["top_pool_volume"].fillna(0), features["token_pool_volume"].fillna(0)
    )
    features["source_schema"] = "historical_liquidity"
    return clean_feature_frame(features)


def load_recent_trade_events(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(raw_dir.glob(RECENT_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No recent trade files found with pattern {RECENT_PATTERN}")

    required = {
        "block_time",
        "token_bought_mint_address",
        "token_sold_mint_address",
        "amount_usd",
        "trader_id",
        "tx_id",
    }
    frames = []
    for path in paths:
        frame = read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required trade columns: {missing}")
        frame = frame[list(required)].copy()
        frame["source_file"] = path.name
        frames.append(frame)

    trades = pd.concat(frames, ignore_index=True)
    trades["block_time"] = pd.to_datetime(trades["block_time"], utc=True, errors="coerce")
    trades["amount_usd"] = pd.to_numeric(trades["amount_usd"], errors="coerce")
    trades = trades.dropna(
        subset=[
            "block_time",
            "token_bought_mint_address",
            "token_sold_mint_address",
            "amount_usd",
            "trader_id",
            "tx_id",
        ]
    )
    trades = trades[trades["amount_usd"] > 0].copy()

    buys = trades[["block_time", "amount_usd", "trader_id", "tx_id"]].copy()
    buys["mint"] = trades["token_bought_mint_address"].astype(str)
    buys["side"] = "buy"

    sells = trades[["block_time", "amount_usd", "trader_id", "tx_id"]].copy()
    sells["mint"] = trades["token_sold_mint_address"].astype(str)
    sells["side"] = "sell"

    events = pd.concat([buys, sells], ignore_index=True)
    events = events[~events["mint"].isin(BASE_MINTS)].copy()
    return events


def engineer_recent_features(events: pd.DataFrame) -> pd.DataFrame:
    side_counts = (
        events.pivot_table(index="mint", columns="side", values="tx_id", aggfunc="count", fill_value=0)
        .rename(columns={"buy": "buy_count", "sell": "sell_count"})
        .reset_index()
    )
    side_volume = (
        events.pivot_table(index="mint", columns="side", values="amount_usd", aggfunc="sum", fill_value=0)
        .rename(columns={"buy": "buy_volume", "sell": "sell_volume"})
        .reset_index()
    )
    token_features = (
        events.groupby("mint", as_index=False)
        .agg(
            first_activity=("block_time", "min"),
            last_activity=("block_time", "max"),
            activity_count=("tx_id", "count"),
            total_volume=("amount_usd", "sum"),
        )
        .merge(side_counts, on="mint", how="left")
        .merge(side_volume, on="mint", how="left")
    )

    for column in ["buy_count", "sell_count", "buy_volume", "sell_volume"]:
        token_features[column] = token_features[column].fillna(0)

    wallet_token = (
        events.groupby(["mint", "trader_id"], as_index=False)
        .agg(wallet_volume=("amount_usd", "sum"), wallet_activity=("tx_id", "count"))
    )
    wallet_token["wallet_rank"] = wallet_token.groupby("mint")["wallet_volume"].rank(
        method="first", ascending=False
    )
    top_wallet = (
        wallet_token[wallet_token["wallet_rank"] == 1]
        .groupby("mint", as_index=False)
        .agg(top_wallet_volume=("wallet_volume", "sum"))
    )

    graph_features = compute_bipartite_graph_features(
        edges=wallet_token.rename(columns={"trader_id": "entity", "wallet_volume": "weight"}),
        token_column="mint",
        entity_column="entity",
    )

    token_features = token_features.merge(top_wallet, on="mint", how="left").merge(
        graph_features, on="mint", how="left"
    )
    token_features["lifespan_hours"] = (
        token_features["last_activity"] - token_features["first_activity"]
    ).dt.total_seconds() / 3600
    token_features["imbalance_ratio"] = safe_divide(token_features["sell_volume"], token_features["buy_volume"])
    token_features["entity_concentration_ratio"] = safe_divide(
        token_features["top_wallet_volume"].fillna(0), token_features["total_volume"].fillna(0)
    )
    token_features["year"] = 2025
    token_features["source_schema"] = "recent_swaps"
    token_features = create_recent_heuristic_labels(token_features)
    return clean_feature_frame(token_features)


def compute_bipartite_graph_features(
    edges: pd.DataFrame,
    token_column: str,
    entity_column: str,
    year_column: str | None = None,
) -> pd.DataFrame:
    group_columns = [year_column] if year_column else []
    group_columns = [column for column in group_columns if column is not None]

    frames = []
    grouped: Iterable[tuple[object, pd.DataFrame]]
    grouped = edges.groupby(group_columns, dropna=False) if group_columns else [(None, edges)]
    for key, group in grouped:
        graph = nx.Graph()
        aggregated = group.groupby([token_column, entity_column], as_index=False)["weight"].sum()
        for row in aggregated.itertuples(index=False):
            token = f"token:{getattr(row, token_column)}"
            entity = f"entity:{getattr(row, entity_column)}"
            graph.add_edge(token, entity, weight=float(getattr(row, "weight")))

        token_values = sorted(group[token_column].astype(str).unique())
        records = []
        for mint in token_values:
            node = f"token:{mint}"
            records.append(
                {
                    "mint": mint,
                    "connected_entities": int(graph.degree(node)) if graph.has_node(node) else 0,
                    "graph_weighted_degree": float(graph.degree(node, weight="weight")) if graph.has_node(node) else 0.0,
                }
            )
        frame = pd.DataFrame(records)
        if year_column:
            frame[year_column] = key[0] if isinstance(key, tuple) else key
        frames.append(frame)

    if not frames:
        columns = ["mint", "connected_entities", "graph_weighted_degree"]
        if year_column:
            columns.append(year_column)
        return pd.DataFrame(columns=columns)

    result = pd.concat(frames, ignore_index=True)
    result["graph_degree"] = result["connected_entities"]
    return result


def create_recent_heuristic_labels(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    features["heuristic_rug_label"] = (
        (features["lifespan_hours"] <= 24)
        & (features["imbalance_ratio"] >= 2.0)
        & (features["entity_concentration_ratio"] >= 0.50)
    ).astype(int)
    return features


def clean_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for column in COMMON_FEATURES:
        if column not in features.columns:
            features[column] = np.nan
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features[COMMON_FEATURES] = features[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan)
    return features


def build_models(random_state: int, y_train: pd.Series) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        positives = int(y_train.sum())
        negatives = int((y_train == 0).sum())
        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=negatives / positives if positives else 1.0,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except ImportError:
        warnings.warn("xgboost is not installed; skipping XGBoost.")

    return models


def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_values = safe_divide(2 * precision * recall, precision + recall)
    if len(thresholds) == 0:
        return 0.5, float(f1_values[0]) if len(f1_values) else math.nan
    best_index = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_index]), float(f1_values[best_index])


def train_validate_apply(
    historical: pd.DataFrame,
    recent: pd.DataFrame,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = historical[historical["year"].isin([2021, 2022, 2023])].dropna(subset=["rug_label"]).copy()
    valid = historical[historical["year"] == 2024].dropna(subset=["rug_label"]).copy()
    if train["rug_label"].nunique() < 2 or valid["rug_label"].nunique() < 2:
        raise ValueError("Train and validation splits both need positive and negative labels.")

    x_train = train[COMMON_FEATURES]
    y_train = train["rug_label"].astype(int)
    x_valid = valid[COMMON_FEATURES]
    y_valid = valid["rug_label"].astype(int)
    x_recent = recent[COMMON_FEATURES]

    metric_rows = []
    validation_predictions = []
    recent_predictions = []
    importance_frames = []
    threshold_sensitivity_frames = []

    for model_name, model in build_models(random_state, y_train).items():
        model.fit(x_train, y_train)
        valid_scores = model.predict_proba(x_valid)[:, 1]
        threshold, best_f1 = best_f1_threshold(y_valid, valid_scores)
        valid_pred = (valid_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, valid_pred, labels=[0, 1]).ravel()

        metric_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "validation_accuracy": accuracy_score(y_valid, valid_pred),
                "validation_precision": precision_score(y_valid, valid_pred, zero_division=0),
                "validation_f1": f1_score(y_valid, valid_pred, zero_division=0),
                "validation_best_f1": best_f1,
                "validation_recall": recall_score(y_valid, valid_pred, zero_division=0),
                "validation_pr_auc": average_precision_score(y_valid, valid_scores),
                "validation_roc_auc": roc_auc_score(y_valid, valid_scores),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "train_rows": len(train),
                "validation_rows": len(valid),
            }
        )

        valid_frame = valid[["year", "mint", "rug_label"]].copy()
        valid_frame["model"] = model_name
        valid_frame["rug_probability"] = valid_scores
        valid_frame["predicted_rug_label"] = valid_pred
        validation_predictions.append(valid_frame)

        recent_scores = model.predict_proba(x_recent)[:, 1]
        recent_pred = (recent_scores >= threshold).astype(int)
        recent_heuristic = recent["heuristic_rug_label"].astype(int)
        metric_rows[-1].update(
            {
                "recent_2025_rows": len(recent),
                "recent_2025_predicted_risk_rate": float(np.mean(recent_pred)),
                "recent_2025_heuristic_risk_rate": float(recent_heuristic.mean()),
                "recent_2025_agreement_with_heuristic": float(np.mean(recent_pred == recent_heuristic)),
                "recent_2025_f1_vs_heuristic": f1_score(recent_heuristic, recent_pred, zero_division=0),
                "recent_2025_recall_vs_heuristic": recall_score(recent_heuristic, recent_pred, zero_division=0),
            }
        )
        recent_frame = recent[["year", "mint", "heuristic_rug_label"]].copy()
        recent_frame["model"] = model_name
        recent_frame["rug_probability"] = recent_scores
        recent_frame["predicted_rug_label"] = recent_pred
        recent_predictions.append(recent_frame)
        threshold_sensitivity_frames.append(
            threshold_sensitivity_2025(
                model_name=model_name,
                scores=recent_scores,
                heuristic_labels=recent_heuristic,
                validation_threshold=threshold,
            )
        )

        fitted_model = model.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importance = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            importance = np.abs(fitted_model.coef_[0])
        else:
            importance = None

        if importance is not None:
            importance_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "feature": COMMON_FEATURES,
                        "importance": importance,
                    }
                ).sort_values(["model", "importance"], ascending=[True, False])
            )

    metrics = pd.DataFrame(metric_rows).sort_values("validation_f1", ascending=False)
    validation_output = pd.concat(validation_predictions, ignore_index=True)
    recent_output = pd.concat(recent_predictions, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    threshold_sensitivity = pd.concat(threshold_sensitivity_frames, ignore_index=True)
    return metrics, validation_output, recent_output, feature_importance, threshold_sensitivity


def threshold_sensitivity_2025(
    model_name: str,
    scores: np.ndarray,
    heuristic_labels: pd.Series,
    validation_threshold: float,
) -> pd.DataFrame:
    heuristic = heuristic_labels.astype(int).to_numpy()
    total_tokens = len(heuristic)

    strategies = [
        ("default_0.5", 0.5, scores >= 0.5),
        ("best_f1_threshold_2024", validation_threshold, scores >= validation_threshold),
    ]
    for top_rate in [0.01, 0.05, 0.10]:
        if total_tokens:
            cutoff_count = max(1, int(math.ceil(total_tokens * top_rate)))
            ranked_indices = np.argsort(scores, kind="mergesort")[-cutoff_count:]
            cutoff_score = float(np.min(scores[ranked_indices]))
            predictions = np.zeros(total_tokens, dtype=bool)
            predictions[ranked_indices] = True
        else:
            cutoff_score = math.nan
            predictions = np.array([], dtype=bool)
        strategies.append((f"top_{int(top_rate * 100)}pct_highest_risk", cutoff_score, predictions))

    rows = []
    heuristic_rug_rate = float(np.mean(heuristic)) if total_tokens else math.nan
    for strategy, threshold_value, predictions in strategies:
        predicted = predictions.astype(int)
        rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "threshold_value": threshold_value,
                "total_tokens": total_tokens,
                "predicted_rug_tokens": int(predicted.sum()),
                "predicted_rug_rate": float(predicted.mean()) if total_tokens else math.nan,
                "heuristic_rug_rate": heuristic_rug_rate,
                "agreement_with_heuristic": float(np.mean(predicted == heuristic)) if total_tokens else math.nan,
                "f1_vs_heuristic": f1_score(heuristic, predicted, zero_division=0),
                "recall_vs_heuristic": recall_score(heuristic, predicted, zero_division=0),
                "precision_vs_heuristic": precision_score(heuristic, predicted, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def label_distribution_report(historical: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame, label_column in [
        ("train_2021_2023", historical[historical["year"].isin([2021, 2022, 2023])], "rug_label"),
        ("validation_2024", historical[historical["year"] == 2024], "rug_label"),
        ("historical_2021_2024", historical, "rug_label"),
        ("heuristic_2025", recent, "heuristic_rug_label"),
    ]:
        labels = frame[label_column].dropna().astype(int)
        rows.append(
            {
                "dataset": name,
                "total_tokens": int(len(labels)),
                "rug_flag_1": int((labels == 1).sum()),
                "rug_flag_0": int((labels == 0).sum()),
                "rug_flag_1_rate": float((labels == 1).mean()) if len(labels) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ks_statistic(left: pd.Series, right: pd.Series) -> float:
    left_values = np.sort(left.dropna().to_numpy(dtype=float))
    right_values = np.sort(right.dropna().to_numpy(dtype=float))
    if len(left_values) == 0 or len(right_values) == 0:
        return math.nan

    values = np.sort(np.unique(np.concatenate([left_values, right_values])))
    left_cdf = np.searchsorted(left_values, values, side="right") / len(left_values)
    right_cdf = np.searchsorted(right_values, values, side="right") / len(right_values)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def distribution_shift_report(historical: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in COMMON_FEATURES:
        hist = historical[feature]
        rec = recent[feature]
        hist_mean = hist.mean()
        rec_mean = rec.mean()
        pooled_std = math.sqrt((hist.var(ddof=0) + rec.var(ddof=0)) / 2)
        rows.append(
            {
                "feature": feature,
                "historical_mean": hist_mean,
                "recent_2025_mean": rec_mean,
                "historical_median": hist.median(),
                "recent_2025_median": rec.median(),
                "standardized_mean_difference": (rec_mean - hist_mean) / pooled_std if pooled_std else math.nan,
                "ks_statistic": ks_statistic(hist, rec),
                "historical_missing_rate": hist.isna().mean(),
                "recent_2025_missing_rate": rec.isna().mean(),
            }
        )
    drift = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)
    drift["drift_severity"] = drift["ks_statistic"].apply(classify_drift_severity)
    drift["drift_interpretation"] = drift["drift_severity"].map(
        {
            "low": "Feature distribution is relatively stable.",
            "medium": "Feature shows noticeable distribution shift.",
            "high": "Feature shows severe distribution shift and may hurt transfer.",
            "unknown": "Drift could not be computed.",
        }
    )
    return drift


def classify_drift_severity(ks_statistic: float) -> str:
    if pd.isna(ks_statistic):
        return "unknown"
    if ks_statistic < 0.2:
        return "low"
    if ks_statistic < 0.5:
        return "medium"
    return "high"


def write_outputs(
    output_dir: Path,
    historical: pd.DataFrame,
    recent: pd.DataFrame,
    metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    recent_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    label_distribution: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    historical.to_csv(output_dir / "aligned_historical_features.csv", index=False)
    recent.to_csv(output_dir / "aligned_2025_features.csv", index=False)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions_2024.csv", index=False)
    recent_predictions.to_csv(output_dir / "predictions_2025.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    label_distribution.to_csv(output_dir / "label_distribution.csv", index=False)
    threshold_sensitivity.to_csv(output_dir / "threshold_sensitivity_2025.csv", index=False)
    drift.to_csv(output_dir / "distribution_shift.csv", index=False)

    high_drift_features = drift.loc[drift["drift_severity"] == "high", "feature"].tolist()
    summary = {
        "common_features": COMMON_FEATURES,
        "training_years": [2021, 2022, 2023],
        "validation_year": 2024,
        "prediction_year": 2025,
        "historical_rows": int(len(historical)),
        "recent_2025_rows": int(len(recent)),
        "best_model_by_validation_f1": metrics.iloc[0].to_dict() if len(metrics) else None,
        "label_distribution": label_distribution.to_dict(orient="records"),
        "prediction_behavior_2025": (
            recent_predictions.groupby("model")["predicted_rug_label"]
            .agg(total_tokens="count", predicted_rug_tokens="sum", predicted_rug_rate="mean")
            .reset_index()
            .to_dict(orient="records")
        ),
        "top_5_feature_importance": (
            feature_importance.sort_values(["model", "importance"], ascending=[True, False])
            .groupby("model")
            .head(5)
            .to_dict(orient="records")
            if len(feature_importance)
            else []
        ),
        "threshold_sensitivity_summary": threshold_sensitivity.to_dict(orient="records"),
        "high_drift_feature_count": len(high_drift_features),
        "high_drift_features": high_drift_features,
        "feature_alignment": {
            "lifespan_hours": "Historical: last-first pool activity. 2025: last-first swap activity.",
            "activity_count": "Historical: liquidity adds+removes. 2025: token-side swap events.",
            "total_volume": "Historical: added+removed liquidity amount. 2025: swap USD volume.",
            "imbalance_ratio": "Historical: removed/added liquidity. 2025: sell/buy USD volume.",
            "entity_concentration_ratio": "Historical: top liquidity pool share. 2025: top wallet volume share.",
            "graph_degree": "Historical: token-pool bipartite degree. 2025: token-wallet bipartite degree.",
            "connected_entities": "Same as graph degree, retained with an interpretable name.",
        },
        "limitations": [
            "2025 labels are heuristic risk labels only, not ground truth rug labels.",
            "Historical wallet-level concentration is unavailable, so top pool share is used as a proxy.",
            "Historical volume is liquidity amount, while 2025 volume is swap USD; drift should be expected.",
            "2024 timestamps in the provided file are time fragments, so validation lifespan is relative only.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_pipeline(raw_dir: Path, output_dir: Path, random_state: int) -> None:
    historical = load_historical_features(raw_dir)
    recent_events = load_recent_trade_events(raw_dir)
    recent = engineer_recent_features(recent_events)
    (
        metrics,
        validation_predictions,
        recent_predictions,
        feature_importance,
        threshold_sensitivity,
    ) = train_validate_apply(
        historical=historical,
        recent=recent,
        random_state=random_state,
    )
    label_distribution = label_distribution_report(historical, recent)
    drift = distribution_shift_report(historical, recent)
    write_outputs(
        output_dir,
        historical,
        recent,
        metrics,
        validation_predictions,
        recent_predictions,
        feature_importance,
        label_distribution,
        threshold_sensitivity,
        drift,
    )

    print("\nValidation metrics:")
    print(metrics.to_string(index=False))
    print("\nLabel distribution:")
    print(label_distribution.to_string(index=False))
    print("\n2025 prediction behavior:")
    print(
        recent_predictions.groupby("model")["predicted_rug_label"]
        .agg(total_tokens="count", predicted_rug_tokens="sum", predicted_rug_rate="mean")
        .reset_index()
        .to_string(index=False)
    )
    print("\nTop 5 feature importances per model:")
    print(feature_importance.groupby("model").head(5).to_string(index=False))
    print("\n2025 threshold sensitivity:")
    print(threshold_sensitivity.to_string(index=False))
    print("\nLargest feature shifts:")
    print(drift.head(10).to_string(index=False))
    print(f"\nWrote outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-time rug-risk pipeline with shared features.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.raw_dir, args.output_dir, args.random_state)
