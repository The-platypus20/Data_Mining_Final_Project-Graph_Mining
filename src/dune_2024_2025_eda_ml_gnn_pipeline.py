from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DUNE_DIR = PROJECT_ROOT / "data" / "raw" / "dune"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "dune_eda_ml_gnn_2024_2025"
EDA_DIR = OUTPUT_DIR / "eda"
ML_DIR = OUTPUT_DIR / "ml"
GNN_DIR = OUTPUT_DIR / "gnn"

TOKEN_FEATURES_PATH = PROJECT_ROOT / "data" / "gold" / "dune_token_features_2024_2025.parquet"
PYG_NODES_PATH = PROJECT_ROOT / "data" / "gold" / "pyg_nodes_2024_2025.parquet"
PYG_EDGES_PATH = PROJECT_ROOT / "data" / "gold" / "pyg_edges_2024_2025.parquet"

LABELED_FEATURES_PATH = OUTPUT_DIR / "heuristic_labeled_token_features.csv"
EDA_SUMMARY_PATH = EDA_DIR / "eda_summary.json"
FEATURE_DEMO_PATH = EDA_DIR / "feature_demonstrations.csv"
MODEL_METRICS_PATH = ML_DIR / "model_metrics.csv"
MODEL_PREDICTIONS_PATH = ML_DIR / "model_predictions_2025.csv"
MODEL_IMPORTANCE_PATH = ML_DIR / "feature_importance.csv"
GNN_SCORES_PATH = GNN_DIR / "gnn_or_graph_scores_2025.csv"
GNN_NOTES_PATH = GNN_DIR / "gnn_notes.txt"

BASE_MINTS = {
    "So11111111111111111111111111111111111111112",
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "Es9vMFrzaCERmJfrF4H2FYD4QGLXGdEmC8v4HtFf4gJ",
}

RAW_COLUMNS = [
    "block_time",
    "block_date",
    "project",
    "trade_source",
    "token_bought_mint_address",
    "token_sold_mint_address",
    "token_bought_amount",
    "token_sold_amount",
    "amount_usd",
    "fee_usd",
    "trader_id",
    "tx_id",
]

FEATURE_COLUMNS = [
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


def log(message: str) -> None:
    print(f"[dune-2024-2025] {message}", flush=True)


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, EDA_DIR, ML_DIR, GNN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0)


def find_raw_files() -> list[Path]:
    files = sorted((RAW_DUNE_DIR / "2024").glob("*.csv")) + sorted((RAW_DUNE_DIR / "2025").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No 2024-2025 Dune CSV files found under {RAW_DUNE_DIR}")
    return files


def load_raw_dune_swaps() -> pd.DataFrame:
    frames = []
    for path in find_raw_files():
        frame = pd.read_csv(path, usecols=RAW_COLUMNS, na_values=["", "NULL", "nil", "<nil>"])
        missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        year, month = path.stem.split("_")
        frame["year"] = int(year)
        frame["month"] = int(month)
        frame["source_file"] = path.name
        frames.append(frame)

    swaps = pd.concat(frames, ignore_index=True)
    swaps["block_time"] = pd.to_datetime(swaps["block_time"].astype("string").str.replace(" UTC", "", regex=False), utc=True, errors="coerce")
    swaps["block_date"] = pd.to_datetime(swaps["block_date"], errors="coerce")
    for column in ["token_bought_amount", "token_sold_amount", "amount_usd", "fee_usd"]:
        swaps[column] = pd.to_numeric(swaps[column], errors="coerce")
    swaps = swaps.dropna(subset=["block_time", "token_bought_mint_address", "token_sold_mint_address", "trader_id", "tx_id", "amount_usd"])
    swaps = swaps[swaps["amount_usd"].ge(0)].copy()
    return swaps


def create_token_events(swaps: pd.DataFrame) -> pd.DataFrame:
    common = ["block_time", "block_date", "project", "trade_source", "amount_usd", "fee_usd", "trader_id", "tx_id", "year", "month"]
    buys = swaps[common].copy()
    buys["token_address"] = swaps["token_bought_mint_address"].astype(str)
    buys["counterparty_token"] = swaps["token_sold_mint_address"].astype(str)
    buys["token_amount"] = swaps["token_bought_amount"]
    buys["side"] = "buy"

    sells = swaps[common].copy()
    sells["token_address"] = swaps["token_sold_mint_address"].astype(str)
    sells["counterparty_token"] = swaps["token_bought_mint_address"].astype(str)
    sells["token_amount"] = swaps["token_sold_amount"]
    sells["side"] = "sell"

    events = pd.concat([buys, sells], ignore_index=True)
    events = events[~events["token_address"].isin(BASE_MINTS)].copy()
    events = events.dropna(subset=["token_address", "trader_id", "tx_id"])
    return events


def engineer_token_features(events: pd.DataFrame) -> pd.DataFrame:
    side_counts = (
        events.pivot_table(index=["year", "token_address"], columns="side", values="tx_id", aggfunc="count", fill_value=0)
        .rename(columns={"buy": "buy_count", "sell": "sell_count"})
        .reset_index()
    )
    side_volume = (
        events.pivot_table(index=["year", "token_address"], columns="side", values="amount_usd", aggfunc="sum", fill_value=0)
        .rename(columns={"buy": "buy_volume_usd", "sell": "sell_volume_usd"})
        .reset_index()
    )
    token_base = (
        events.groupby(["year", "token_address"], as_index=False)
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
        events.groupby(["year", "token_address", "trader_id"], as_index=False)
        .agg(wallet_volume=("amount_usd", "sum"), wallet_activity=("tx_id", "count"))
    )
    concentration = (
        wallet_token.groupby(["year", "token_address"], as_index=False)
        .agg(max_wallet_volume=("wallet_volume", "max"), graph_degree=("trader_id", "nunique"))
    )
    features = (
        token_base.merge(side_counts, on=["year", "token_address"], how="left")
        .merge(side_volume, on=["year", "token_address"], how="left")
        .merge(concentration, on=["year", "token_address"], how="left")
    )
    for column in ["buy_count", "sell_count", "buy_volume_usd", "sell_volume_usd", "max_wallet_volume", "graph_degree"]:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    features["lifespan_hours"] = (features["last_seen"] - features["first_seen"]).dt.total_seconds().div(3600).fillna(0.0)
    features["connected_entities"] = features["graph_degree"]
    features["imbalance_ratio"] = safe_divide(features["sell_volume_usd"], features["buy_volume_usd"])
    features.loc[(features["buy_volume_usd"].eq(0)) & (features["sell_volume_usd"].gt(0)), "imbalance_ratio"] = features["sell_volume_usd"]
    features["entity_concentration_ratio"] = safe_divide(features["max_wallet_volume"], features["total_volume"])
    return add_derived_features(features.drop(columns=["max_wallet_volume"]))


def add_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    numeric = [
        "activity_count",
        "buy_count",
        "sell_count",
        "total_volume",
        "buy_volume_usd",
        "sell_volume_usd",
        "unique_wallets",
        "lifespan_hours",
        "graph_degree",
        "connected_entities",
        "entity_concentration_ratio",
        "active_days",
    ]
    for column in numeric:
        if column not in features:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["buy_sell_count_ratio"] = safe_divide(features["buy_count"], features["sell_count"])
    features["sell_pressure"] = safe_divide(features["sell_volume_usd"], features["total_volume"])
    features["volume_per_wallet"] = safe_divide(features["total_volume"], features["unique_wallets"])
    features["activity_per_wallet"] = safe_divide(features["activity_count"], features["unique_wallets"])
    features["log_total_volume"] = np.log1p(features["total_volume"].clip(lower=0))
    features["log_activity_count"] = np.log1p(features["activity_count"].clip(lower=0))
    features["is_base_mint"] = features["token_address"].isin(BASE_MINTS).astype(int)
    return features[features["is_base_mint"].eq(0)].copy()


def load_or_build_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if TOKEN_FEATURES_PATH.exists():
        log(f"Loading existing gold token features: {TOKEN_FEATURES_PATH}")
        features = pd.read_parquet(TOKEN_FEATURES_PATH)
        features = add_derived_features(features)
        swaps = load_raw_dune_swaps()
        events = create_token_events(swaps)
        return features, events

    log("Gold token features not found, rebuilding directly from raw 2024-2025 Dune CSVs")
    swaps = load_raw_dune_swaps()
    events = create_token_events(swaps)
    features = engineer_token_features(events)
    return features, events


def quantile(series: pd.Series, q: float, default: float = 0.0) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return default
    return float(clean.quantile(q))


def add_heuristic_rug_labels(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    labeled = features.copy()
    baseline = labeled[labeled["year"].eq(2024)].copy()
    if baseline.empty:
        baseline = labeled.copy()

    thresholds = {
        "imbalance_ratio_p85": quantile(baseline["imbalance_ratio"], 0.85),
        "sell_pressure_p75": quantile(baseline["sell_pressure"], 0.75),
        "entity_concentration_ratio_p75": quantile(baseline["entity_concentration_ratio"], 0.75),
        "lifespan_hours_p35": quantile(baseline["lifespan_hours"], 0.35),
        "activity_count_p55": quantile(baseline["activity_count"], 0.55),
        "total_volume_p40": quantile(baseline["total_volume"], 0.40),
        "unique_wallets_p35": quantile(baseline["unique_wallets"], 0.35),
    }
    evidence = pd.DataFrame(index=labeled.index)
    evidence["high_sell_imbalance"] = labeled["imbalance_ratio"].ge(thresholds["imbalance_ratio_p85"]) | labeled["sell_pressure"].ge(thresholds["sell_pressure_p75"])
    evidence["concentrated_wallet_flow"] = labeled["entity_concentration_ratio"].ge(thresholds["entity_concentration_ratio_p75"])
    evidence["short_lived"] = labeled["lifespan_hours"].le(thresholds["lifespan_hours_p35"])
    evidence["enough_activity"] = labeled["activity_count"].ge(max(2.0, thresholds["activity_count_p55"]))
    evidence["enough_volume"] = labeled["total_volume"].ge(max(10.0, thresholds["total_volume_p40"]))
    evidence["enough_wallets"] = labeled["unique_wallets"].ge(max(2.0, thresholds["unique_wallets_p35"]))
    labeled["rug_heuristic_evidence_count"] = evidence[["high_sell_imbalance", "concentrated_wallet_flow", "short_lived"]].sum(axis=1)

    score_parts = pd.DataFrame(index=labeled.index)
    for column in ["imbalance_ratio", "sell_pressure", "entity_concentration_ratio", "activity_count", "total_volume"]:
        values = np.log1p(pd.to_numeric(labeled[column], errors="coerce").fillna(0.0).clip(lower=0.0))
        score_parts[column] = (values - values.min()) / (values.max() - values.min() + 1e-12)
    lifespan = pd.to_numeric(labeled["lifespan_hours"], errors="coerce").fillna(0.0).clip(lower=0.0)
    score_parts["short_lifespan"] = 1.0 - ((lifespan - lifespan.min()) / (lifespan.max() - lifespan.min() + 1e-12))
    labeled["heuristic_rug_score"] = (
        0.25 * score_parts["imbalance_ratio"]
        + 0.20 * score_parts["sell_pressure"]
        + 0.20 * score_parts["entity_concentration_ratio"]
        + 0.15 * score_parts["short_lifespan"]
        + 0.10 * score_parts["activity_count"]
        + 0.10 * score_parts["total_volume"]
    ).clip(0.0, 1.0)
    score_cutoff = quantile(labeled.loc[labeled["year"].eq(2024), "heuristic_rug_score"], 0.95)
    thresholds["heuristic_rug_score_p95"] = score_cutoff

    strict_rule = (
        evidence["enough_activity"]
        & evidence["enough_volume"]
        & evidence["enough_wallets"]
        & labeled["rug_heuristic_evidence_count"].ge(2)
    )
    high_tail_rule = (
        evidence["enough_activity"]
        & evidence["enough_volume"]
        & evidence["enough_wallets"]
        & labeled["heuristic_rug_score"].ge(score_cutoff)
        & labeled["rug_heuristic_evidence_count"].ge(1)
    )
    labeled["heuristic_rug_label"] = (strict_rule | high_tail_rule).astype(int)
    for column in evidence.columns:
        labeled[column] = evidence[column].astype(int)
    return labeled, thresholds


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_deep_eda(features: pd.DataFrame, events: pd.DataFrame, thresholds: dict[str, float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    log("Running deep EDA and feature demonstrations")
    numeric_cols = [column for column in FEATURE_COLUMNS if column in features.columns]
    summary = {
        "raw_scope": "Only Dune swap CSVs from data/raw/dune/2024 and data/raw/dune/2025 are used.",
        "token_feature_rows": int(len(features)),
        "event_rows": int(len(events)),
        "year_counts": features["year"].value_counts().sort_index().to_dict(),
        "label_distribution": features.groupby("year")["heuristic_rug_label"].value_counts().unstack(fill_value=0).to_dict(),
        "missing_by_column": features.isna().sum().sort_values(ascending=False).head(30).to_dict(),
        "numeric_summary": features[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
        "heuristic_thresholds_from_2024": thresholds,
        "top_projects": events["project"].value_counts().head(15).to_dict(),
        "top_trade_sources": events["trade_source"].value_counts().head(15).to_dict(),
    }
    write_json(EDA_SUMMARY_PATH, summary)

    feature_demo_columns = [
        "year",
        "token_address",
        "heuristic_rug_label",
        "rug_heuristic_evidence_count",
        "activity_count",
        "total_volume",
        "sell_pressure",
        "imbalance_ratio",
        "entity_concentration_ratio",
        "lifespan_hours",
        "unique_wallets",
    ]
    demo = pd.concat(
        [
            features.sort_values("total_volume", ascending=False).head(10).assign(demo_reason="highest total volume"),
            features.sort_values("sell_pressure", ascending=False).head(10).assign(demo_reason="highest sell pressure"),
            features.sort_values("entity_concentration_ratio", ascending=False).head(10).assign(demo_reason="highest wallet concentration"),
            features[features["heuristic_rug_label"].eq(1)].sort_values("rug_heuristic_evidence_count", ascending=False).head(20).assign(demo_reason="heuristic rug-positive examples"),
        ],
        ignore_index=True,
    )
    demo[["demo_reason", *feature_demo_columns]].drop_duplicates().to_csv(FEATURE_DEMO_PATH, index=False)

    yearly = features.groupby("year", as_index=False).agg(
        tokens=("token_address", "nunique"),
        heuristic_rugs=("heuristic_rug_label", "sum"),
        median_volume=("total_volume", "median"),
        median_lifespan_hours=("lifespan_hours", "median"),
        median_wallets=("unique_wallets", "median"),
    )
    yearly.to_csv(EDA_DIR / "yearly_feature_summary.csv", index=False)

    monthly = events.groupby(["year", "month"], as_index=False).agg(
        events=("tx_id", "count"),
        wallets=("trader_id", "nunique"),
        tokens=("token_address", "nunique"),
        volume_usd=("amount_usd", "sum"),
    )
    monthly.to_csv(EDA_DIR / "monthly_event_summary.csv", index=False)

    corr = features[numeric_cols].corr(method="spearman").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr.to_csv(EDA_DIR / "spearman_correlation.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=yearly, x="year", y="tokens", color="#4c78a8")
    plt.title("Dune Token Coverage by Year")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "token_coverage_by_year.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly, x="month", y="volume_usd", hue="year", marker="o")
    plt.yscale("symlog")
    plt.title("Monthly Dune Token-Side Volume")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "monthly_volume.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_frame = features.sample(min(len(features), 20000), random_state=42)
    sns.scatterplot(data=plot_frame, x="entity_concentration_ratio", y="sell_pressure", hue="heuristic_rug_label", s=12, alpha=0.5)
    plt.title("Feature Demonstration: Concentration vs Sell Pressure")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "concentration_vs_sell_pressure.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, square=False)
    plt.title("Spearman Feature Correlation")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "feature_correlation_heatmap.png", dpi=160)
    plt.close()


def clean_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in FEATURE_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    clean[FEATURE_COLUMNS] = clean[FEATURE_COLUMNS].fillna(0.0)
    return clean


def build_supervised_models(class_weight: dict[int, float] | str = "balanced") -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=42)),
            ]
        ),
        "linear_svc": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight=class_weight, random_state=42, max_iter=5000)),
            ]
        ),
        "knn": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight=class_weight, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight=class_weight, random_state=42, n_jobs=-1)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=250, min_samples_leaf=5, class_weight=class_weight, random_state=42, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42, max_iter=160, learning_rate=0.06)),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.002, max_iter=220, random_state=42, early_stopping=True)),
            ]
        ),
    }


def predict_scores(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        raw = np.asarray(raw, dtype=float)
        if raw.max() == raw.min():
            return np.zeros_like(raw, dtype=float)
        return (raw - raw.min()) / (raw.max() - raw.min())
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return {key: float(value) for key, value in metrics.items()}


def run_machine_learning(labeled: pd.DataFrame) -> pd.DataFrame:
    log("Training classical machine-learning models on heuristic labels")
    data = clean_model_frame(labeled)
    train_pool = data[data["year"].eq(2024)].copy()
    score_2025 = data[data["year"].eq(2025)].copy()
    if train_pool["heuristic_rug_label"].nunique() < 2:
        raise ValueError("Heuristic labels have only one class in 2024; supervised ML cannot be evaluated.")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, valid_idx = next(splitter.split(train_pool[FEATURE_COLUMNS], train_pool["heuristic_rug_label"]))
    train = train_pool.iloc[train_idx].copy()
    valid = train_pool.iloc[valid_idx].copy()

    models = build_supervised_models()
    metrics_rows = []
    prediction_frame = score_2025[["year", "token_address", "heuristic_rug_label", "rug_heuristic_evidence_count", "total_volume", "sell_pressure", "entity_concentration_ratio", "lifespan_hours", "unique_wallets"]].copy()
    importances = []

    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[FEATURE_COLUMNS], train["heuristic_rug_label"])
        valid_scores = predict_scores(model, valid[FEATURE_COLUMNS])
        metric = evaluate_scores(valid["heuristic_rug_label"].to_numpy(), valid_scores)
        metric.update({"model": name, "validation_scope": "2024 stratified holdout"})
        metrics_rows.append(metric)
        prediction_frame[f"{name}_risk_score"] = predict_scores(model, score_2025[FEATURE_COLUMNS]) if not score_2025.empty else []

        fitted = model.named_steps.get("model") if isinstance(model, Pipeline) else model
        if hasattr(fitted, "feature_importances_"):
            for feature, value in zip(FEATURE_COLUMNS, fitted.feature_importances_):
                importances.append({"model": name, "feature": feature, "importance": float(value)})
        elif hasattr(fitted, "coef_"):
            coefs = np.ravel(fitted.coef_)
            for feature, value in zip(FEATURE_COLUMNS, np.abs(coefs)):
                importances.append({"model": name, "feature": feature, "importance": float(value)})

    iso = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1)),
        ]
    )
    iso.fit(train_pool[FEATURE_COLUMNS])
    anomaly_valid = -iso.named_steps["model"].decision_function(iso.named_steps["scaler"].transform(iso.named_steps["imputer"].transform(valid[FEATURE_COLUMNS])))
    anomaly_valid = (anomaly_valid - anomaly_valid.min()) / (anomaly_valid.max() - anomaly_valid.min() + 1e-12)
    metric = evaluate_scores(valid["heuristic_rug_label"].to_numpy(), anomaly_valid)
    metric.update({"model": "isolation_forest_unsupervised", "validation_scope": "2024 stratified holdout"})
    metrics_rows.append(metric)
    anomaly_2025 = -iso.named_steps["model"].decision_function(iso.named_steps["scaler"].transform(iso.named_steps["imputer"].transform(score_2025[FEATURE_COLUMNS])))
    prediction_frame["isolation_forest_unsupervised_risk_score"] = (anomaly_2025 - anomaly_2025.min()) / (anomaly_2025.max() - anomaly_2025.min() + 1e-12) if len(anomaly_2025) else []

    score_columns = [column for column in prediction_frame.columns if column.endswith("_risk_score")]
    prediction_frame["mean_ml_risk_score"] = prediction_frame[score_columns].mean(axis=1)
    prediction_frame["ml_risk_rank"] = prediction_frame["mean_ml_risk_score"].rank(method="first", ascending=False).astype(int)
    prediction_frame = prediction_frame.sort_values("mean_ml_risk_score", ascending=False)

    metrics = pd.DataFrame(metrics_rows).sort_values(["f1", "average_precision"], ascending=False)
    metrics.to_csv(MODEL_METRICS_PATH, index=False)
    prediction_frame.to_csv(MODEL_PREDICTIONS_PATH, index=False)
    pd.DataFrame(importances).sort_values(["model", "importance"], ascending=[True, False]).to_csv(MODEL_IMPORTANCE_PATH, index=False)
    return metrics


def build_wallet_token_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for row in nodes[["node_id", "node_type", "original_id", "year"]].itertuples(index=False):
        graph.add_node(row.node_id, node_type=row.node_type, original_id=row.original_id, year=int(row.year))
    for row in edges[["src_node_id", "dst_node_id", "total_volume_usd", "total_count"]].itertuples(index=False):
        graph.add_edge(row.src_node_id, row.dst_node_id, weight=float(row.total_volume_usd), count=float(row.total_count))
    return graph


def try_run_graphsage_gnn(labeled: pd.DataFrame) -> bool:
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch_geometric.nn import SAGEConv
    except Exception as exc:
        GNN_NOTES_PATH.write_text(
            "GraphSAGE was selected as the best GNN type because the natural Solana Dune structure is a "
            "heterogeneous wallet-token bipartite interaction graph. GraphSAGE is suitable for inductive "
            "wallet-token neighborhoods and can score unseen token neighborhoods.\n\n"
            f"PyTorch/PyTorch Geometric is unavailable in this environment, so the script ran graph-aware "
            f"NetworkX fallback scores instead. Import error: {exc}\n",
            encoding="utf-8",
        )
        return False

    if not PYG_NODES_PATH.exists() or not PYG_EDGES_PATH.exists():
        GNN_NOTES_PATH.write_text("PyG graph tables are missing, so GraphSAGE was skipped.", encoding="utf-8")
        return False

    nodes = pd.read_parquet(PYG_NODES_PATH)
    edges = pd.read_parquet(PYG_EDGES_PATH)
    label_map = labeled[["year", "token_address", "heuristic_rug_label"]].rename(columns={"token_address": "original_id"})
    nodes = nodes.merge(label_map, on=["year", "original_id"], how="left")
    for column in FEATURE_COLUMNS[:13]:
        if column not in nodes:
            nodes[column] = 0.0
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0.0)

    node_lookup = {node_id: i for i, node_id in enumerate(nodes["node_id"].astype(str))}
    src = edges["src_node_id"].astype(str).map(node_lookup)
    dst = edges["dst_node_id"].astype(str).map(node_lookup)
    valid_edges = src.notna() & dst.notna()
    edge_pairs = np.vstack([src[valid_edges].astype(int), dst[valid_edges].astype(int)])
    edge_index = torch.tensor(np.hstack([edge_pairs, edge_pairs[[1, 0], :]]), dtype=torch.long)

    gnn_features = FEATURE_COLUMNS[:13]
    x_np = np.log1p(nodes[gnn_features].clip(lower=0).to_numpy(dtype=np.float32))
    x_np = np.nan_to_num(x_np)
    train_mask_np = nodes["node_type"].eq("token") & nodes["year"].eq(2024) & nodes["heuristic_rug_label"].notna()
    train_indices = np.flatnonzero(train_mask_np.to_numpy())
    if len(train_indices) < 20 or nodes.loc[train_mask_np, "heuristic_rug_label"].nunique() < 2:
        GNN_NOTES_PATH.write_text("GraphSAGE skipped because there were not enough labeled 2024 token nodes.", encoding="utf-8")
        return False

    mean = x_np[train_indices].mean(axis=0)
    std = x_np[train_indices].std(axis=0)
    std[std == 0] = 1
    x = torch.tensor((x_np - mean) / std, dtype=torch.float32)
    y = torch.tensor(nodes["heuristic_rug_label"].fillna(0).to_numpy(dtype=np.float32), dtype=torch.float32)
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.25,
        random_state=42,
        stratify=nodes.loc[train_mask_np, "heuristic_rug_label"].astype(int),
    )

    class GraphSAGE(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.conv1 = SAGEConv(input_dim, 64)
            self.conv2 = SAGEConv(64, 32)
            self.out = nn.Linear(32, 1)

        def forward(self, x_tensor, edge_idx):
            hidden = F.relu(self.conv1(x_tensor, edge_idx))
            hidden = F.dropout(hidden, p=0.25, training=self.training)
            embedding = F.relu(self.conv2(hidden, edge_idx))
            return self.out(embedding).squeeze(-1), embedding

    model = GraphSAGE(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    positives = float(y[train_idx].sum().item())
    negatives = float(len(train_idx) - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx, dtype=torch.long)
    log_rows = []
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, edge_index)
        loss = F.binary_cross_entropy_with_logits(logits[train_tensor], y[train_tensor], pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(x, edge_index)
                val_scores = torch.sigmoid(val_logits[val_tensor]).numpy()
            metrics = evaluate_scores(y[val_idx].numpy().astype(int), val_scores)
            log_rows.append({"epoch": epoch, "loss": float(loss.item()), **metrics})

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(x, edge_index)
        scores = torch.sigmoid(logits).numpy()
    nodes["gnn_risk_score"] = scores
    token_scores = nodes[nodes["node_type"].eq("token") & nodes["year"].eq(2025)].copy()
    token_scores = token_scores.rename(columns={"original_id": "token_address"})
    token_scores[["year", "token_address", "gnn_risk_score"]].sort_values("gnn_risk_score", ascending=False).to_csv(GNN_SCORES_PATH, index=False)
    pd.DataFrame(log_rows).to_csv(GNN_DIR / "graphsage_training_log.csv", index=False)
    GNN_NOTES_PATH.write_text(
        "Ran GraphSAGE on the wallet-token bipartite graph. This graph type is preferred for this Solana Dune "
        "project because wallets connect directly to traded token mints, preserving concentration, repeated "
        "trader behavior, and neighborhood risk signals.\n",
        encoding="utf-8",
    )
    return True


def run_graph_fallback(labeled: pd.DataFrame) -> None:
    log("Running graph-aware fallback scores with NetworkX")
    if not PYG_NODES_PATH.exists() or not PYG_EDGES_PATH.exists():
        GNN_NOTES_PATH.write_text("Graph fallback skipped because PyG graph tables are missing.", encoding="utf-8")
        return

    nodes = pd.read_parquet(PYG_NODES_PATH)
    edges = pd.read_parquet(PYG_EDGES_PATH)
    graph = build_wallet_token_graph(nodes, edges)
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight", max_iter=100, tol=1e-6)
    token_nodes = nodes[nodes["node_type"].eq("token")].copy()
    token_nodes["degree"] = token_nodes["node_id"].map(degree).fillna(0)
    token_nodes["weighted_degree"] = token_nodes["node_id"].map(weighted_degree).fillna(0)
    token_nodes["pagerank"] = token_nodes["node_id"].map(pagerank).fillna(0)
    token_nodes = token_nodes.rename(columns={"original_id": "token_address"})
    token_nodes = token_nodes.merge(
        labeled[["year", "token_address", "heuristic_rug_label", "sell_pressure", "entity_concentration_ratio", "lifespan_hours", "total_volume"]],
        on=["year", "token_address"],
        how="left",
        suffixes=("", "_labeled"),
    )
    for column in ["entity_concentration_ratio", "lifespan_hours", "total_volume"]:
        labeled_column = f"{column}_labeled"
        if labeled_column in token_nodes.columns:
            token_nodes[column] = pd.to_numeric(token_nodes[labeled_column], errors="coerce").fillna(
                pd.to_numeric(token_nodes.get(column, 0.0), errors="coerce")
            )
    for column in ["degree", "weighted_degree", "pagerank", "sell_pressure", "entity_concentration_ratio", "total_volume"]:
        values = pd.to_numeric(token_nodes[column], errors="coerce").fillna(0.0)
        token_nodes[f"{column}_scaled"] = (values - values.min()) / (values.max() - values.min() + 1e-12)
    lifespan = pd.to_numeric(token_nodes["lifespan_hours"], errors="coerce").fillna(0.0)
    token_nodes["short_lifespan_scaled"] = 1.0 - ((lifespan - lifespan.min()) / (lifespan.max() - lifespan.min() + 1e-12))
    token_nodes["graph_fallback_risk_score"] = (
        0.25 * token_nodes["pagerank_scaled"]
        + 0.20 * token_nodes["weighted_degree_scaled"]
        + 0.20 * token_nodes["sell_pressure_scaled"]
        + 0.20 * token_nodes["entity_concentration_ratio_scaled"]
        + 0.15 * token_nodes["short_lifespan_scaled"]
    ).clip(0.0, 1.0)
    token_nodes[token_nodes["year"].eq(2025)][
        ["year", "token_address", "graph_fallback_risk_score", "heuristic_rug_label", "degree", "weighted_degree", "pagerank"]
    ].sort_values("graph_fallback_risk_score", ascending=False).to_csv(GNN_SCORES_PATH, index=False)


def main() -> None:
    ensure_dirs()
    features, events = load_or_build_feature_data()
    labeled, thresholds = add_heuristic_rug_labels(features)
    labeled.to_csv(LABELED_FEATURES_PATH, index=False)
    run_deep_eda(labeled, events, thresholds)
    metrics = run_machine_learning(labeled)
    gnn_ran = try_run_graphsage_gnn(labeled)
    if not gnn_ran:
        run_graph_fallback(labeled)

    best_model = metrics.iloc[0]["model"] if not metrics.empty else "none"
    print("Dune 2024-2025 EDA + ML + GNN pipeline completed.")
    print(f"Heuristic-labeled features: {LABELED_FEATURES_PATH}")
    print(f"EDA summary: {EDA_SUMMARY_PATH}")
    print(f"Feature demonstrations: {FEATURE_DEMO_PATH}")
    print(f"ML metrics: {MODEL_METRICS_PATH}")
    print(f"2025 ML predictions: {MODEL_PREDICTIONS_PATH}")
    print(f"GNN/graph scores: {GNN_SCORES_PATH}")
    print(f"Best 2024 holdout model by F1/AP: {best_model}")
    print("Important: heuristic_rug_label is a rule-based research label, not confirmed rug-pull ground truth.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Pipeline failed: {exc}")
        raise
