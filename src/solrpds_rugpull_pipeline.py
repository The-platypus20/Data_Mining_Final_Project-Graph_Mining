from __future__ import annotations

import argparse
import json
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
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

IDENTIFIER_COLUMNS = [
    "mint",
    "first_activity_time",
    "last_activity_time",
]

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# PART 1 - LOAD DATA
# ---------------------------------------------------------------------------

def find_input_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} matching {pattern!r}")
    return files


def load_trade_data(input_files: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for file_path in input_files:
        frame = pd.read_csv(file_path, na_values=["<nil>", "nil", "NULL", ""])
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ValueError(f"{file_path} is missing columns: {missing_columns}")
        frame = frame[REQUIRED_COLUMNS].copy()
        frame["source_file"] = file_path.name
        frames.append(frame)

    trades = pd.concat(frames, ignore_index=True)
    trades["block_time"] = pd.to_datetime(trades["block_time"], utc=True, errors="coerce")
    trades["block_date"] = pd.to_datetime(trades["block_date"], errors="coerce").dt.date

    numeric_columns = [
        "token_bought_amount",
        "token_sold_amount",
        "amount_usd",
        "fee_usd",
    ]
    for column in numeric_columns:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")

    required_non_missing = [
        "block_time",
        "token_bought_mint_address",
        "token_sold_mint_address",
        "trader_id",
        "amount_usd",
    ]
    trades = trades.dropna(subset=required_non_missing)
    trades = trades[trades["amount_usd"] > 0].copy()
    trades["fee_usd"] = trades["fee_usd"].fillna(0.0)
    return trades


# ---------------------------------------------------------------------------
# PART 2 - CREATE TOKEN-SIDE EVENTS
# ---------------------------------------------------------------------------

def create_token_side_events(trades: pd.DataFrame) -> pd.DataFrame:
    common_columns = [
        "block_time",
        "block_date",
        "project",
        "trade_source",
        "amount_usd",
        "fee_usd",
        "trader_id",
        "tx_id",
    ]

    buys = trades[common_columns].copy()
    buys["mint"] = trades["token_bought_mint_address"]
    buys["counterparty_mint"] = trades["token_sold_mint_address"]
    buys["token_amount"] = trades["token_bought_amount"]
    buys["side"] = "buy"

    sells = trades[common_columns].copy()
    sells["mint"] = trades["token_sold_mint_address"]
    sells["counterparty_mint"] = trades["token_bought_mint_address"]
    sells["token_amount"] = trades["token_sold_amount"]
    sells["side"] = "sell"

    columns = [
        "block_time",
        "block_date",
        "project",
        "trade_source",
        "mint",
        "counterparty_mint",
        "token_amount",
        "amount_usd",
        "fee_usd",
        "trader_id",
        "tx_id",
        "side",
    ]
    events = pd.concat([buys[columns], sells[columns]], ignore_index=True)
    events["token_amount"] = pd.to_numeric(events["token_amount"], errors="coerce")
    return events


# ---------------------------------------------------------------------------
# PART 3 - FEATURE ENGINEERING (TOKEN LEVEL)
# ---------------------------------------------------------------------------

def engineer_token_features(events: pd.DataFrame) -> pd.DataFrame:
    side_counts = (
        events.pivot_table(index="mint", columns="side", values="tx_id", aggfunc="count", fill_value=0)
        .rename(columns={"buy": "num_buy_swaps", "sell": "num_sell_swaps"})
        .reset_index()
    )
    side_volumes = (
        events.pivot_table(index="mint", columns="side", values="amount_usd", aggfunc="sum", fill_value=0)
        .rename(columns={"buy": "buy_volume_usd", "sell": "sell_volume_usd"})
        .reset_index()
    )

    token_stats = (
        events.groupby("mint")
        .agg(
            total_swaps=("tx_id", "count"),
            total_volume_usd=("amount_usd", "sum"),
            first_activity_time=("block_time", "min"),
            last_activity_time=("block_time", "max"),
            unique_wallets=("trader_id", "nunique"),
            avg_trade_size=("amount_usd", "mean"),
            max_trade_size=("amount_usd", "max"),
        )
        .reset_index()
    )

    features = token_stats.merge(side_counts, on="mint", how="left").merge(side_volumes, on="mint", how="left")
    for column in ["num_buy_swaps", "num_sell_swaps", "buy_volume_usd", "sell_volume_usd"]:
        if column not in features:
            features[column] = 0
        features[column] = features[column].fillna(0)

    features["lifespan_hours"] = (
        features["last_activity_time"] - features["first_activity_time"]
    ).dt.total_seconds() / 3600
    features["swaps_per_wallet"] = safe_divide(features["total_swaps"], features["unique_wallets"])
    features["volume_per_wallet"] = safe_divide(features["total_volume_usd"], features["unique_wallets"])
    return features


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=float),
        where=denominator_array != 0,
    )


# ---------------------------------------------------------------------------
# PART 4 - WALLET CONCENTRATION
# ---------------------------------------------------------------------------

def compute_wallet_concentration(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wallet_token = (
        events.groupby(["mint", "trader_id"])
        .agg(wallet_volume_usd=("amount_usd", "sum"), wallet_swap_count=("tx_id", "count"))
        .reset_index()
    )
    wallet_token["volume_rank"] = wallet_token.groupby("mint")["wallet_volume_usd"].rank(
        method="first", ascending=False
    )
    wallet_token["swap_rank"] = wallet_token.groupby("mint")["wallet_swap_count"].rank(
        method="first", ascending=False
    )

    token_totals = (
        wallet_token.groupby("mint")
        .agg(token_wallet_volume_usd=("wallet_volume_usd", "sum"), token_wallet_swap_count=("wallet_swap_count", "sum"))
        .reset_index()
    )

    top_volume = (
        wallet_token.assign(
            top1_volume=lambda df: np.where(df["volume_rank"] <= 1, df["wallet_volume_usd"], 0),
            top5_volume=lambda df: np.where(df["volume_rank"] <= 5, df["wallet_volume_usd"], 0),
            top1_swaps=lambda df: np.where(df["swap_rank"] <= 1, df["wallet_swap_count"], 0),
            top5_swaps=lambda df: np.where(df["swap_rank"] <= 5, df["wallet_swap_count"], 0),
        )
        .groupby("mint")
        .agg(
            top1_volume=("top1_volume", "sum"),
            top5_volume=("top5_volume", "sum"),
            top1_swaps=("top1_swaps", "sum"),
            top5_swaps=("top5_swaps", "sum"),
        )
        .reset_index()
        .merge(token_totals, on="mint", how="left")
    )

    top_volume["top1_wallet_volume_ratio"] = safe_divide(
        top_volume["top1_volume"], top_volume["token_wallet_volume_usd"]
    )
    top_volume["top5_wallet_volume_ratio"] = safe_divide(
        top_volume["top5_volume"], top_volume["token_wallet_volume_usd"]
    )
    top_volume["top1_wallet_swap_ratio"] = safe_divide(
        top_volume["top1_swaps"], top_volume["token_wallet_swap_count"]
    )
    top_volume["top5_wallet_swap_ratio"] = safe_divide(
        top_volume["top5_swaps"], top_volume["token_wallet_swap_count"]
    )
    concentration = top_volume[
        [
            "mint",
            "top1_wallet_volume_ratio",
            "top5_wallet_volume_ratio",
            "top1_wallet_swap_ratio",
            "top5_wallet_swap_ratio",
        ]
    ].copy()
    return concentration, wallet_token


# ---------------------------------------------------------------------------
# PART 5 - GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

def build_graph_edges(events: pd.DataFrame) -> pd.DataFrame:
    edges = events[["trader_id", "mint", "amount_usd", "side", "block_time"]].copy()
    edges = edges.rename(
        columns={
            "trader_id": "source",
            "mint": "target",
            "amount_usd": "weight",
            "block_time": "timestamp",
        }
    )
    edges["timestamp"] = edges["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return edges


# ---------------------------------------------------------------------------
# PART 6 - GRAPH FEATURES
# ---------------------------------------------------------------------------

def compute_graph_features(edges: pd.DataFrame, token_mints: pd.Series) -> pd.DataFrame:
    graph = nx.Graph()
    token_nodes = [f"token:{mint}" for mint in token_mints.astype(str)]
    graph.add_nodes_from(token_nodes, node_type="token")

    aggregated = edges.groupby(["source", "target"], as_index=False)["weight"].sum()
    for row in aggregated.itertuples(index=False):
        wallet_node = f"wallet:{row.source}"
        token_node = f"token:{row.target}"
        graph.add_node(wallet_node, node_type="wallet")
        graph.add_node(token_node, node_type="token")
        graph.add_edge(wallet_node, token_node, weight=float(row.weight))

    degree_map = dict(graph.degree(token_nodes))
    weighted_degree_map = dict(graph.degree(token_nodes, weight="weight"))
    clustering_map = nx.clustering(graph, nodes=token_nodes)

    # Degree centrality is optional but useful as a simple graph-derived signal.
    centrality_map = nx.degree_centrality(graph) if graph.number_of_nodes() > 1 else {}
    graph_features = pd.DataFrame(
        {
            "mint": token_mints.astype(str),
            "graph_degree": [degree_map.get(node, 0) for node in token_nodes],
            "graph_weighted_degree": [weighted_degree_map.get(node, 0.0) for node in token_nodes],
            "connected_wallets": [degree_map.get(node, 0) for node in token_nodes],
            "graph_clustering_coefficient": [clustering_map.get(node, 0.0) for node in token_nodes],
            "graph_degree_centrality": [centrality_map.get(node, 0.0) for node in token_nodes],
        }
    )
    return graph_features


# ---------------------------------------------------------------------------
# PART 7 - CREATE LABELS (HEURISTIC)
# ---------------------------------------------------------------------------

def create_heuristic_labels(features: pd.DataFrame) -> pd.DataFrame:
    sell_buy_ratio = safe_divide(features["sell_volume_usd"], features["buy_volume_usd"])
    features = features.copy()
    features["sell_buy_volume_ratio"] = sell_buy_ratio
    features["rug_flag"] = (
        (features["lifespan_hours"] < 24)
        & (features["top1_wallet_volume_ratio"] > 0.5)
        & (features["sell_buy_volume_ratio"] > 2)
    ).astype(int)
    return features


# ---------------------------------------------------------------------------
# PART 8 / PART 11 - BASELINE MODELS AND XGBOOST MODEL
# ---------------------------------------------------------------------------

def prepare_model_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    excluded = set(IDENTIFIER_COLUMNS + ["rug_flag"])
    candidate_columns = [column for column in features.columns if column not in excluded]
    model_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(features[column]) and column != "mint"
    ]
    x = features[model_columns].replace([np.inf, -np.inf], np.nan)
    y = features["rug_flag"].astype(int)
    return x, y, model_columns


def train_and_evaluate_models(features: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x, y, model_columns = prepare_model_frame(features)
    if y.nunique() < 2:
        warnings.warn("Only one label class is present. Model training is skipped.")
        predictions = features[["mint", "rug_flag"]].copy()
        predictions["split"] = "all"
        predictions["model"] = "not_trained_one_class"
        predictions["predicted_rug_flag"] = y.iloc[0] if len(y) else 0
        predictions["rug_probability"] = float(y.iloc[0]) if len(y) else 0.0
        metrics = pd.DataFrame(
            [{"model": "not_trained_one_class", "status": "skipped", "feature_count": len(model_columns)}]
        )
        return predictions, metrics, pd.DataFrame()

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(
        x,
        y,
        features.index,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        positive_count = int(y_train.sum())
        negative_count = int((y_train == 0).sum())
        scale_pos_weight = negative_count / positive_count if positive_count else 1.0
        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    except ImportError:
        warnings.warn(
            "xgboost is not installed. Install it to train the requested XGBoost model: pip install xgboost"
        )

    prediction_frames = []
    metric_rows = []
    importance_frames = []
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_test)[:, 1]
        else:
            y_score = y_pred.astype(float)

        metric_rows.append(
            {
                "model": model_name,
                "status": "trained",
                "feature_count": len(model_columns),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_score) if y_test.nunique() == 2 else np.nan,
            }
        )

        prediction_frame = features.loc[test_index, ["mint", "rug_flag"]].copy()
        prediction_frame["split"] = "test"
        prediction_frame["model"] = model_name
        prediction_frame["predicted_rug_flag"] = y_pred
        prediction_frame["rug_probability"] = y_score
        prediction_frames.append(prediction_frame)

        fitted_model = model.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importance = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            importance = np.abs(fitted_model.coef_[0])
        else:
            importance = None

        if importance is not None:
            importance_frame = pd.DataFrame(
                {
                    "model": model_name,
                    "feature": model_columns,
                    "importance": importance,
                }
            ).sort_values("importance", ascending=False)
            importance_frames.append(importance_frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["status", "f1"], ascending=[True, False])
    feature_importance = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    return predictions, metrics, feature_importance


# ---------------------------------------------------------------------------
# PART 12 - GNN PREPARATION ONLY
# ---------------------------------------------------------------------------

def prepare_gnn_outputs(events: pd.DataFrame, graph_edges: pd.DataFrame, token_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wallet_nodes = pd.DataFrame({"node_id": events["trader_id"].dropna().astype(str).unique(), "node_type": "wallet"})
    token_nodes = pd.DataFrame({"node_id": events["mint"].dropna().astype(str).unique(), "node_type": "token"})
    nodes = pd.concat([wallet_nodes, token_nodes], ignore_index=True).drop_duplicates("node_id")

    gnn_edges = graph_edges.rename(columns={"weight": "edge_weight", "side": "edge_type"}).copy()
    token_labels = token_features[["mint", "rug_flag"]].rename(columns={"mint": "node_id"}).copy()
    return nodes, gnn_edges, token_labels


# ---------------------------------------------------------------------------
# PART 11 - VALIDATION
# ---------------------------------------------------------------------------

def run_validations(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(events.duplicated(["tx_id", "mint", "side"]).sum())
    side_counts = events["side"].value_counts().to_dict()
    trade_volume = float(trades["amount_usd"].sum())
    event_volume = float(events["amount_usd"].sum())
    expected_event_volume = 2.0 * trade_volume
    volume_difference = event_volume - expected_event_volume

    validation_rows = [
        {"check": "trade_rows", "value": len(trades), "status": "info"},
        {"check": "event_rows", "value": len(events), "status": "info"},
        {"check": "missing_values_events", "value": int(events.isna().sum().sum()), "status": "warn" if events.isna().sum().sum() else "pass"},
        {"check": "duplicate_tx_id_mint_side", "value": duplicate_count, "status": "warn" if duplicate_count else "pass"},
        {"check": "buy_events", "value": int(side_counts.get("buy", 0)), "status": "info"},
        {"check": "sell_events", "value": int(side_counts.get("sell", 0)), "status": "info"},
        {"check": "trade_volume_usd", "value": trade_volume, "status": "info"},
        {"check": "event_volume_usd", "value": event_volume, "status": "info"},
        {
            "check": "volume_consistency_event_minus_2x_trade",
            "value": volume_difference,
            "status": "pass" if abs(volume_difference) < 1e-6 * max(expected_event_volume, 1.0) else "warn",
        },
    ]
    return pd.DataFrame(validation_rows)


# ---------------------------------------------------------------------------
# PART 13 - VISUALIZATION
# ---------------------------------------------------------------------------

def create_visualizations(
    output_dir: Path,
    token_features: pd.DataFrame,
    events: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is not installed. Install it to create charts: pip install matplotlib")
        return []

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[str] = []

    def save_current_figure(filename: str) -> None:
        path = viz_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        created_files.append(str(path))

    # Label distribution shows how imbalanced the heuristic rug labels are.
    label_counts = token_features["rug_flag"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["non_rug", "rug"], [label_counts.get(0, 0), label_counts.get(1, 0)], color=["#4C78A8", "#E45756"])
    plt.title("Heuristic Rug Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Token count")
    save_current_figure("label_distribution.png")

    # Side distribution validates that each trade produced one buy and one sell token-side event.
    side_counts = events["side"].value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(side_counts.index, side_counts.values, color=["#59A14F", "#F28E2B"])
    plt.title("Token-Side Event Distribution")
    plt.xlabel("Side")
    plt.ylabel("Event count")
    save_current_figure("side_distribution.png")

    # Log-scaled volume distribution makes heavy-tailed token volume easier to inspect.
    plt.figure(figsize=(7, 4))
    plt.hist(np.log1p(token_features["total_volume_usd"]), bins=50, color="#4C78A8", edgecolor="white")
    plt.title("Token Total Volume Distribution")
    plt.xlabel("log1p(total_volume_usd)")
    plt.ylabel("Token count")
    save_current_figure("total_volume_distribution.png")

    # Lifespan distribution helps verify whether short-lived tokens dominate the rug proxy.
    clipped_lifespan = token_features["lifespan_hours"].clip(lower=0, upper=token_features["lifespan_hours"].quantile(0.99))
    plt.figure(figsize=(7, 4))
    plt.hist(clipped_lifespan, bins=50, color="#72B7B2", edgecolor="white")
    plt.title("Token Lifespan Distribution")
    plt.xlabel("lifespan_hours, clipped at p99")
    plt.ylabel("Token count")
    save_current_figure("lifespan_distribution.png")

    # Concentration scatter visualizes the heuristic relationship between wallet dominance and sell pressure.
    sample = token_features.sample(n=min(6000, len(token_features)), random_state=42)
    colors = np.where(sample["rug_flag"] == 1, "#E45756", "#4C78A8")
    plt.figure(figsize=(7, 5))
    plt.scatter(
        sample["top1_wallet_volume_ratio"],
        np.log1p(sample["sell_buy_volume_ratio"]),
        c=colors,
        alpha=0.45,
        s=14,
    )
    plt.title("Wallet Concentration vs Sell/Buy Pressure")
    plt.xlabel("top1_wallet_volume_ratio")
    plt.ylabel("log1p(sell_buy_volume_ratio)")
    save_current_figure("concentration_vs_sell_pressure.png")

    # Model comparison chart summarizes the requested evaluation metrics in one view.
    trained_metrics = metrics[metrics["status"].eq("trained")].copy()
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    if not trained_metrics.empty:
        plot_metrics = trained_metrics.set_index("model")[metric_columns]
        ax = plot_metrics.plot(kind="bar", figsize=(9, 5), ylim=(0, 1.05))
        ax.set_title("Model Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.legend(loc="lower right")
        save_current_figure("model_comparison.png")

    # Confusion matrices show false positives and false negatives for each trained model.
    for model_name, model_predictions in predictions.groupby("model"):
        if model_name == "not_trained_one_class":
            continue
        matrix = confusion_matrix(
            model_predictions["rug_flag"],
            model_predictions["predicted_rug_flag"],
            labels=[0, 1],
        )
        plt.figure(figsize=(5, 4))
        plt.imshow(matrix, cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.xticks([0, 1], ["non_rug", "rug"])
        plt.yticks([0, 1], ["non_rug", "rug"])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
        save_current_figure(f"confusion_matrix_{model_name}.png")

    # ROC curves compare ranking quality across models using predicted probabilities.
    if predictions["rug_flag"].nunique() == 2:
        plt.figure(figsize=(6, 5))
        for model_name, model_predictions in predictions.groupby("model"):
            if model_name == "not_trained_one_class" or model_predictions["rug_flag"].nunique() < 2:
                continue
            fpr, tpr, _thresholds = roc_curve(model_predictions["rug_flag"], model_predictions["rug_probability"])
            auc_value = roc_auc_score(model_predictions["rug_flag"], model_predictions["rug_probability"])
            plt.plot(fpr, tpr, label=f"{model_name} AUC={auc_value:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="random")
        plt.title("ROC Curves")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend(loc="lower right")
        save_current_figure("roc_curves.png")

    # Feature importance highlights which engineered features each model relies on most.
    if not feature_importance.empty:
        feature_importance.to_csv(output_dir / "feature_importance_2025.csv", index=False)
        for model_name, model_importance in feature_importance.groupby("model"):
            top_features = model_importance.sort_values("importance", ascending=False).head(15).iloc[::-1]
            plt.figure(figsize=(8, 6))
            plt.barh(top_features["feature"], top_features["importance"], color="#F28E2B")
            plt.title(f"Top Feature Importance - {model_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            save_current_figure(f"feature_importance_{model_name}.png")

    return created_files


# ---------------------------------------------------------------------------
# PART 10 - OUTPUT FILES
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    token_features: pd.DataFrame,
    graph_edges: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    validation_report: pd.DataFrame,
    gnn_nodes: pd.DataFrame,
    gnn_edges: pd.DataFrame,
    gnn_token_labels: pd.DataFrame,
    feature_importance: pd.DataFrame,
    visualization_files: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    token_features_to_save = token_features.copy()
    for column in ["first_activity_time", "last_activity_time"]:
        token_features_to_save[column] = token_features_to_save[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    token_features_to_save.to_csv(output_dir / "token_features_2025.csv", index=False)
    graph_edges.to_csv(output_dir / "graph_edges_2025.csv", index=False)
    predictions.to_csv(output_dir / "model_predictions.csv", index=False)
    metrics.to_csv(output_dir / "model_comparison.csv", index=False)
    metrics.to_csv(output_dir / "model_metrics_2025.csv", index=False)
    validation_report.to_csv(output_dir / "validation_report_2025.csv", index=False)
    if not feature_importance.empty:
        feature_importance.to_csv(output_dir / "feature_importance_2025.csv", index=False)
    gnn_nodes.to_csv(output_dir / "gnn_nodes.csv", index=False)
    gnn_edges.to_csv(output_dir / "gnn_edges.csv", index=False)
    gnn_token_labels.to_csv(output_dir / "gnn_token_labels.csv", index=False)
    pd.DataFrame({"visualization_file": visualization_files}).to_csv(
        output_dir / "visualization_manifest_2025.csv", index=False
    )


def run_pipeline(input_dir: Path, output_dir: Path, pattern: str, random_state: int) -> dict[str, int | str]:
    input_files = find_input_files(input_dir, pattern)
    trades = load_trade_data(input_files)
    events = create_token_side_events(trades)

    token_features = engineer_token_features(events)
    concentration_features, _wallet_token = compute_wallet_concentration(events)
    graph_edges = build_graph_edges(events)
    graph_features = compute_graph_features(graph_edges, token_features["mint"])

    token_features = (
        token_features.merge(concentration_features, on="mint", how="left")
        .merge(graph_features, on="mint", how="left")
        .fillna(
            {
                "top1_wallet_volume_ratio": 0,
                "top5_wallet_volume_ratio": 0,
                "top1_wallet_swap_ratio": 0,
                "top5_wallet_swap_ratio": 0,
                "graph_degree": 0,
                "graph_weighted_degree": 0,
                "connected_wallets": 0,
                "graph_clustering_coefficient": 0,
                "graph_degree_centrality": 0,
            }
        )
    )
    token_features = create_heuristic_labels(token_features)

    predictions, metrics, feature_importance = train_and_evaluate_models(token_features, random_state=random_state)
    validation_report = run_validations(trades, events)

    # This milestone prepares graph data for future Graph Neural Network training.
    # Full GNN training is deferred because reliable ground-truth labels and
    # historical graph-aligned training data are required.
    gnn_nodes, gnn_edges, gnn_token_labels = prepare_gnn_outputs(events, graph_edges, token_features)
    visualization_files = create_visualizations(
        output_dir=output_dir,
        token_features=token_features,
        events=events,
        predictions=predictions,
        metrics=metrics,
        feature_importance=feature_importance,
    )

    save_outputs(
        output_dir=output_dir,
        token_features=token_features,
        graph_edges=graph_edges,
        predictions=predictions,
        metrics=metrics,
        validation_report=validation_report,
        gnn_nodes=gnn_nodes,
        gnn_edges=gnn_edges,
        gnn_token_labels=gnn_token_labels,
        feature_importance=feature_importance,
        visualization_files=visualization_files,
    )

    summary = {
        "input_files": len(input_files),
        "trade_rows": len(trades),
        "event_rows": len(events),
        "token_count": len(token_features),
        "rug_flag_count": int(token_features["rug_flag"].sum()),
        "visualization_count": len(visualization_files),
        "output_dir": str(output_dir),
    }
    with (output_dir / "pipeline_summary_2025.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solana DEX rug-pull detection pipeline for 2025 token activity.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "results")
    parser.add_argument("--pattern", default="2025_*.csv")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        random_state=args.random_state,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
