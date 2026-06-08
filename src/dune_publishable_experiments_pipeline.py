from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


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
from dune_paper_roadmap_pipeline import engineer_window_features  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"
SILVER_LABELS_PATH = OUTPUT_DIR / "silver_labeled_token_features.csv"
WINDOW_FEATURES_PATH = OUTPUT_DIR / "early_window_silver_features.csv"
INDUCTIVE_GRAPH_STATS_PATH = OUTPUT_DIR / "inductive_graph_stats.csv"
GRAPH_EMBEDDINGS_PATH = OUTPUT_DIR / "graphsage_or_graph_embeddings.csv"
GRAPHSAGE_EMBEDDINGS_PATH = OUTPUT_DIR / "graphsage_inductive_embeddings.csv"
GRAPHSAGE_METRICS_PATH = OUTPUT_DIR / "graphsage_inductive_metrics.csv"
GRAPHSAGE_TRAINING_LOG_PATH = OUTPUT_DIR / "graphsage_training_log.csv"
XGBOOST_ABLATION_PATH = OUTPUT_DIR / "xgboost_window_ablation.csv"
BASELINE_METRICS_PATH = OUTPUT_DIR / "simple_baseline_metrics.csv"
MANUAL_VALIDATION_CANDIDATES_PATH = OUTPUT_DIR / "manual_validation_candidates_for_rugcheck_solscan.csv"
CHECKLIST_PATH = OUTPUT_DIR / "checklist_status.csv"
SUMMARY_PATH = OUTPUT_DIR / "publishable_experiments_summary.json"

WINDOW_HOURS = [1, 6, 24]

TOKEN_ONLY_COLUMNS = [
    "activity_count",
    "buy_count",
    "sell_count",
    "total_volume",
    "buy_volume_usd",
    "sell_volume_usd",
    "imbalance_ratio",
    "sell_pressure",
    "lifespan_hours",
    "active_days",
    "log_total_volume",
    "log_activity_count",
]

GRAPH_EMBEDDING_COLUMNS = [
    "graph_emb_degree",
    "graph_emb_weighted_degree",
    "graph_emb_pagerank",
    "graph_emb_wallet_count",
    "graph_emb_max_wallet_volume",
    "graph_emb_mean_wallet_volume",
    "graph_emb_concentration",
    "graph_emb_log_weighted_degree",
]

GRAPHSAGE_EMBEDDING_COLUMNS = [f"graphsage_emb_{index:02d}" for index in range(16)]

NODE_FEATURE_COLUMNS = [
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
    "wallet_total_volume",
    "wallet_trade_count",
    "wallet_unique_tokens",
    "wallet_buy_count",
    "wallet_sell_count",
]


def log(message: str) -> None:
    print(f"[dune-publishable] {message}", flush=True)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_binary(y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_xgboost_or_fallback():
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


def predict_scores(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    decision = model.decision_function(frame)
    return (decision - decision.min()) / (decision.max() - decision.min() + 1e-12)


def prepare_silver_labels() -> tuple[pd.DataFrame, dict[str, float]]:
    features, _ = load_or_build_feature_data()
    labeled, thresholds = add_heuristic_rug_labels(features)
    labeled = labeled.rename(columns={"heuristic_rug_label": "silver_label", "heuristic_rug_score": "silver_label_score"})
    labeled["weak_label"] = labeled["silver_label"]
    labeled["label_source"] = "silver_heuristic_from_dune_behavior"
    labeled.to_csv(SILVER_LABELS_PATH, index=False)
    return labeled, thresholds


def build_early_windows(events: pd.DataFrame, silver_labels: pd.DataFrame) -> pd.DataFrame:
    label_view = silver_labels.rename(columns={"silver_label": "heuristic_rug_label", "silver_label_score": "heuristic_rug_score"})
    frames = []
    for hours in WINDOW_HOURS:
        frame = engineer_window_features(events, label_view, hours)
        frame = frame.rename(columns={"heuristic_rug_label": "silver_label", "heuristic_rug_score": "silver_label_score"})
        frame["weak_label"] = frame["silver_label"]
        frames.append(frame)
    windows = pd.concat(frames, ignore_index=True)
    windows.to_csv(WINDOW_FEATURES_PATH, index=False)
    return windows


def graph_embeddings_for_window(events: pd.DataFrame, window_features: pd.DataFrame, window_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_seen = events.groupby(["year", "token_address"], as_index=False).agg(first_seen_window=("block_time", "min"))
    scoped = events.merge(first_seen, on=["year", "token_address"], how="inner")
    scoped = scoped[scoped["block_time"].le(scoped["first_seen_window"] + pd.to_timedelta(window_hours, unit="h"))].copy()

    embedding_frames = []
    stats_rows = []
    for year, year_events in scoped.groupby("year"):
        graph = nx.Graph()
        edge_table = (
            year_events.groupby(["token_address", "trader_id"], as_index=False)
            .agg(total_volume=("amount_usd", "sum"), total_count=("tx_id", "count"))
        )
        for row in edge_table.itertuples(index=False):
            token_node = f"token:{row.token_address}"
            wallet_node = f"wallet:{row.trader_id}"
            graph.add_node(token_node, node_type="token")
            graph.add_node(wallet_node, node_type="wallet")
            graph.add_edge(token_node, wallet_node, weight=float(row.total_volume), count=float(row.total_count))

        degree = dict(graph.degree())
        weighted_degree = dict(graph.degree(weight="weight"))
        pagerank = nx.pagerank(graph, weight="weight", max_iter=100, tol=1e-6) if graph.number_of_edges() else {}

        token_wallet = edge_table.groupby("token_address", as_index=False).agg(
            graph_emb_wallet_count=("trader_id", "nunique"),
            graph_emb_max_wallet_volume=("total_volume", "max"),
            graph_emb_mean_wallet_volume=("total_volume", "mean"),
            graph_emb_weighted_degree=("total_volume", "sum"),
        )
        token_wallet["node_id"] = "token:" + token_wallet["token_address"].astype(str)
        token_wallet["graph_emb_degree"] = token_wallet["node_id"].map(degree).fillna(0.0)
        token_wallet["graph_emb_pagerank"] = token_wallet["node_id"].map(pagerank).fillna(0.0)
        token_wallet["graph_emb_concentration"] = safe_divide(token_wallet["graph_emb_max_wallet_volume"], token_wallet["graph_emb_weighted_degree"])
        token_wallet["graph_emb_log_weighted_degree"] = np.log1p(token_wallet["graph_emb_weighted_degree"].clip(lower=0.0))
        token_wallet["year"] = int(year)
        token_wallet["window_hours"] = int(window_hours)
        embedding_frames.append(token_wallet[["year", "window_hours", "token_address", *GRAPH_EMBEDDING_COLUMNS]])

        stats_rows.append(
            {
                "year": int(year),
                "window_hours": int(window_hours),
                "graph_scope": "inductive_year_specific",
                "token_nodes": int((pd.Series(dict(graph.nodes(data="node_type"))) == "token").sum()) if graph.number_of_nodes() else 0,
                "wallet_nodes": int((pd.Series(dict(graph.nodes(data="node_type"))) == "wallet").sum()) if graph.number_of_nodes() else 0,
                "edges": int(graph.number_of_edges()),
            }
        )

    embeddings = pd.concat(embedding_frames, ignore_index=True)
    stats = pd.DataFrame(stats_rows)
    return embeddings, stats


def build_graph_embeddings(events: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    status = "graphsage_not_run_torch_unavailable; used inductive NetworkX graph embeddings"
    try:
        import torch  # noqa: F401

        status = "fallback_graph_embeddings_also_created; torch_available"
    except Exception as exc:
        status = f"graphsage_not_run_torch_unavailable: {exc}; used inductive NetworkX graph embeddings"

    embedding_frames = []
    stats_frames = []
    for hours in WINDOW_HOURS:
        embeddings, stats = graph_embeddings_for_window(events, windows[windows["window_hours"].eq(hours)], hours)
        embedding_frames.append(embeddings)
        stats_frames.append(stats)
    all_embeddings = pd.concat(embedding_frames, ignore_index=True)
    all_stats = pd.concat(stats_frames, ignore_index=True)
    all_embeddings.to_csv(GRAPH_EMBEDDINGS_PATH, index=False)
    all_stats.to_csv(INDUCTIVE_GRAPH_STATS_PATH, index=False)
    return all_embeddings, all_stats, status


def build_graphsage_graph_frame(events: pd.DataFrame, windows: pd.DataFrame, year: int, window_hours: int) -> pd.DataFrame:
    scoped_features = windows[(windows["year"].eq(year)) & (windows["window_hours"].eq(window_hours))].copy()
    scoped_features["node_id"] = "token:" + scoped_features["token_address"].astype(str)
    for column in NODE_FEATURE_COLUMNS:
        if column not in scoped_features:
            scoped_features[column] = 0.0

    first_seen = events.groupby(["year", "token_address"], as_index=False).agg(first_seen_window=("block_time", "min"))
    scoped_events = events[events["year"].eq(year)].merge(first_seen[first_seen["year"].eq(year)], on=["year", "token_address"], how="inner")
    scoped_events = scoped_events[scoped_events["block_time"].le(scoped_events["first_seen_window"] + pd.to_timedelta(window_hours, unit="h"))].copy()

    wallet_features = (
        scoped_events.groupby("trader_id", as_index=False)
        .agg(
            wallet_total_volume=("amount_usd", "sum"),
            wallet_trade_count=("tx_id", "count"),
            wallet_unique_tokens=("token_address", "nunique"),
            wallet_buy_count=("side", lambda value: int((value == "buy").sum())),
            wallet_sell_count=("side", lambda value: int((value == "sell").sum())),
        )
        .rename(columns={"trader_id": "wallet_address"})
    )
    wallet_features["node_id"] = "wallet:" + wallet_features["wallet_address"].astype(str)
    wallet_features["token_address"] = ""
    wallet_features["silver_label"] = np.nan
    wallet_features["silver_label_score"] = np.nan
    wallet_features["weak_label"] = np.nan
    wallet_features["window_hours"] = window_hours
    wallet_features["year"] = year
    for column in NODE_FEATURE_COLUMNS:
        if column not in wallet_features:
            wallet_features[column] = 0.0

    token_nodes = scoped_features[["year", "window_hours", "token_address", "node_id", "silver_label", "silver_label_score", "weak_label", *NODE_FEATURE_COLUMNS]].copy()
    wallet_nodes = wallet_features[["year", "window_hours", "token_address", "node_id", "silver_label", "silver_label_score", "weak_label", *NODE_FEATURE_COLUMNS]].copy()
    return pd.concat([token_nodes, wallet_nodes], ignore_index=True)


def build_graphsage_tensors(events: pd.DataFrame, windows: pd.DataFrame, year: int, window_hours: int, deps: dict):
    torch = deps["torch"]
    nodes = build_graphsage_graph_frame(events, windows, year, window_hours).reset_index(drop=True)
    node_lookup = {node_id: index for index, node_id in enumerate(nodes["node_id"].astype(str))}

    first_seen = events.groupby(["year", "token_address"], as_index=False).agg(first_seen_window=("block_time", "min"))
    scoped_events = events[events["year"].eq(year)].merge(first_seen[first_seen["year"].eq(year)], on=["year", "token_address"], how="inner")
    scoped_events = scoped_events[scoped_events["block_time"].le(scoped_events["first_seen_window"] + pd.to_timedelta(window_hours, unit="h"))].copy()
    edge_table = scoped_events[["trader_id", "token_address"]].drop_duplicates()
    src = ("wallet:" + edge_table["trader_id"].astype(str)).map(node_lookup)
    dst = ("token:" + edge_table["token_address"].astype(str)).map(node_lookup)
    valid = src.notna() & dst.notna()
    edge_pairs = np.vstack([src[valid].astype(int).to_numpy(), dst[valid].astype(int).to_numpy()])
    edge_index = torch.tensor(np.hstack([edge_pairs, edge_pairs[[1, 0], :]]), dtype=torch.long)

    x_frame = nodes[NODE_FEATURE_COLUMNS].copy()
    for column in NODE_FEATURE_COLUMNS:
        x_frame[column] = pd.to_numeric(x_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_frame[column] = np.log1p(x_frame[column].clip(lower=0.0))
    x_np = x_frame.to_numpy(dtype=np.float32)
    return nodes, edge_index, x_np


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def try_run_graphsage(events: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame, str]:
    log("Attempting true inductive GraphSAGE training")
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch_geometric.nn import SAGEConv
    except Exception as exc:
        metrics = pd.DataFrame([{"status": f"graphsage_not_run_dependency_error: {exc}"}])
        metrics.to_csv(GRAPHSAGE_METRICS_PATH, index=False)
        return None, metrics, f"graphsage_not_run_dependency_error: {exc}"

    torch.manual_seed(42)
    deps = {"torch": torch}
    metric_rows = []
    log_rows = []
    embedding_frames = []

    class GraphSAGEClassifier(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 32, embed_dim: int = 16):
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, embed_dim)
            self.classifier = nn.Linear(embed_dim, 1)

        def forward(self, x_tensor, edge_idx):
            hidden = F.relu(self.conv1(x_tensor, edge_idx))
            hidden = F.dropout(hidden, p=0.25, training=self.training)
            embedding = F.relu(self.conv2(hidden, edge_idx))
            return self.classifier(embedding).squeeze(-1), embedding

    for window_hours in WINDOW_HOURS:
        log(f"Training GraphSAGE for {window_hours}h window")
        train_nodes, train_edge_index, train_x_np = build_graphsage_tensors(events, windows, 2024, window_hours, deps)
        test_nodes, test_edge_index, test_x_np = build_graphsage_tensors(events, windows, 2025, window_hours, deps)
        train_x_np, test_x_np = standardize_train_test(train_x_np, test_x_np)
        train_x = torch.tensor(train_x_np, dtype=torch.float32)
        test_x = torch.tensor(test_x_np, dtype=torch.float32)
        y = torch.tensor(train_nodes["silver_label"].fillna(0).to_numpy(dtype=np.float32), dtype=torch.float32)
        train_mask = train_nodes["silver_label"].notna().to_numpy()
        train_indices = np.flatnonzero(train_mask)
        if len(train_indices) < 20 or train_nodes.loc[train_mask, "silver_label"].nunique() < 2:
            metric_rows.append({"window_hours": window_hours, "status": "skipped_insufficient_labels"})
            continue

        model = GraphSAGEClassifier(train_x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        positives = float(y[train_indices].sum().item())
        negatives = float(len(train_indices) - positives)
        pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
        index_tensor = torch.tensor(train_indices, dtype=torch.long)

        for epoch in range(1, 41):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(train_x, train_edge_index)
            loss = F.binary_cross_entropy_with_logits(logits[index_tensor], y[index_tensor], pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            if epoch in {1, 10, 20, 30, 40}:
                log_rows.append({"window_hours": window_hours, "epoch": epoch, "train_loss": float(loss.item()), "positive_train": positives, "negative_train": negatives})

        model.eval()
        with torch.no_grad():
            train_logits, train_embedding = model(train_x, train_edge_index)
            test_logits, test_embedding = model(test_x, test_edge_index)
        train_scores = torch.sigmoid(train_logits).cpu().numpy()
        test_scores = torch.sigmoid(test_logits).cpu().numpy()

        for year, nodes, scores, embeddings in [
            (2024, train_nodes, train_scores, train_embedding.cpu().numpy()),
            (2025, test_nodes, test_scores, test_embedding.cpu().numpy()),
        ]:
            token_mask = nodes["token_address"].astype(str).ne("")
            frame = nodes.loc[token_mask, ["year", "window_hours", "token_address", "silver_label"]].copy()
            frame["graphsage_score"] = scores[token_mask.to_numpy()]
            emb = pd.DataFrame(embeddings[token_mask.to_numpy(), :], columns=GRAPHSAGE_EMBEDDING_COLUMNS)
            embedding_frames.append(pd.concat([frame.reset_index(drop=True), emb], axis=1))

        test_token_mask = test_nodes["token_address"].astype(str).ne("") & test_nodes["silver_label"].notna()
        metrics = evaluate_binary(test_nodes.loc[test_token_mask, "silver_label"], test_scores[test_token_mask.to_numpy()])
        metrics.update(
            {
                "window_hours": window_hours,
                "model": "graphsage",
                "train_year": 2024,
                "test_year": 2025,
                "status": "completed",
                "train_token_labels": int(len(train_indices)),
                "train_positive_labels": int(positives),
            }
        )
        metric_rows.append(metrics)

    embeddings_frame = pd.concat(embedding_frames, ignore_index=True) if embedding_frames else pd.DataFrame()
    metrics_frame = pd.DataFrame(metric_rows)
    embeddings_frame.to_csv(GRAPHSAGE_EMBEDDINGS_PATH, index=False)
    metrics_frame.to_csv(GRAPHSAGE_METRICS_PATH, index=False)
    pd.DataFrame(log_rows).to_csv(GRAPHSAGE_TRAINING_LOG_PATH, index=False)
    status = "completed" if not metrics_frame.empty and metrics_frame["status"].eq("completed").any() else "graphsage_not_completed"
    return embeddings_frame, metrics_frame, status


def run_xgboost_window_ablation(windows: pd.DataFrame, graph_embeddings: pd.DataFrame, embedding_columns: list[str], embedding_label: str) -> tuple[pd.DataFrame, str]:
    log("Running XGBoost/fallback token-only, graph-only, and combined ablations")
    merged = windows.merge(graph_embeddings, on=["year", "window_hours", "token_address"], how="left")
    for column in embedding_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    rows = []
    model_statuses = set()
    for hours, frame in merged.groupby("window_hours"):
        frame = clean_model_frame(frame)
        train = frame[frame["year"].eq(2024)].copy()
        test = frame[frame["year"].eq(2025)].copy()
        experiments = {
            "token_only": [column for column in TOKEN_ONLY_COLUMNS if column in frame.columns],
            embedding_label: embedding_columns,
            "combined": [column for column in TOKEN_ONLY_COLUMNS if column in frame.columns] + embedding_columns,
        }
        for experiment_name, columns in experiments.items():
            model, model_name, model_status = make_xgboost_or_fallback()
            model_statuses.add(model_status)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train[columns], train["silver_label"])
            scores = predict_scores(model, test[columns])
            metric = evaluate_binary(test["silver_label"], scores)
            metric.update(
                {
                    "window_hours": int(hours),
                    "experiment": experiment_name,
                    "model": model_name,
                    "model_status": model_status,
                    "train_year": 2024,
                    "test_year": 2025,
                    "feature_count": len(columns),
                }
            )
            rows.append(metric)
    metrics = pd.DataFrame(rows).sort_values(["window_hours", "experiment"])
    metrics.to_csv(XGBOOST_ABLATION_PATH, index=False)
    return metrics, "; ".join(sorted(model_statuses))


def run_simple_baselines(windows: pd.DataFrame) -> pd.DataFrame:
    log("Running simple rule baselines")
    rows = []
    rules = {
        "sell_pressure_ge_0_90": lambda train, test: (train["sell_pressure"].quantile(0.90), test["sell_pressure"]),
        "concentration_ge_0_90": lambda train, test: (train["entity_concentration_ratio"].quantile(0.90), test["entity_concentration_ratio"]),
        "imbalance_ge_p95_train": lambda train, test: (train["imbalance_ratio"].quantile(0.95), test["imbalance_ratio"]),
        "silver_score_ge_p95_train": lambda train, test: (train["silver_label_score"].quantile(0.95), test["silver_label_score"]),
    }
    for hours, frame in windows.groupby("window_hours"):
        train = frame[frame["year"].eq(2024)].copy()
        test = frame[frame["year"].eq(2025)].copy()
        for rule_name, rule_fn in rules.items():
            threshold, scores = rule_fn(train, test)
            metric = evaluate_binary(test["silver_label"], pd.to_numeric(scores, errors="coerce").fillna(0.0).to_numpy(), threshold=float(threshold))
            metric.update({"window_hours": int(hours), "baseline_rule": rule_name, "threshold_from_2024": float(threshold), "train_year": 2024, "test_year": 2025})
            rows.append(metric)
    metrics = pd.DataFrame(rows).sort_values(["window_hours", "f1"], ascending=[True, False])
    metrics.to_csv(BASELINE_METRICS_PATH, index=False)
    return metrics


def create_manual_validation_candidates(silver_labels: pd.DataFrame, ablation_metrics: pd.DataFrame | None = None) -> None:
    candidates = (
        silver_labels[silver_labels["year"].eq(2025)]
        .sort_values("silver_label_score", ascending=False)
        .head(100)
        [["year", "token_address", "silver_label", "silver_label_score", "total_volume", "sell_pressure", "entity_concentration_ratio", "lifespan_hours", "unique_wallets"]]
        .copy()
    )
    candidates["rugcheck_status"] = ""
    candidates["solscan_notes"] = ""
    candidates["manual_validation_label"] = ""
    candidates["manual_validator"] = ""
    candidates["manual_validation_date"] = ""
    candidates["manual_validation_status"] = "not_verified"
    candidates.to_csv(MANUAL_VALIDATION_CANDIDATES_PATH, index=False)


def write_checklist(graphsage_status: str, xgboost_status: str) -> pd.DataFrame:
    rows = [
        ("1.1", "Strict temporal split: 2024 train, 2025 test, no random train_test_split in publishable pipeline", "done", "All publishable metrics train on year==2024 and test on year==2025."),
        ("1.2", "Early-warning windows: 1h, 6h, 24h snapshots", "done", str(WINDOW_FEATURES_PATH)),
        ("1.3", "Rename label to weak_label/silver_label", "done", str(SILVER_LABELS_PATH)),
        ("1.3b", "Manual RugCheck/Solscan benchmark 50-100 verified tokens", "not_done", "Created 100-token candidate file, but no manual external verification was performed."),
        ("2.1", "Inductive wallet-token bipartite graph: separate 2024 train and 2025 test graphs", "done", str(INDUCTIVE_GRAPH_STATS_PATH)),
        ("2.2", "Train GraphSAGE and produce inductive embeddings", "done" if graphsage_status == "completed" else "not_done", graphsage_status),
        ("2.3", "XGBoost ablation for token-only, GraphSAGE/graph-only, combined per window", "done", f"{xgboost_status}; {XGBOOST_ABLATION_PATH}"),
        ("2.4", "Simple rule baselines", "done", str(BASELINE_METRICS_PATH)),
    ]
    checklist = pd.DataFrame(rows, columns=["task_id", "task", "status", "notes"])
    checklist.to_csv(CHECKLIST_PATH, index=False)
    return checklist


def main() -> None:
    ensure_output_dir()
    silver_labels, thresholds = prepare_silver_labels()
    swaps = load_raw_dune_swaps()
    events = create_token_events(swaps)
    windows = build_early_windows(events, silver_labels)
    graph_embeddings, graph_stats, graph_fallback_status = build_graph_embeddings(events, windows)
    graphsage_embeddings, graphsage_metrics, graphsage_status = try_run_graphsage(events, windows)
    if graphsage_embeddings is not None and not graphsage_embeddings.empty and graphsage_status == "completed":
        ablation_embeddings = graphsage_embeddings.drop(columns=["silver_label", "graphsage_score"], errors="ignore")
        ablation_embedding_columns = GRAPHSAGE_EMBEDDING_COLUMNS
        ablation_embedding_label = "graphsage_only"
    else:
        ablation_embeddings = graph_embeddings
        ablation_embedding_columns = GRAPH_EMBEDDING_COLUMNS
        ablation_embedding_label = "graph_fallback_only"
        graphsage_status = f"{graphsage_status}; fallback_used_for_ablation={graph_fallback_status}"
    ablation_metrics, xgboost_status = run_xgboost_window_ablation(windows, ablation_embeddings, ablation_embedding_columns, ablation_embedding_label)
    baseline_metrics = run_simple_baselines(windows)
    create_manual_validation_candidates(silver_labels, ablation_metrics)
    checklist = write_checklist(graphsage_status, xgboost_status)

    summary = {
        "scope": "publishable experiments with strict temporal split and early-warning windows",
        "thresholds_from_2024": thresholds,
        "label_columns": ["weak_label", "silver_label"],
        "train_year": 2024,
        "test_year": 2025,
        "graphsage_status": graphsage_status,
        "graph_fallback_status": graph_fallback_status,
        "xgboost_status": xgboost_status,
        "best_graphsage_rows": graphsage_metrics.sort_values(["f1", "average_precision"], ascending=False).head(5).to_dict("records") if not graphsage_metrics.empty and "f1" in graphsage_metrics else [],
        "best_ablation_rows": ablation_metrics.sort_values(["f1", "average_precision"], ascending=False).head(5).to_dict("records"),
        "best_baseline_rows": baseline_metrics.sort_values(["f1", "average_precision"], ascending=False).head(5).to_dict("records"),
        "checklist": checklist.to_dict("records"),
        "outputs": {
            "silver_labels": str(SILVER_LABELS_PATH),
            "early_windows": str(WINDOW_FEATURES_PATH),
            "graph_stats": str(INDUCTIVE_GRAPH_STATS_PATH),
            "graph_fallback_embeddings": str(GRAPH_EMBEDDINGS_PATH),
            "graphsage_embeddings": str(GRAPHSAGE_EMBEDDINGS_PATH),
            "graphsage_metrics": str(GRAPHSAGE_METRICS_PATH),
            "graphsage_training_log": str(GRAPHSAGE_TRAINING_LOG_PATH),
            "xgboost_ablation": str(XGBOOST_ABLATION_PATH),
            "baselines": str(BASELINE_METRICS_PATH),
            "manual_validation_candidates": str(MANUAL_VALIDATION_CANDIDATES_PATH),
            "checklist": str(CHECKLIST_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Publishable experiment checklist completed.")
    print(f"Checklist: {CHECKLIST_PATH}")
    print(f"XGBoost/ablation metrics: {XGBOOST_ABLATION_PATH}")
    print(f"Baselines: {BASELINE_METRICS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
