from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = PROJECT_ROOT / "data" / "gold" / "pyg_nodes_2024_2025.parquet"
EDGES_PATH = PROJECT_ROOT / "data" / "gold" / "pyg_edges_2024_2025.parquet"
TOKEN_FEATURES_PATH = PROJECT_ROOT / "data" / "gold" / "dune_token_features_2024_2025.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "gnn"
TRAINING_LOG_PATH = RESULTS_DIR / "graphsage_training_log.csv"
METRICS_PATH = RESULTS_DIR / "graphsage_metrics_2024.csv"
LABEL_NOTES_PATH = RESULTS_DIR / "graphsage_labeling_notes.txt"
TOP_RISKY_PATH = RESULTS_DIR / "graphsage_top_risky_2025.csv"
THRESHOLD_PATH = RESULTS_DIR / "graphsage_threshold_2024.csv"
LOSS_CURVE_PATH = RESULTS_DIR / "graphsage_loss_curve.png"
EMBEDDING_PLOT_PATH = RESULTS_DIR / "graphsage_embedding_umap.png"
RISK_SCORES_PATH = PROJECT_ROOT / "data" / "gold" / "gnn_risk_scores_2025.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "gold" / "graphsage_embeddings_2024_2025.parquet"

OPTIONAL_LABEL_PATHS = [
    PROJECT_ROOT / "data" / "results" / "cross_time" / "validation_predictions_2024.csv",
    PROJECT_ROOT / "data" / "results" / "cross_time" / "predictions_2025.csv",
    PROJECT_ROOT / "data" / "results" / "cross_time" / "aligned_historical_features.csv",
]

FEATURE_COLUMNS = [
    "activity_count",
    "total_volume",
    "buy_volume_usd",
    "sell_volume_usd",
    "imbalance_ratio",
    "unique_wallets",
    "lifespan_hours",
    "graph_degree",
    "connected_entities",
    "entity_concentration_ratio",
    "wallet_degree",
    "wallet_total_volume",
    "wallet_total_count",
    "wallet_unique_tokens",
]

LOG1P_FEATURE_COLUMNS = [
    "activity_count",
    "total_volume",
    "buy_volume_usd",
    "sell_volume_usd",
    "imbalance_ratio",
    "unique_wallets",
    "lifespan_hours",
    "graph_degree",
    "connected_entities",
    "wallet_degree",
    "wallet_total_volume",
    "wallet_total_count",
    "wallet_unique_tokens",
]


def log(message: str) -> None:
    print(f"[dune-graphsage] {message}", flush=True)


def require_dependencies():
    missing = []
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
    except ImportError:
        missing.append("torch")
        torch = F = nn = None

    try:
        from torch_geometric.nn import SAGEConv
    except ImportError:
        missing.append("torch_geometric")
        SAGEConv = None

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("pip install pandas pyarrow networkx matplotlib pyvis scikit-learn torch")
        sys.exit(1)

    if "torch_geometric" in missing:
        print("PyTorch Geometric is required for GraphSAGE.")
        print("Please install torch_geometric using the official installation guide matching your Python, PyTorch, CPU/CUDA, and operating system.")
        sys.exit(1)
    if "torch" in missing:
        print("PyTorch is required for this script.")
        print("pip install torch")
        sys.exit(1)

    return {
        "torch": torch,
        "F": F,
        "nn": nn,
        "SAGEConv": SAGEConv,
        "plt": plt,
        "PCA": PCA,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "train_test_split": train_test_split,
    }


def check_inputs() -> None:
    missing = [path for path in [NODES_PATH, EDGES_PATH, TOKEN_FEATURES_PATH] if not path.exists()]
    if missing:
        print("Missing lakehouse graph inputs:")
        for path in missing:
            print(f"- {path}")
        print("Please run:")
        print("python src/dune_spark_lakehouse_pipeline.py")
        sys.exit(1)


def clean_numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def load_external_2024_labels(token_features: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    label_frames = []
    for path in OPTIONAL_LABEL_PATHS:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            log(f"Skipping optional label source {path}: {exc}")
            continue
        mint_col = "mint" if "mint" in frame.columns else "token_address" if "token_address" in frame.columns else None
        label_col = "rug_label" if "rug_label" in frame.columns else "heuristic_rug_label" if "heuristic_rug_label" in frame.columns else None
        if mint_col and label_col and "year" in frame.columns:
            labels = (
                frame.loc[frame["year"].eq(2024), [mint_col, label_col]]
                .rename(columns={mint_col: "token_address", label_col: "label"})
                .dropna()
            )
            labels["label"] = labels["label"].astype(int)
            label_frames.append(labels)

    if not label_frames:
        return token_features, None

    labels = pd.concat(label_frames, ignore_index=True).groupby("token_address", as_index=False)["label"].max()
    merged = token_features.merge(labels, on="token_address", how="left")
    matched = int(merged.loc[merged["year"].eq(2024), "label"].notna().sum())
    if matched == 0:
        return token_features, None
    return merged, f"Matched {matched} 2024 token labels from existing cross-time outputs by token mint address."


def add_heuristic_labels(token_features: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    features = token_features.copy()
    train = features[features["year"].eq(2024)].copy()
    if train.empty:
        features["label"] = np.nan
        return features, "No 2024 tokens were available, so no heuristic training labels were created."

    thresholds = {
        "imbalance_ratio_p85": float(train["imbalance_ratio"].quantile(0.85)),
        "lifespan_hours_p35": float(train["lifespan_hours"].quantile(0.35)),
        "activity_count_p70": float(train["activity_count"].quantile(0.70)),
        "sell_count_p70": float(train["sell_count"].quantile(0.70)),
        "entity_concentration_ratio_p75": float(train["entity_concentration_ratio"].quantile(0.75)),
    }
    heuristic = (
        (features["year"].eq(2024))
        & (features["imbalance_ratio"] >= thresholds["imbalance_ratio_p85"])
        & (features["lifespan_hours"] <= thresholds["lifespan_hours_p35"])
        & ((features["activity_count"] >= thresholds["activity_count_p70"]) | (features["sell_count"] >= thresholds["sell_count_p70"]))
        & (features["entity_concentration_ratio"] >= thresholds["entity_concentration_ratio_p75"])
    )
    features["label"] = np.where(features["year"].eq(2024), heuristic.astype(int), np.nan)
    note = (
        "No reliable 2024 labels matched by token address. Created heuristic 2024 labels only for experimentation:\n"
        f"- imbalance_ratio >= {thresholds['imbalance_ratio_p85']:.6f}\n"
        f"- lifespan_hours <= {thresholds['lifespan_hours_p35']:.6f}\n"
        f"- activity_count >= {thresholds['activity_count_p70']:.6f} OR sell_count >= {thresholds['sell_count_p70']:.6f}\n"
        f"- entity_concentration_ratio >= {thresholds['entity_concentration_ratio_p75']:.6f}\n"
        "These labels are not ground truth and must only be described as heuristic risk agreement."
    )
    return features, note


def prepare_labels(token_features: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    labeled, source_note = load_external_2024_labels(token_features)
    if source_note is not None:
        mode = "supervised"
        note = source_note + "\n2025 outputs are risk scores only, not true predictive accuracy."
    else:
        labeled, note = add_heuristic_labels(token_features)
        mode = "heuristic supervised"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_NOTES_PATH.write_text(note, encoding="utf-8")
    return labeled, mode, note


def build_graph_tensors(nodes: pd.DataFrame, edges: pd.DataFrame, deps):
    torch = deps["torch"]
    nodes = nodes.reset_index(drop=True).copy()
    node_lookup = {node_id: index for index, node_id in enumerate(nodes["node_id"].astype(str))}
    src = edges["src_node_id"].astype(str).map(node_lookup)
    dst = edges["dst_node_id"].astype(str).map(node_lookup)
    valid = src.notna() & dst.notna()
    edge_pairs = np.vstack([src[valid].astype(int).to_numpy(), dst[valid].astype(int).to_numpy()])
    reverse_pairs = edge_pairs[[1, 0], :]
    edge_index = torch.tensor(np.hstack([edge_pairs, reverse_pairs]), dtype=torch.long)
    x = torch.tensor(nodes[FEATURE_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32)
    return nodes, edge_index, x


def log1p_scale_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    scaled = frame.copy()
    for column in LOG1P_FEATURE_COLUMNS:
        if column in scaled.columns:
            scaled[column] = np.log1p(pd.to_numeric(scaled[column], errors="coerce").fillna(0.0).clip(lower=0.0))
    return scaled


def standardize_features(x_np: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = x_np[train_indices]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0
    return (x_np - mean) / std, mean, std


def make_model(input_dim: int, hidden_dim: int, dropout: float, deps):
    torch = deps["torch"]
    nn = deps["nn"]
    F = deps["F"]
    SAGEConv = deps["SAGEConv"]

    class GraphSAGEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, 1)
            self.dropout = dropout

        def encode(self, features, edge_index):
            hidden = self.conv1(features, edge_index)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.conv2(hidden, edge_index)
            return hidden

        def forward(self, features, edge_index):
            embeddings = self.encode(features, edge_index)
            logits = self.classifier(embeddings).squeeze(-1)
            return logits, embeddings

    return GraphSAGEModel()


def safe_auc(y_true, y_score, roc_auc_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(y_true, y_score, threshold, deps) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    cm = deps["confusion_matrix"](y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(deps["accuracy_score"](y_true, y_pred)),
        "precision": float(deps["precision_score"](y_true, y_pred, zero_division=0)),
        "recall": float(deps["recall_score"](y_true, y_pred, zero_division=0)),
        "f1": float(deps["f1_score"](y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(y_true, y_score, deps["roc_auc_score"]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def tune_threshold(y_true, y_score, deps) -> tuple[float, dict[str, float]]:
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_metrics = compute_metrics(y_true, y_score, best_threshold, deps)
    best_key = (best_metrics["f1"], best_metrics["recall"], best_metrics["precision"])
    for threshold in candidates:
        metrics = compute_metrics(y_true, y_score, float(threshold), deps)
        key = (metrics["f1"], metrics["recall"], metrics["precision"])
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def class_pos_weight(labels: np.ndarray, train_idx: np.ndarray, deps):
    torch = deps["torch"]
    train_labels = labels[train_idx].astype(int)
    positive_count = int((train_labels == 1).sum())
    negative_count = int((train_labels == 0).sum())
    if positive_count == 0:
        return torch.tensor(1.0, dtype=torch.float32), positive_count, negative_count
    return torch.tensor(negative_count / positive_count, dtype=torch.float32), positive_count, negative_count


def train_model(model, x, edge_index, labels, train_idx, val_idx, pos_weight, deps):
    torch = deps["torch"]
    F = deps["F"]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    y = torch.tensor(labels, dtype=torch.float32)
    train_tensor = torch.tensor(train_idx, dtype=torch.long)
    val_tensor = torch.tensor(val_idx, dtype=torch.long)
    log_rows = []
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, edge_index)
        train_loss = F.binary_cross_entropy_with_logits(logits[train_tensor], y[train_tensor], pos_weight=pos_weight)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _ = model(x, edge_index)
            val_loss = F.binary_cross_entropy_with_logits(logits[val_tensor], y[val_tensor], pos_weight=pos_weight)
            val_scores = torch.sigmoid(logits[val_tensor]).cpu().numpy()
            val_true = labels[val_idx].astype(int)
            metrics = compute_metrics(val_true, val_scores, 0.5, deps)

        row = {"epoch": epoch, "train_loss": float(train_loss.item()), "val_loss": float(val_loss.item()), **{f"val_{k}": v for k, v in metrics.items() if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}}
        log_rows.append(row)
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(log_rows)


def heuristic_score(frame: pd.DataFrame) -> pd.Series:
    columns = ["imbalance_ratio", "activity_count", "entity_concentration_ratio", "sell_count"]
    values = frame[columns].copy()
    for column in columns:
        series = np.log1p(pd.to_numeric(values[column], errors="coerce").fillna(0).clip(lower=0))
        denom = series.max() - series.min()
        values[column] = 0.0 if denom == 0 else (series - series.min()) / denom
    lifespan = pd.to_numeric(frame["lifespan_hours"], errors="coerce").fillna(0).clip(lower=0)
    lifespan_score = 1.0 - (lifespan / lifespan.max()) if lifespan.max() else 0.0
    score = 0.30 * values["imbalance_ratio"] + 0.25 * values["entity_concentration_ratio"] + 0.20 * values["activity_count"] + 0.15 * values["sell_count"] + 0.10 * lifespan_score
    return pd.Series(score, index=frame.index, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def add_2025_evidence_columns(token_2025: pd.DataFrame) -> pd.DataFrame:
    scored = token_2025.copy()
    numeric_columns = ["risk_score", "total_volume", "activity_count", "imbalance_ratio", "entity_concentration_ratio", "unique_wallets", "lifespan_hours"]
    for column in numeric_columns:
        scored[column] = pd.to_numeric(scored[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    volume_floor = max(100.0, float(scored["total_volume"].quantile(0.25))) if len(scored) else 100.0
    activity_floor = max(3.0, float(scored["activity_count"].quantile(0.25))) if len(scored) else 3.0
    wallet_floor = max(2.0, float(scored["unique_wallets"].quantile(0.25))) if len(scored) else 2.0
    imbalance_cutoff = float(scored["imbalance_ratio"].quantile(0.75)) if len(scored) else 0.0
    concentration_cutoff = float(scored["entity_concentration_ratio"].quantile(0.75)) if len(scored) else 0.0
    short_lifespan_cutoff = float(scored["lifespan_hours"].quantile(0.25)) if len(scored) else 0.0

    scored["evidence_activity_pass"] = scored["activity_count"] >= activity_floor
    scored["evidence_volume_pass"] = scored["total_volume"] >= volume_floor
    scored["evidence_wallet_pass"] = scored["unique_wallets"] >= wallet_floor
    scored["evidence_imbalance_pass"] = scored["imbalance_ratio"] >= imbalance_cutoff
    scored["evidence_concentration_pass"] = scored["entity_concentration_ratio"] >= concentration_cutoff
    scored["evidence_short_lifespan_pass"] = scored["lifespan_hours"] <= short_lifespan_cutoff
    signal_columns = ["evidence_imbalance_pass", "evidence_concentration_pass", "evidence_short_lifespan_pass"]
    scored["evidence_signal_count"] = scored[signal_columns].sum(axis=1).astype(int)
    scored["evidence_filter_pass"] = (
        scored["evidence_activity_pass"]
        & scored["evidence_volume_pass"]
        & scored["evidence_wallet_pass"]
        & (scored["evidence_signal_count"] >= 1)
    )
    scored["evidence_reason"] = scored.apply(
        lambda row: "; ".join(
            reason
            for reason, passed in [
                ("high imbalance", row["evidence_imbalance_pass"]),
                ("high concentration", row["evidence_concentration_pass"]),
                ("short lifespan", row["evidence_short_lifespan_pass"]),
            ]
            if passed
        ),
        axis=1,
    )
    return scored


def save_plots(training_log: pd.DataFrame, embeddings: pd.DataFrame, token_mask: np.ndarray, deps) -> None:
    plt = deps["plt"]
    if not training_log.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(training_log["epoch"], training_log["train_loss"], label="Train loss")
        plt.plot(training_log["epoch"], training_log["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("GraphSAGE Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOSS_CURVE_PATH, dpi=160)
        plt.close()
    else:
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "Embedding-only fallback: no supervised loss curve", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(LOSS_CURVE_PATH, dpi=160)
        plt.close()

    embedding_columns = [col for col in embeddings.columns if col.startswith("embedding_")]
    sample = embeddings.loc[token_mask, ["node_type", "year", *embedding_columns]].copy()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    coords = None
    method = None
    try:
        import umap

        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(sample[embedding_columns].to_numpy())
        method = "UMAP"
    except Exception as exc:
        log(f"UMAP unavailable or failed, using PCA if possible: {exc}")
        try:
            coords = deps["PCA"](n_components=2, random_state=42).fit_transform(sample[embedding_columns].to_numpy())
            method = "PCA"
        except Exception as pca_exc:
            log(f"Embedding visualization skipped: {pca_exc}")

    if coords is not None:
        plt.figure(figsize=(8, 6))
        colors = np.where(sample["year"].to_numpy() == 2025, "#d62728", "#2ca02c")
        plt.scatter(coords[:, 0], coords[:, 1], s=10, c=colors, alpha=0.65)
        plt.title(f"GraphSAGE Token Embeddings ({method})")
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        plt.tight_layout()
        plt.savefig(EMBEDDING_PLOT_PATH, dpi=160)
        plt.close()


def main() -> None:
    check_inputs()
    deps = require_dependencies()
    torch = deps["torch"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_parquet(NODES_PATH)
    edges = pd.read_parquet(EDGES_PATH)
    token_features = pd.read_parquet(TOKEN_FEATURES_PATH)
    nodes = clean_numeric_frame(nodes, FEATURE_COLUMNS)
    token_features = clean_numeric_frame(token_features, ["activity_count", "sell_count", "total_volume", "imbalance_ratio", "entity_concentration_ratio", "unique_wallets", "lifespan_hours"])

    token_features, training_mode, _ = prepare_labels(token_features)
    label_map = token_features[["year", "token_address", "label"]].rename(columns={"token_address": "original_id"})
    nodes = nodes.merge(label_map, on=["year", "original_id"], how="left")
    nodes, edge_index, x = build_graph_tensors(nodes, edges, deps)

    token_2024_mask = nodes["node_type"].eq("token") & nodes["year"].eq(2024) & nodes["label"].notna()
    labeled_indices = np.flatnonzero(token_2024_mask.to_numpy())
    labels = nodes["label"].fillna(0).to_numpy(dtype=np.float32)
    class_counts = pd.Series(labels[labeled_indices]).value_counts().to_dict() if len(labeled_indices) else {}

    supervised_ready = len(labeled_indices) >= 10 and len(class_counts) == 2 and min(class_counts.values()) >= 2
    tuned_threshold = 0.5
    if not supervised_ready:
        log("Not enough labeled 2024 token nodes for reliable supervised training; using embedding-only fallback.")
        training_mode = "embedding-only fallback"
        scaled_nodes = log1p_scale_feature_frame(nodes)
        x_np = scaled_nodes[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        x_np = np.nan_to_num(x_np, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(x_np, dtype=torch.float32)
        model = make_model(x.shape[1], 64, 0.3, deps)
        model.eval()
        with torch.no_grad():
            logits, embedding_tensor = model(x, edge_index)
        training_log = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc"])
        pd.DataFrame([{"threshold": np.nan, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan}]).to_csv(METRICS_PATH, index=False)
        pd.DataFrame([{"threshold": np.nan, "selection_metric": "validation_f1", "positive_train_count": np.nan, "negative_train_count": np.nan, "pos_weight": np.nan}]).to_csv(THRESHOLD_PATH, index=False)
    else:
        train_idx, val_idx = deps["train_test_split"](
            labeled_indices,
            test_size=0.20,
            random_state=42,
            stratify=labels[labeled_indices].astype(int),
        )
        scaled_nodes = log1p_scale_feature_frame(nodes)
        x_np = scaled_nodes[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        x_scaled, _, _ = standardize_features(x_np, train_idx)
        x = torch.tensor(x_scaled.astype(np.float32), dtype=torch.float32)
        model = make_model(x.shape[1], 64, 0.3, deps)
        pos_weight, positive_train_count, negative_train_count = class_pos_weight(labels, train_idx, deps)
        log(f"Using BCEWithLogits pos_weight={float(pos_weight.item()):.6f} from {positive_train_count:,} positive and {negative_train_count:,} negative train labels")
        training_log = train_model(model, x, edge_index, labels, train_idx, val_idx, pos_weight, deps)
        model.eval()
        with torch.no_grad():
            logits, embedding_tensor = model(x, edge_index)
            scores = torch.sigmoid(logits).cpu().numpy()
        tuned_threshold, metrics = tune_threshold(labels[val_idx].astype(int), scores[val_idx], deps)
        metrics = {"threshold": tuned_threshold, **metrics}
        log(f"Tuned validation threshold={tuned_threshold:.3f} using validation F1")
        pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)
        pd.DataFrame(
            [
                {
                    "threshold": tuned_threshold,
                    "selection_metric": "validation_f1",
                    "positive_train_count": positive_train_count,
                    "negative_train_count": negative_train_count,
                    "pos_weight": float(pos_weight.item()),
                }
            ]
        ).to_csv(THRESHOLD_PATH, index=False)

    training_log.to_csv(TRAINING_LOG_PATH, index=False)
    embeddings_np = embedding_tensor.detach().cpu().numpy()
    embedding_columns = [f"embedding_{i}" for i in range(embeddings_np.shape[1])]
    embedding_frame = pd.concat(
        [
            nodes[["node_id", "node_type", "original_id", "year"]].reset_index(drop=True),
            pd.DataFrame(embeddings_np, columns=embedding_columns),
        ],
        axis=1,
    )
    embedding_frame.to_parquet(EMBEDDINGS_PATH, index=False)

    token_2025 = token_features[token_features["year"].eq(2025)].copy()
    if training_mode == "embedding-only fallback":
        token_2025["risk_score"] = pd.to_numeric(heuristic_score(token_2025), errors="coerce").clip(0.0, 1.0)
        print(f"Missing 2025 risk_score values before fillna: 0")
        print(f"Missing 2025 risk_score values after fillna: {int(token_2025['risk_score'].isna().sum()):,}")
    else:
        score_lookup = nodes[["node_type", "original_id", "year"]].copy()
        score_lookup["risk_score"] = torch.sigmoid(logits).detach().cpu().numpy()
        score_lookup = score_lookup[score_lookup["node_type"].eq("token")].rename(columns={"original_id": "token_address"})
        token_2025 = token_2025.merge(score_lookup[["year", "token_address", "risk_score"]], on=["year", "token_address"], how="left")
        token_2025["risk_score"] = pd.to_numeric(token_2025["risk_score"], errors="coerce")
        missing_before_fill = int(token_2025["risk_score"].isna().sum())
        fallback_scores = heuristic_score(token_2025)
        token_2025["risk_score"] = token_2025["risk_score"].fillna(fallback_scores)
        token_2025["risk_score"] = pd.to_numeric(token_2025["risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        missing_after_fill = int(token_2025["risk_score"].isna().sum())
        print(f"Missing 2025 risk_score values before fillna: {missing_before_fill:,}")
        print(f"Missing 2025 risk_score values after fillna: {missing_after_fill:,}")

    token_2025 = token_2025.sort_values("risk_score", ascending=False).reset_index(drop=True)
    token_2025["risk_rank"] = np.arange(1, len(token_2025) + 1)
    token_2025["risk_predicted_label"] = (token_2025["risk_score"] >= tuned_threshold).astype(int)
    token_2025 = add_2025_evidence_columns(token_2025)
    evidence_filtered_top = token_2025[token_2025["evidence_filter_pass"]].copy()
    if evidence_filtered_top.empty:
        log("Evidence-aware 2025 filter produced no tokens; falling back to score-only top 100.")
        evidence_filtered_top = token_2025.copy()
    risk_columns = ["token_address", "year", "risk_score", "risk_rank", "risk_predicted_label", "total_volume", "activity_count", "imbalance_ratio", "entity_concentration_ratio", "unique_wallets", "lifespan_hours", "evidence_filter_pass", "evidence_signal_count", "evidence_reason"]
    token_2025[risk_columns].to_csv(RISK_SCORES_PATH, index=False)
    evidence_filtered_top.sort_values("risk_score", ascending=False)[risk_columns].head(100).to_csv(TOP_RISKY_PATH, index=False)
    save_plots(training_log, embedding_frame, embedding_frame["node_type"].eq("token").to_numpy(), deps)

    print("GraphSAGE validation summary")
    print(f"Number of nodes: {len(nodes):,}")
    print(f"Number of edges: {len(edges):,}")
    print(f"Feature dimension: {len(FEATURE_COLUMNS):,}")
    print(f"Number of 2024 labeled token nodes: {len(labeled_indices):,}")
    print(f"Number of 2025 scored token nodes: {len(token_2025):,}")
    print(f"Training mode: {training_mode}")
    print(f"Validation threshold used for risk labels: {tuned_threshold:.3f}")
    print(f"2025 evidence-aware top-risk candidates: {len(evidence_filtered_top):,}")
    print("Output paths:")
    for path in [TRAINING_LOG_PATH, METRICS_PATH, THRESHOLD_PATH, RISK_SCORES_PATH, TOP_RISKY_PATH, EMBEDDINGS_PATH, LOSS_CURVE_PATH, EMBEDDING_PLOT_PATH, LABEL_NOTES_PATH]:
        print(f"- {path}")
    print("Reminder: 2025 outputs are risk scores and heuristic agreement artifacts, not definitive ground-truth prediction accuracy.")


if __name__ == "__main__":
    main()
