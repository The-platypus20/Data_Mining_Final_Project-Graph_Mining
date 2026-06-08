from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGES_PATH = PROJECT_ROOT / "data" / "silver" / "dune_token_wallet_edges_2024_2025.parquet"
TOKEN_FEATURES_PATH = PROJECT_ROOT / "data" / "gold" / "dune_token_features_2024_2025.parquet"
RISK_SCORES_PATH = PROJECT_ROOT / "data" / "gold" / "gnn_risk_scores_2025.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
STATIC_PNG_PATH = REPORTS_DIR / "topology_star_cluster_examples.png"
INTERACTIVE_TOPOLOGY_PATH = REPORTS_DIR / "interactive_token_wallet_topology.html"
GNN_DIAGRAM_PATH = REPORTS_DIR / "interactive_gnn_layer_diagram.html"


def log(message: str) -> None:
    print(f"[dune-topology] {message}", flush=True)


def require_dependencies():
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("pip install pandas pyarrow networkx matplotlib pyvis scikit-learn torch")
        sys.exit(1)
    try:
        from pyvis.network import Network
    except ImportError:
        Network = None
        log("PyVis is missing. Install with: pip install pyvis. A plain HTML fallback will be written.")
    return plt, nx, Network


def check_inputs() -> None:
    missing = [path for path in [EDGES_PATH, TOKEN_FEATURES_PATH] if not path.exists()]
    if missing:
        print("Missing Dune lakehouse files:")
        for path in missing:
            print(f"- {path}")
        print("Please run:")
        print("python src/dune_spark_lakehouse_pipeline.py")
        sys.exit(1)


def node_id(kind: str, original_id: str) -> str:
    return f"{kind}:{original_id}"


def normalize_width(values: pd.Series, min_width: float = 0.4, max_width: float = 5.0) -> pd.Series:
    values = np.log1p(pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0))
    if values.max() == values.min():
        return pd.Series(np.full(len(values), min_width), index=values.index)
    return min_width + (values - values.min()) / (values.max() - values.min()) * (max_width - min_width)


def graph_from_edges(edge_frame: pd.DataFrame, nx):
    graph = nx.Graph()
    widths = normalize_width(edge_frame["total_volume_usd"])
    for idx, row in edge_frame.iterrows():
        wallet = node_id("wallet", row["wallet_address"])
        token = node_id("token", row["token_address"])
        graph.add_node(wallet, node_type="wallet", label=str(row["wallet_address"])[:8], volume=float(row["total_volume_usd"]))
        graph.add_node(token, node_type="token", label=str(row["token_address"])[:8], volume=float(row["total_volume_usd"]))
        graph.add_edge(
            wallet,
            token,
            weight=float(row["total_volume_usd"]),
            width=float(widths.loc[idx]),
            buy_count=int(row["buy_count"]),
            sell_count=int(row["sell_count"]),
            total_count=int(row["total_count"]),
        )
    return graph


def select_star_edges(edges_2025: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    wallet_degree = edges_2025.groupby("wallet_address")["token_address"].nunique().sort_values(ascending=False)
    if wallet_degree.empty:
        return edges_2025.head(0), ""
    center = wallet_degree.index[0]
    center_edges = edges_2025[edges_2025["wallet_address"].eq(center)].nlargest(80, "total_volume_usd")
    tokens = center_edges["token_address"].unique()[:40]
    neighbor_edges = edges_2025[edges_2025["token_address"].isin(tokens)].nlargest(140, "total_volume_usd")
    star_edges = pd.concat([center_edges, neighbor_edges], ignore_index=True).drop_duplicates(["wallet_address", "token_address", "year"])
    return star_edges.head(180), str(center)


def select_cluster_edges(edges_2025: pd.DataFrame, token_features: pd.DataFrame, risk_scores: pd.DataFrame | None) -> pd.DataFrame:
    if risk_scores is not None and not risk_scores.empty:
        selected_tokens = risk_scores.nlargest(25, "risk_score")["token_address"].astype(str).tolist()
    else:
        selected_tokens = token_features[token_features["year"].eq(2025)].nlargest(25, "total_volume")["token_address"].astype(str).tolist()
    cluster = edges_2025[edges_2025["token_address"].astype(str).isin(selected_tokens)].copy()
    if cluster.empty:
        return cluster
    top_wallets = cluster.groupby("wallet_address")["total_volume_usd"].sum().nlargest(120).index
    cluster = cluster[cluster["wallet_address"].isin(top_wallets)].nlargest(240, "total_volume_usd")
    return cluster


def draw_static_png(edges: pd.DataFrame, token_features: pd.DataFrame, risk_scores: pd.DataFrame | None, plt, nx):
    edges_2025 = edges[edges["year"].eq(2025)].copy()
    star_edges, center_wallet = select_star_edges(edges_2025)
    cluster_edges = select_cluster_edges(edges_2025, token_features, risk_scores)
    risky_tokens = set(risk_scores["token_address"].astype(str).head(100)) if risk_scores is not None and not risk_scores.empty else set()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    panels = [
        (axes[0], star_edges, "(a) Star-like topology", center_wallet),
        (axes[1], cluster_edges, "(b) Cluster-like topology", ""),
    ]
    for axis, frame, title, center in panels:
        axis.set_title(title, fontsize=14)
        axis.axis("off")
        if frame.empty:
            axis.text(0.5, 0.5, "No 2025 graph edges available", ha="center", va="center")
            continue
        graph = graph_from_edges(frame, nx)
        try:
            core = nx.k_core(graph, k=2)
            if title.startswith("(b)") and core.number_of_nodes() >= 8:
                graph = core
        except Exception:
            pass
        pos = nx.spring_layout(graph, seed=42, k=0.35)
        colors = []
        sizes = []
        for node, attrs in graph.nodes(data=True):
            degree = graph.degree(node)
            if attrs["node_type"] == "token":
                token = node.split("token:", 1)[1]
                colors.append("#d62728" if token in risky_tokens else "#2ca02c")
                sizes.append(80 + degree * 18)
            else:
                wallet = node.split("wallet:", 1)[1]
                colors.append("#ff7f0e" if wallet == center or degree >= 8 else "#9e9e9e")
                sizes.append(55 + degree * 12)
        widths = [attrs.get("width", 1.0) for _, _, attrs in graph.edges(data=True)]
        nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.28, edge_color="#555555", ax=axis)
        nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, alpha=0.88, linewidths=0.5, edgecolors="#222222", ax=axis)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", label="Token", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e", label="Wallet", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", label="High-volume wallet", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", label="High-risk token", markersize=10),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(STATIC_PNG_PATH, dpi=180)
    plt.close(fig)
    return center_wallet, len(set(cluster_edges["token_address"]).union(set(cluster_edges["wallet_address"])))


def tooltip_lines(lines: list[tuple[str, object]]) -> str:
    return "<br>".join(f"<b>{html.escape(str(key))}</b>: {html.escape(str(value))}" for key, value in lines)


def build_interactive_topology(edges: pd.DataFrame, token_features: pd.DataFrame, risk_scores: pd.DataFrame | None, Network):
    edges_2025 = edges[edges["year"].eq(2025)].copy()
    if risk_scores is not None and not risk_scores.empty:
        selected_tokens = risk_scores.nlargest(20, "risk_score")["token_address"].astype(str).tolist()
        risk_lookup = dict(zip(risk_scores["token_address"].astype(str), risk_scores["risk_score"]))
    else:
        selected_tokens = token_features[token_features["year"].eq(2025)].nlargest(20, "total_volume")["token_address"].astype(str).tolist()
        risk_lookup = {}

    frame = edges_2025[edges_2025["token_address"].astype(str).isin(selected_tokens)].copy()
    top_wallets = frame.groupby("wallet_address")["total_volume_usd"].sum().nlargest(300).index
    frame = frame[frame["wallet_address"].isin(top_wallets)].nlargest(1000, "total_volume_usd")
    feature_lookup = token_features[token_features["year"].eq(2025)].set_index("token_address").to_dict("index")
    wallet_degree = frame.groupby("wallet_address")["token_address"].nunique().to_dict()
    wallet_volume = frame.groupby("wallet_address")["total_volume_usd"].sum().to_dict()

    if Network is None:
        write_plain_topology_html(frame, feature_lookup, risk_lookup, wallet_degree, wallet_volume)
    else:
        net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False)
        net.force_atlas_2based(gravity=-45, central_gravity=0.012, spring_length=120, spring_strength=0.08)
        for token in selected_tokens:
            if token not in feature_lookup:
                continue
            features = feature_lookup[token]
            risk = risk_lookup.get(token)
            value = float(risk) if risk is not None else float(features.get("total_volume", 0) or 0)
            size = 14 + min(28, np.log1p(max(value, 0)) * (8 if risk is None else 20))
            title = tooltip_lines(
                [
                    ("node type", "token"),
                    ("token address", token),
                    ("year", 2025),
                    ("total volume", round(float(features.get("total_volume", 0) or 0), 4)),
                    ("activity count", int(features.get("activity_count", 0) or 0)),
                    ("risk score", "" if risk is None else round(float(risk), 6)),
                ]
            )
            net.add_node(node_id("token", token), label=token[:8], title=title, color="#d62728" if risk is not None else "#2ca02c", size=size)
        for wallet in top_wallets:
            degree = int(wallet_degree.get(wallet, 0))
            volume = float(wallet_volume.get(wallet, 0) or 0)
            title = tooltip_lines(
                [
                    ("node type", "wallet"),
                    ("wallet address", wallet),
                    ("degree", degree),
                    ("total connected volume", round(volume, 4)),
                ]
            )
            net.add_node(node_id("wallet", wallet), label=str(wallet)[:8], title=title, color="#9e9e9e", size=8 + min(24, degree * 1.5))
        widths = normalize_width(frame["total_volume_usd"], 0.5, 6.0)
        for idx, row in frame.iterrows():
            title = tooltip_lines(
                [
                    ("token address", row["token_address"]),
                    ("wallet address", row["wallet_address"]),
                    ("year", int(row["year"])),
                    ("buy_count", int(row["buy_count"])),
                    ("sell_count", int(row["sell_count"])),
                    ("total_count", int(row["total_count"])),
                    ("total_volume_usd", round(float(row["total_volume_usd"]), 4)),
                ]
            )
            net.add_edge(node_id("wallet", row["wallet_address"]), node_id("token", row["token_address"]), title=title, width=float(widths.loc[idx]))
        net.write_html(str(INTERACTIVE_TOPOLOGY_PATH), notebook=False, open_browser=False)
    return len(set(frame["token_address"])), len(set(frame["wallet_address"])), len(frame)


def write_plain_topology_html(frame: pd.DataFrame, feature_lookup: dict, risk_lookup: dict, wallet_degree: dict, wallet_volume: dict) -> None:
    nodes = []
    edges = []
    for token in sorted(set(frame["token_address"])):
        features = feature_lookup.get(token, {})
        nodes.append({"id": node_id("token", token), "label": str(token)[:8], "type": "token", "risk": risk_lookup.get(token), "volume": features.get("total_volume", 0)})
    for wallet in sorted(set(frame["wallet_address"])):
        nodes.append({"id": node_id("wallet", wallet), "label": str(wallet)[:8], "type": "wallet", "degree": wallet_degree.get(wallet, 0), "volume": wallet_volume.get(wallet, 0)})
    for _, row in frame.iterrows():
        edges.append({"from": node_id("wallet", row["wallet_address"]), "to": node_id("token", row["token_address"]), "volume": float(row["total_volume_usd"])})
    INTERACTIVE_TOPOLOGY_PATH.write_text(
        f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Dune Token-Wallet Topology</title></head>
<body>
<h1>Dune Token-Wallet Topology</h1>
<p>Install pyvis for the interactive force-directed version: <code>pip install pyvis</code>.</p>
<pre>{html.escape(json.dumps({"nodes": nodes, "edges": edges}, indent=2))}</pre>
</body>
</html>
""",
        encoding="utf-8",
    )


def build_gnn_diagram_html() -> None:
    GNN_DIAGRAM_PATH.write_text(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphSAGE Risk Scoring Diagram</title>
<style>
body { margin: 0; font-family: Arial, sans-serif; background: #f7f7f5; color: #202020; }
main { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }
h1 { font-size: 28px; margin-bottom: 8px; }
.flow { display: grid; grid-template-columns: repeat(7, minmax(110px, 1fr)); gap: 12px; align-items: stretch; margin-top: 28px; }
.step { background: white; border: 1px solid #d9d9d2; border-radius: 8px; padding: 18px 14px; min-height: 110px; box-shadow: 0 1px 4px rgba(0,0,0,.06); cursor: pointer; }
.step strong { display: block; font-size: 15px; margin-bottom: 10px; }
.step span { font-size: 13px; line-height: 1.35; color: #555; }
.arrow { display: flex; align-items: center; justify-content: center; color: #777; font-size: 24px; }
.detail { margin-top: 24px; background: #202020; color: white; padding: 18px; border-radius: 8px; min-height: 60px; }
@media (max-width: 900px) { .flow { grid-template-columns: 1fr; } .arrow { transform: rotate(90deg); } }
</style>
</head>
<body>
<main>
<h1>Dune GraphSAGE Risk Scoring Architecture</h1>
<p>Raw Dune swaps -> token-wallet graph -> GraphSAGE message passing -> token embedding -> risk score.</p>
<section class="flow">
<div class="step" data-detail="Monthly 2024-2025 Dune swap CSVs provide token, wallet, amount, time, and transaction fields."><strong>Raw Dune swaps</strong><span>Swap records from Dune CSV exports.</span></div>
<div class="arrow">-></div>
<div class="step" data-detail="Bronze swaps become silver token-side events and token-wallet edges with counts, volumes, and activity windows."><strong>Token-wallet graph</strong><span>Wallet nodes connect to token nodes by observed trades.</span></div>
<div class="arrow">-></div>
<div class="step" data-detail="GraphSAGE layer 1 aggregates nearby wallet and token information, then applies ReLU and dropout."><strong>GraphSAGE layer 1</strong><span>Message passing over local neighborhoods.</span></div>
<div class="arrow">-></div>
<div class="step" data-detail="GraphSAGE layer 2 forms compact token embeddings used for downstream exploratory scoring."><strong>GraphSAGE layer 2</strong><span>Second-hop topology and feature aggregation.</span></div>
<div class="step" data-detail="The token embedding feeds a small classifier head. Outputs are risk scores, not confirmed 2025 labels."><strong>MLP classifier</strong><span>Token embedding to risk score.</span></div>
</section>
<section class="detail" id="detail">Click a stage to inspect the role it plays in the pipeline.</section>
</main>
<script>
document.querySelectorAll('.step').forEach(step => {
  step.addEventListener('click', () => {
    document.getElementById('detail').textContent = step.dataset.detail;
  });
});
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    check_inputs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt, nx, Network = require_dependencies()

    edges = pd.read_parquet(EDGES_PATH)
    token_features = pd.read_parquet(TOKEN_FEATURES_PATH)
    risk_scores = pd.read_csv(RISK_SCORES_PATH) if RISK_SCORES_PATH.exists() else None
    for column in ["total_volume_usd", "buy_count", "sell_count", "total_count"]:
        edges[column] = pd.to_numeric(edges[column], errors="coerce").fillna(0)
    token_features["total_volume"] = pd.to_numeric(token_features["total_volume"], errors="coerce").fillna(0)
    if risk_scores is not None:
        risk_scores["risk_score"] = pd.to_numeric(risk_scores["risk_score"], errors="coerce").fillna(0)

    star_center, cluster_size = draw_static_png(edges, token_features, risk_scores, plt, nx)
    token_count, wallet_count, edge_count = build_interactive_topology(edges, token_features, risk_scores, Network)
    build_gnn_diagram_html()

    print(f"Static PNG path: {STATIC_PNG_PATH}")
    print(f"Interactive topology HTML path: {INTERACTIVE_TOPOLOGY_PATH}")
    print(f"Interactive GNN layer diagram path: {GNN_DIAGRAM_PATH}")
    print(f"Number of token nodes shown: {token_count:,}")
    print(f"Number of wallet nodes shown: {wallet_count:,}")
    print(f"Number of edges shown: {edge_count:,}")
    print(f"Selected star center wallet: {star_center}")
    print(f"Selected cluster size: {cluster_size:,}")
    print("These topology graphs are Dune-based exploratory token-wallet structures. They are not confirmed scammer identity graphs because creator and LP-profitor roles require Solscan or Solana RPC enrichment.")


if __name__ == "__main__":
    main()
