from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DUNE_DIR = PROJECT_ROOT / "data" / "raw" / "dune"

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "dune_swaps_2024_2025.parquet"
SILVER_EVENTS_PATH = PROJECT_ROOT / "data" / "silver" / "dune_token_events_2024_2025.parquet"
SILVER_EDGES_PATH = PROJECT_ROOT / "data" / "silver" / "dune_token_wallet_edges_2024_2025.parquet"
GOLD_FEATURES_PATH = PROJECT_ROOT / "data" / "gold" / "dune_token_features_2024_2025.parquet"
PYG_NODES_PATH = PROJECT_ROOT / "data" / "gold" / "pyg_nodes_2024_2025.parquet"
PYG_EDGES_PATH = PROJECT_ROOT / "data" / "gold" / "pyg_edges_2024_2025.parquet"

OUTPUT_PATHS = [
    BRONZE_PATH,
    SILVER_EVENTS_PATH,
    SILVER_EDGES_PATH,
    GOLD_FEATURES_PATH,
    PYG_NODES_PATH,
    PYG_EDGES_PATH,
]

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

BRONZE_COLUMNS = [
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
    "source_file",
    "year",
    "month",
]

EVENT_COLUMNS = [
    "token_address",
    "wallet_address",
    "side",
    "token_amount",
    "amount_usd",
    "block_time",
    "block_date",
    "project",
    "trade_source",
    "tx_id",
    "year",
    "month",
]

TOKEN_FEATURE_COLUMNS = [
    "year",
    "token_address",
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
    "first_seen",
    "last_seen",
    "active_days",
]

NODE_COLUMNS = [
    "node_id",
    "node_type",
    "original_id",
    "year",
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

EDGE_COLUMNS = [
    "src_node_id",
    "dst_node_id",
    "src_original_id",
    "dst_original_id",
    "edge_type",
    "year",
    "total_count",
    "total_volume_usd",
    "buy_count",
    "sell_count",
    "buy_volume_usd",
    "sell_volume_usd",
]


def log(message: str) -> None:
    print(f"[dune-polars] {message}", flush=True)


def require_polars():
    try:
        import polars as pl
    except ImportError:
        print("Polars is required for this script.")
        print("pip install polars pyarrow")
        sys.exit(1)
    return pl


def ensure_output_dirs() -> None:
    for path in OUTPUT_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)


def find_raw_files() -> list[Path]:
    files = sorted((RAW_DUNE_DIR / "2024").glob("*.csv")) + sorted((RAW_DUNE_DIR / "2025").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw Dune CSV files found under {RAW_DUNE_DIR}")
    return files


def parse_year_month(path: Path) -> tuple[int, int]:
    match = re.match(r"^(\d{4})_(\d{2})\.csv$", path.name)
    if not match:
        raise ValueError(f"Expected file name like 2024_01.csv, got {path.name}")
    return int(match.group(1)), int(match.group(2))


def validate_month_files(raw_files: list[Path]) -> None:
    expected = {f"{year}_{month:02d}.csv" for year in (2024, 2025) for month in range(1, 13)}
    found = {path.name for path in raw_files}
    missing = sorted(expected - found)
    if missing:
        log(f"Warning: missing expected monthly Dune CSV files: {missing}")


def read_one_csv(path: Path, pl):
    year, month = parse_year_month(path)
    frame = pl.read_csv(
        path,
        columns=RAW_COLUMNS,
        null_values=["", "NULL", "nil", "<nil>"],
        infer_schema_length=10000,
        ignore_errors=True,
    )
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame.with_columns(
        pl.lit(path.name).alias("source_file"),
        pl.lit(year, dtype=pl.Int32).alias("year"),
        pl.lit(month, dtype=pl.Int32).alias("month"),
    )


def parse_datetime_expr(pl):
    cleaned = pl.col("block_time").cast(pl.Utf8).str.replace(r"\s+UTC$", "")
    return cleaned.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)


def build_bronze(raw_files: list[Path], pl):
    log("Phase 1: building Bronze swaps from raw Dune CSVs")
    frames = [read_one_csv(path, pl) for path in raw_files]
    bronze = (
        pl.concat(frames, how="vertical")
        .select(
            parse_datetime_expr(pl).alias("block_time"),
            pl.col("block_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias("block_date"),
            pl.col("project").cast(pl.Utf8),
            pl.col("trade_source").cast(pl.Utf8),
            pl.col("token_bought_mint_address").cast(pl.Utf8),
            pl.col("token_sold_mint_address").cast(pl.Utf8),
            pl.col("token_bought_amount").cast(pl.Float64, strict=False),
            pl.col("token_sold_amount").cast(pl.Float64, strict=False),
            pl.col("amount_usd").cast(pl.Float64, strict=False),
            pl.col("fee_usd").cast(pl.Float64, strict=False),
            pl.col("trader_id").cast(pl.Utf8),
            pl.col("tx_id").cast(pl.Utf8),
            pl.col("source_file").cast(pl.Utf8),
            pl.col("year").cast(pl.Int32),
            pl.col("month").cast(pl.Int32),
        )
        .filter(
            pl.col("trader_id").is_not_null()
            & pl.col("tx_id").is_not_null()
            & pl.col("amount_usd").is_not_null()
            & (pl.col("amount_usd") >= 0)
        )
    )
    bronze.write_parquet(BRONZE_PATH)
    print_examples("bronze", bronze)
    log(f"Wrote Bronze swaps: {BRONZE_PATH}")
    return bronze


def build_silver_events(bronze, pl):
    log("Phase 2: building Silver token-side events")
    common = ["amount_usd", "block_time", "block_date", "project", "trade_source", "tx_id", "year", "month"]
    bought = bronze.select(
        pl.col("token_bought_mint_address").alias("token_address"),
        pl.col("trader_id").alias("wallet_address"),
        pl.lit("buy").alias("side"),
        pl.col("token_bought_amount").alias("token_amount"),
        *[pl.col(column) for column in common],
    )
    sold = bronze.select(
        pl.col("token_sold_mint_address").alias("token_address"),
        pl.col("trader_id").alias("wallet_address"),
        pl.lit("sell").alias("side"),
        pl.col("token_sold_amount").alias("token_amount"),
        *[pl.col(column) for column in common],
    )
    events = (
        pl.concat([bought, sold], how="vertical")
        .select(EVENT_COLUMNS)
        .filter(
            pl.col("token_address").is_not_null()
            & pl.col("wallet_address").is_not_null()
            & pl.col("tx_id").is_not_null()
            & pl.col("token_amount").is_not_null()
            & (pl.col("token_amount") >= 0)
        )
    )
    events.write_parquet(SILVER_EVENTS_PATH)
    print_examples("silver token events", events)
    log(f"Wrote Silver token events: {SILVER_EVENTS_PATH}")
    return events


def build_silver_edges(events, pl):
    log("Phase 3: building Silver token-wallet edges")
    edges = (
        events.group_by(["year", "token_address", "wallet_address"])
        .agg(
            (pl.col("side") == "buy").sum().cast(pl.Int64).alias("buy_count"),
            (pl.col("side") == "sell").sum().cast(pl.Int64).alias("sell_count"),
            pl.len().cast(pl.Int64).alias("total_count"),
            pl.when(pl.col("side") == "buy").then(pl.col("amount_usd")).otherwise(0.0).sum().alias("buy_volume_usd"),
            pl.when(pl.col("side") == "sell").then(pl.col("amount_usd")).otherwise(0.0).sum().alias("sell_volume_usd"),
            pl.col("amount_usd").sum().alias("total_volume_usd"),
            pl.col("block_time").min().alias("first_seen"),
            pl.col("block_time").max().alias("last_seen"),
            pl.col("block_date").n_unique().cast(pl.Int64).alias("active_days"),
        )
        .sort(["year", "token_address", "wallet_address"])
    )
    edges.write_parquet(SILVER_EDGES_PATH)
    print_examples("silver edges", edges)
    log(f"Wrote Silver token-wallet edges: {SILVER_EDGES_PATH}")
    return edges


def build_gold_features(events, edges, pl):
    log("Phase 4: building Gold token features")
    token_base = events.group_by(["year", "token_address"]).agg(
        pl.len().cast(pl.Int64).alias("activity_count"),
        (pl.col("side") == "buy").sum().cast(pl.Int64).alias("buy_count"),
        (pl.col("side") == "sell").sum().cast(pl.Int64).alias("sell_count"),
        pl.col("amount_usd").sum().alias("total_volume"),
        pl.when(pl.col("side") == "buy").then(pl.col("amount_usd")).otherwise(0.0).sum().alias("buy_volume_usd"),
        pl.when(pl.col("side") == "sell").then(pl.col("amount_usd")).otherwise(0.0).sum().alias("sell_volume_usd"),
        pl.col("wallet_address").n_unique().cast(pl.Int64).alias("unique_wallets"),
        pl.col("block_time").min().alias("first_seen"),
        pl.col("block_time").max().alias("last_seen"),
        pl.col("block_date").n_unique().cast(pl.Int64).alias("active_days"),
    )

    concentration = edges.group_by(["year", "token_address"]).agg(
        pl.col("total_volume_usd").max().alias("max_wallet_volume")
    )

    features = (
        token_base.join(concentration, on=["year", "token_address"], how="left")
        .with_columns(
            pl.when(pl.col("buy_volume_usd") > 0)
            .then(pl.col("sell_volume_usd") / pl.col("buy_volume_usd"))
            .when((pl.col("buy_volume_usd") == 0) & (pl.col("sell_volume_usd") > 0))
            .then(pl.col("sell_volume_usd"))
            .otherwise(0.0)
            .alias("imbalance_ratio"),
            (
                (pl.col("last_seen").cast(pl.Int64) - pl.col("first_seen").cast(pl.Int64))
                / 1_000_000
                / 3600
            ).alias("lifespan_hours"),
            pl.col("unique_wallets").alias("graph_degree"),
            pl.col("unique_wallets").alias("connected_entities"),
            pl.when(pl.col("total_volume") > 0)
            .then(pl.col("max_wallet_volume") / pl.col("total_volume"))
            .otherwise(0.0)
            .alias("entity_concentration_ratio"),
        )
        .drop("max_wallet_volume")
        .select(TOKEN_FEATURE_COLUMNS)
        .fill_null(0)
        .sort(["year", "token_address"])
    )
    features.write_parquet(GOLD_FEATURES_PATH)
    print_examples("gold token features", features)
    log(f"Wrote Gold token features: {GOLD_FEATURES_PATH}")
    return features


def make_node_id_expr(pl, node_type: str, id_column: str):
    return (
        pl.col("year").cast(pl.Utf8)
        + ":"
        + pl.lit(node_type)
        + ":"
        + pl.col(id_column).cast(pl.Utf8)
    )


def build_pyg_tables(token_features, edges, pl):
    log("Phase 5: building PyTorch Geometric node and edge tables")
    token_nodes = token_features.select(
        make_node_id_expr(pl, "token", "token_address").alias("node_id"),
        pl.lit("token").alias("node_type"),
        pl.col("token_address").alias("original_id"),
        pl.col("year"),
        pl.col("activity_count").cast(pl.Float64),
        pl.col("total_volume").cast(pl.Float64),
        pl.col("buy_volume_usd").cast(pl.Float64),
        pl.col("sell_volume_usd").cast(pl.Float64),
        pl.col("imbalance_ratio").cast(pl.Float64),
        pl.col("unique_wallets").cast(pl.Float64),
        pl.col("lifespan_hours").cast(pl.Float64),
        pl.col("graph_degree").cast(pl.Float64),
        pl.col("connected_entities").cast(pl.Float64),
        pl.col("entity_concentration_ratio").cast(pl.Float64),
        pl.lit(0.0).alias("wallet_degree"),
        pl.lit(0.0).alias("wallet_total_volume"),
        pl.lit(0.0).alias("wallet_total_count"),
        pl.lit(0.0).alias("wallet_unique_tokens"),
    )

    wallet_features = edges.group_by(["year", "wallet_address"]).agg(
        pl.col("token_address").n_unique().cast(pl.Float64).alias("wallet_degree"),
        pl.col("total_volume_usd").sum().alias("wallet_total_volume"),
        pl.col("total_count").sum().cast(pl.Float64).alias("wallet_total_count"),
        pl.col("token_address").n_unique().cast(pl.Float64).alias("wallet_unique_tokens"),
    )

    wallet_nodes = wallet_features.select(
        make_node_id_expr(pl, "wallet", "wallet_address").alias("node_id"),
        pl.lit("wallet").alias("node_type"),
        pl.col("wallet_address").alias("original_id"),
        pl.col("year"),
        pl.lit(0.0).alias("activity_count"),
        pl.lit(0.0).alias("total_volume"),
        pl.lit(0.0).alias("buy_volume_usd"),
        pl.lit(0.0).alias("sell_volume_usd"),
        pl.lit(0.0).alias("imbalance_ratio"),
        pl.lit(0.0).alias("unique_wallets"),
        pl.lit(0.0).alias("lifespan_hours"),
        pl.lit(0.0).alias("graph_degree"),
        pl.lit(0.0).alias("connected_entities"),
        pl.lit(0.0).alias("entity_concentration_ratio"),
        pl.col("wallet_degree"),
        pl.col("wallet_total_volume"),
        pl.col("wallet_total_count"),
        pl.col("wallet_unique_tokens"),
    )

    nodes = pl.concat([token_nodes, wallet_nodes], how="vertical").select(NODE_COLUMNS).sort(["year", "node_type", "original_id"])
    duplicate_node_count = nodes.select(pl.col("node_id").is_duplicated().sum()).item()
    if duplicate_node_count:
        raise ValueError(f"node_id must be unique, found {duplicate_node_count} duplicate rows")
    nodes.write_parquet(PYG_NODES_PATH)
    print_examples("pyg nodes", nodes)
    log(f"Wrote PyG nodes: {PYG_NODES_PATH}")

    pyg_edges = (
        edges.select(
            make_node_id_expr(pl, "wallet", "wallet_address").alias("src_node_id"),
            make_node_id_expr(pl, "token", "token_address").alias("dst_node_id"),
            pl.col("wallet_address").alias("src_original_id"),
            pl.col("token_address").alias("dst_original_id"),
            pl.lit("wallet_token_trade").alias("edge_type"),
            pl.col("year"),
            pl.col("total_count"),
            pl.col("total_volume_usd"),
            pl.col("buy_count"),
            pl.col("sell_count"),
            pl.col("buy_volume_usd"),
            pl.col("sell_volume_usd"),
        )
        .select(EDGE_COLUMNS)
        .sort(["year", "src_original_id", "dst_original_id"])
    )
    pyg_edges.write_parquet(PYG_EDGES_PATH)
    print_examples("pyg edges", pyg_edges)
    log(f"Wrote PyG edges: {PYG_EDGES_PATH}")
    return nodes, pyg_edges


def print_examples(name: str, frame) -> None:
    log(f"Example rows from {name}:")
    rows = frame.head(5).to_dicts()
    print(json.dumps(rows, indent=2, default=str, ensure_ascii=True))


def print_validation(raw_files: list[Path], bronze, events, edges, token_features, nodes, pyg_edges, pl) -> None:
    log("Phase 6: validation summary")
    token_counts = (
        token_features.group_by("year")
        .agg(pl.col("token_address").n_unique().alias("token_count"))
        .to_dict(as_series=False)
    )
    wallet_counts = (
        nodes.filter(pl.col("node_type") == "wallet")
        .group_by("year")
        .agg(pl.len().alias("wallet_count"))
        .to_dict(as_series=False)
    )
    token_count_by_year = dict(zip(token_counts.get("year", []), token_counts.get("token_count", [])))
    wallet_count_by_year = dict(zip(wallet_counts.get("year", []), wallet_counts.get("wallet_count", [])))

    print(f"Number of raw CSV files found: {len(raw_files):,}")
    print(f"Bronze row count: {bronze.height:,}")
    print(f"Silver token event row count: {events.height:,}")
    print(f"Silver edge row count: {edges.height:,}")
    print(f"Gold token feature row count: {token_features.height:,}")
    print(f"PyG node count: {nodes.height:,}")
    print(f"PyG edge count: {pyg_edges.height:,}")
    print(f"2024 token count: {token_count_by_year.get(2024, 0):,}")
    print(f"2025 token count: {token_count_by_year.get(2025, 0):,}")
    print(f"2024 wallet count: {wallet_count_by_year.get(2024, 0):,}")
    print(f"2025 wallet count: {wallet_count_by_year.get(2025, 0):,}")
    print("Output file paths:")
    for path in OUTPUT_PATHS:
        print(f"- {path}")


def main() -> None:
    pl = require_polars()
    ensure_output_dirs()
    raw_files = find_raw_files()
    validate_month_files(raw_files)
    log(f"Found {len(raw_files):,} raw Dune CSV files")

    bronze = build_bronze(raw_files, pl)
    events = build_silver_events(bronze, pl)
    edges = build_silver_edges(events, pl)
    token_features = build_gold_features(events, edges, pl)
    nodes, pyg_edges = build_pyg_tables(token_features, edges, pl)
    print_validation(raw_files, bronze, events, edges, token_features, nodes, pyg_edges, pl)
    print("Polars lakehouse pipeline completed successfully.")


if __name__ == "__main__":
    main()
