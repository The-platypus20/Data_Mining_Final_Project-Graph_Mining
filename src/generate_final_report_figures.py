from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOLRPDS_DIR = PROJECT_ROOT / "data" / "raw" / "solrpds"
DUNE_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"
FIGURE_DIR = PROJECT_ROOT / "figures" / "final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"

DUNE_MASTER_PATH = DUNE_RESULTS_DIR / "rugcheck_model_validation_master.csv"
DUNE_SILVER_PATH = DUNE_RESULTS_DIR / "silver_labeled_token_features.csv"

SOLRPDS_FILES = {
    2021: SOLRPDS_DIR / "2021.csv",
    2022: SOLRPDS_DIR / "2022.csv",
    2023: SOLRPDS_DIR / "2023.csv",
    2024: SOLRPDS_DIR / "Jan_2024-Nov_2024.csv",
}

CATEGORY_ORDER = ["rug", "non-rug", "uncertain", "no RugCheck coverage"]
SOURCE_ORDER = ["SolRPDS 2021", "SolRPDS 2022", "SolRPDS 2023", "SolRPDS 2024", "Dune 2024", "Dune 2025"]


def log(message: str) -> None:
    print(f"[final-figures] {message}", flush=True)


def require_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    return plt


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def parse_solrpds_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed
    return pd.Series(pd.NaT, index=series.index)


def load_solrpds_tokens() -> pd.DataFrame:
    frames = []
    for year, path in SOLRPDS_FILES.items():
        if not path.exists():
            log(f"Skipping missing SolRPDS file: {path}")
            continue
        raw = pd.read_csv(path, low_memory=False)
        mint_col = first_existing(raw, ["MINT", "mint", "token_mint", "token_address"])
        status_col = first_existing(raw, ["INACTIVITY_STATUS", "inactivity_status"])
        first_col = first_existing(raw, ["FIRST_POOL_ACTIVITY_TIMESTAMP", "timestamp", "first_timestamp"])
        last_col = first_existing(raw, ["LAST_POOL_ACTIVITY_TIMESTAMP", "last_timestamp"])
        lifespan_col = first_existing(raw, ["lifespan_hours", "lifespan_days", "lifespan_min"])

        if mint_col is None:
            log(f"Skipping {path.name}: no token/mint column")
            continue

        frame = pd.DataFrame(
            {
                "source": "SolRPDS",
                "year": year,
                "token_address": raw[mint_col].astype(str),
            }
        )

        if status_col:
            status = raw[status_col].astype(str).str.lower()
            frame["label_category"] = np.select(
                [status.eq("inactive"), status.eq("active")],
                ["rug", "non-rug"],
                default="uncertain",
            )
        else:
            frame["label_category"] = "uncertain"

        if lifespan_col:
            lifespan = pd.to_numeric(raw[lifespan_col], errors="coerce")
            if lifespan_col == "lifespan_days":
                frame["lifespan_hours"] = lifespan * 24
            elif lifespan_col == "lifespan_min":
                frame["lifespan_hours"] = lifespan / 60
            else:
                frame["lifespan_hours"] = lifespan
        elif first_col and last_col:
            first_time = parse_solrpds_timestamps(raw[first_col])
            last_time = parse_solrpds_timestamps(raw[last_col])
            frame["lifespan_hours"] = (last_time - first_time).dt.total_seconds() / 3600
        else:
            frame["lifespan_hours"] = np.nan

        frame["lifespan_hours"] = pd.to_numeric(frame["lifespan_hours"], errors="coerce")
        frame = frame[frame["token_address"].ne("") & frame["token_address"].ne("nan")]
        frame = (
            frame.sort_values(["year", "token_address", "lifespan_hours"])
            .groupby(["source", "year", "token_address"], as_index=False)
            .agg(label_category=("label_category", label_priority), lifespan_hours=("lifespan_hours", "max"))
        )
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def label_priority(values: pd.Series) -> str:
    labels = set(values.dropna().astype(str))
    for label in ["rug", "non-rug", "uncertain", "no RugCheck coverage"]:
        if label in labels:
            return label
    return "uncertain"


def load_dune_tokens() -> pd.DataFrame:
    if DUNE_MASTER_PATH.exists():
        columns = [
            "token_address",
            "year",
            "window_hours",
            "rugcheck_label",
            "api_ok",
            "weak_label",
            "silver_label",
            "lifespan_hours",
        ]
        header = pd.read_csv(DUNE_MASTER_PATH, nrows=0).columns
        usecols = [column for column in columns if column in header]
        raw = pd.read_csv(DUNE_MASTER_PATH, usecols=usecols, low_memory=False)
        if "window_hours" in raw.columns:
            raw = raw[raw["window_hours"].eq(24)].copy()
        label = pd.to_numeric(raw.get("rugcheck_label"), errors="coerce")
        api_ok = raw.get("api_ok", pd.Series(False, index=raw.index)).astype(str).str.lower().isin(["true", "1"])
        raw["label_category"] = np.select(
            [api_ok & label.eq(1), api_ok & label.eq(0), api_ok & ~label.isin([0, 1])],
            ["rug", "non-rug", "uncertain"],
            default="no RugCheck coverage",
        )
    elif DUNE_SILVER_PATH.exists():
        raw = pd.read_csv(DUNE_SILVER_PATH, low_memory=False)
        weak = pd.to_numeric(raw.get("weak_label", raw.get("silver_label")), errors="coerce")
        raw["label_category"] = np.select([weak.eq(1), weak.eq(0)], ["rug", "non-rug"], default="uncertain")
    else:
        return pd.DataFrame()

    required = {"token_address", "year", "label_category"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        log(f"Skipping Dune tokens: missing {missing}")
        return pd.DataFrame()

    raw["source"] = "Dune"
    raw["lifespan_hours"] = pd.to_numeric(raw.get("lifespan_hours"), errors="coerce")
    raw = raw[raw["year"].isin([2024, 2025])].copy()
    raw["token_address"] = raw["token_address"].astype(str)
    return (
        raw.sort_values(["year", "token_address", "lifespan_hours"])
        .groupby(["source", "year", "token_address"], as_index=False)
        .agg(label_category=("label_category", label_priority), lifespan_hours=("lifespan_hours", "max"))
    )


def source_year_label(frame: pd.DataFrame) -> pd.Series:
    return frame["source"] + " " + frame["year"].astype(int).astype(str)


def plot_token_distribution(tokens: pd.DataFrame, plt) -> Path:
    tokens = tokens.copy()
    tokens["source_year"] = source_year_label(tokens)
    counts = (
        tokens.groupby(["source_year", "label_category"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=SOURCE_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    )
    counts.to_csv(OUTPUT_DIR / "token_distribution_by_source_year.csv")

    colors = {
        "rug": "#b8403a",
        "non-rug": "#2f7d59",
        "uncertain": "#d6a23f",
        "no RugCheck coverage": "#8d96a8",
    }
    fig, ax = plt.subplots(figsize=(11, 6.2))
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts.index))
    for category in CATEGORY_ORDER:
        values = counts[category].to_numpy()
        ax.bar(x, values, bottom=bottom, label=category, color=colors[category], width=0.72)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("Unique tokens")
    ax.set_title("Token Distribution by Source Year and Label Category")
    ax.legend(ncols=2, frameon=True)
    ax.margins(x=0.02)
    fig.tight_layout()
    out = FIGURE_DIR / "token_distribution_by_source_year.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean >= 0]
    if clean.empty:
        return np.array([]), np.array([])
    x = np.sort(clean.to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def load_dune_weak_lifespan() -> pd.DataFrame:
    if not DUNE_SILVER_PATH.exists():
        return pd.DataFrame()
    header = pd.read_csv(DUNE_SILVER_PATH, nrows=0).columns
    usecols = [column for column in ["token_address", "year", "weak_label", "silver_label", "lifespan_hours"] if column in header]
    frame = pd.read_csv(DUNE_SILVER_PATH, usecols=usecols, low_memory=False)
    label = pd.to_numeric(frame.get("weak_label", frame.get("silver_label")), errors="coerce")
    frame["label_category"] = np.select([label.eq(1), label.eq(0)], ["rug", "non-rug"], default="uncertain")
    frame["lifespan_hours"] = pd.to_numeric(frame.get("lifespan_hours"), errors="coerce")
    return frame.drop_duplicates(["year", "token_address"])


def build_lifespan_groups(solrpds: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    groups: list[tuple[str, pd.Series]] = []
    likely_rug = []
    likely_non_rug = []
    if not solrpds.empty:
        likely_rug.append(solrpds.loc[solrpds["label_category"].eq("rug"), "lifespan_hours"])
        likely_non_rug.append(solrpds.loc[solrpds["label_category"].eq("non-rug"), "lifespan_hours"])

    dune_weak = load_dune_weak_lifespan()
    if not dune_weak.empty:
        likely_rug.append(dune_weak.loc[dune_weak["label_category"].eq("rug"), "lifespan_hours"])
        likely_non_rug.append(dune_weak.loc[dune_weak["label_category"].eq("non-rug"), "lifespan_hours"])

    if likely_rug:
        groups.append(("Likely rug (SolRPDS + Dune weak)", pd.concat(likely_rug, ignore_index=True)))
    if likely_non_rug:
        groups.append(("Likely non-rug (SolRPDS + Dune weak)", pd.concat(likely_non_rug, ignore_index=True)))

    if DUNE_MASTER_PATH.exists():
        header = pd.read_csv(DUNE_MASTER_PATH, nrows=0).columns
        usecols = [column for column in ["token_address", "year", "window_hours", "rugcheck_label", "api_ok", "lifespan_hours"] if column in header]
        master = pd.read_csv(DUNE_MASTER_PATH, usecols=usecols, low_memory=False)
        if "window_hours" in master:
            master = master[master["window_hours"].eq(24)].copy()
        master["rugcheck_label"] = pd.to_numeric(master["rugcheck_label"], errors="coerce")
        master["lifespan_hours"] = pd.to_numeric(master["lifespan_hours"], errors="coerce")
        master = master.drop_duplicates(["year", "token_address"])
        groups.append(("RugCheck risky", master.loc[master["rugcheck_label"].eq(1), "lifespan_hours"]))
        groups.append(("RugCheck safe", master.loc[master["rugcheck_label"].eq(0), "lifespan_hours"]))
    return groups


def plot_lifespan_curve(solrpds: pd.DataFrame, dune: pd.DataFrame, plt) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6))
    rows = []
    colors = {
        "Likely rug (SolRPDS + Dune weak)": "#b8403a",
        "Likely non-rug (SolRPDS + Dune weak)": "#2f7d59",
        "RugCheck risky": "#7b2d26",
        "RugCheck safe": "#1f6f50",
    }
    line_styles = {
        "Likely rug (SolRPDS + Dune weak)": "-",
        "Likely non-rug (SolRPDS + Dune weak)": "-",
        "RugCheck risky": "--",
        "RugCheck safe": "--",
    }
    for label, values in build_lifespan_groups(solrpds):
        x, y = ecdf(values)
        if len(x) == 0:
            continue
        clipped_x = np.clip(x, 1e-6, None)
        ax.step(
            clipped_x,
            y,
            where="post",
            label=f"{label} (n={len(x):,})",
            color=colors.get(label),
            linestyle=line_styles.get(label, "-"),
            linewidth=2,
        )
        rows.append(
            {
                "group": label,
                "n": len(x),
                "median_lifespan_hours": float(np.median(x)),
                "p90_lifespan_hours": float(np.percentile(x, 90)),
            }
        )

    if not rows:
        raise RuntimeError("No usable lifespan_hours/lifespan_days data found for lifespan cumulative curve.")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "lifespan_cumulative_curve_summary.csv", index=False)
    ax.set_xscale("log")
    ax.set_xlabel("Lifespan hours, log scale")
    ax.set_ylabel("Cumulative share of tokens")
    ax.set_title("Cumulative Distribution of Token Lifespan")
    ax.text(
        0.99,
        0.04,
        "Zero-hour lifespans plotted at 1e-6 h for log scale.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    out = FIGURE_DIR / "lifespan_cumulative_curve.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt = require_matplotlib()

    solrpds = load_solrpds_tokens()
    dune = load_dune_tokens()
    if solrpds.empty and dune.empty:
        raise RuntimeError("No SolRPDS or Dune token data found.")

    all_tokens = pd.concat([frame for frame in [solrpds, dune] if not frame.empty], ignore_index=True)
    all_tokens.to_csv(OUTPUT_DIR / "figure_token_source_year_labels.csv", index=False)

    distribution_path = plot_token_distribution(all_tokens, plt)
    lifespan_path = plot_lifespan_curve(solrpds, dune, plt)
    log(f"Saved {distribution_path}")
    log(f"Saved {lifespan_path}")


if __name__ == "__main__":
    main()
