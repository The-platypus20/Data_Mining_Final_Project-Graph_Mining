from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_rugcheck_ground_truth import (
    DEFAULT_RESULTS_DIR,
    find_token_col,
    flatten,
    save_outputs,
)


DEFAULT_INPUT = DEFAULT_RESULTS_DIR / "silver_labeled_token_features.csv"
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "rugcheck_external_labels_full.csv"
DEFAULT_GROUND_TRUTH_OUT = DEFAULT_RESULTS_DIR / "rugcheck_ground_truth_labels_full.csv"
DEFAULT_SUMMARY_OUT = DEFAULT_RESULTS_DIR / "rugcheck_cache_rebuild_summary.json"


def load_token_universe(input_path: Path) -> list[str]:
    frame = pd.read_csv(input_path)
    token_col = find_token_col(frame)
    tokens = frame[token_col].dropna().astype(str).str.strip()
    tokens = tokens[tokens.ne("")]
    return list(dict.fromkeys(tokens.tolist()))


def cache_token(path: Path) -> str | None:
    suffix = "_summary.json"
    if not path.name.endswith(suffix):
        return None
    token = path.name[: -len(suffix)].strip()
    return token or None


def discover_cache_files(cache_dirs: list[Path]) -> dict[str, Path]:
    cache_files: dict[str, Path] = {}
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        for path in cache_dir.glob("*_summary.json"):
            token = cache_token(path)
            if not token:
                continue
            cache_files[token] = path
    return cache_files


def read_cached_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rebuild(args: argparse.Namespace) -> dict[str, Any]:
    tokens = load_token_universe(args.input)
    token_set = set(tokens)
    cache_files = discover_cache_files(args.cache_dirs)

    rows: list[dict[str, Any]] = []
    skipped_not_in_universe = 0
    parse_errors = 0

    for token, cache_path in cache_files.items():
        if token not in token_set:
            skipped_not_in_universe += 1
            continue
        try:
            data = read_cached_payload(cache_path)
            row = flatten(token, data, cache_path.parent)
            row["raw_json_path"] = str(cache_path)
        except Exception as exc:
            parse_errors += 1
            row = {
                "token_address": token,
                "api_status": None,
                "api_ok": False,
                "risk_count": 0,
                "danger_count": 0,
                "warn_count": 0,
                "rugcheck_label": -1,
                "label_reason": f"cache_parse_error: {exc}",
                "raw_json_path": str(cache_path),
            }
        rows.append(row)

    output = pd.DataFrame(rows)
    if not output.empty:
        output["_token_order"] = output["token_address"].map({token: index for index, token in enumerate(tokens)})
        output = output.sort_values("_token_order").drop(columns=["_token_order"])
        rows = output.to_dict("records")

    save_outputs(rows, args.out, args.ground_truth_out)

    labels = pd.DataFrame(rows)
    summary = {
        "input": str(args.input),
        "cache_dirs": [str(path) for path in args.cache_dirs],
        "token_universe_count": len(tokens),
        "cached_token_count_total": len(cache_files),
        "cached_token_count_in_universe": len(rows),
        "skipped_not_in_universe": skipped_not_in_universe,
        "parse_errors": parse_errors,
        "usable_label_count": int(labels["rugcheck_label"].isin([0, 1]).sum()) if not labels.empty else 0,
        "rugcheck_risky_count": int(labels["rugcheck_label"].eq(1).sum()) if not labels.empty else 0,
        "rugcheck_safe_count": int(labels["rugcheck_label"].eq(0).sum()) if not labels.empty else 0,
        "unknown_count": int(labels["rugcheck_label"].eq(-1).sum()) if not labels.empty else 0,
        "out": str(args.out),
        "ground_truth_out": str(args.ground_truth_out),
    }
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild RugCheck label CSV from one or more JSON cache directories.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--cache-dir", dest="cache_dirs", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ground-truth-out", type=Path, default=DEFAULT_GROUND_TRUTH_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary = rebuild(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
