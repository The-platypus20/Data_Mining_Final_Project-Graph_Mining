from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from build_rugcheck_ground_truth import (
    DEFAULT_CACHE,
    DEFAULT_GROUND_TRUTH_OUT,
    DEFAULT_OUT,
    DEFAULT_RESULTS_DIR,
    fetch_summary,
    flatten,
    has_cached_summary,
    load_tokens,
    save_outputs,
)
from evaluate_rugcheck_external_validation import (
    DEFAULT_CALIBRATION,
    DEFAULT_COVERAGE,
    DEFAULT_EXISTING_EVAL,
    DEFAULT_FEATURES,
    DEFAULT_GRAPHSAGE,
    DEFAULT_MASTER,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SUMMARY,
    DEFAULT_WEAK_CROSSTAB,
    build_master,
    calibrate_thresholds,
    coverage_table,
    evaluate_models,
    weak_crosstab,
    write_benchmark_splits,
)


DEFAULT_FULL_OUT = DEFAULT_RESULTS_DIR / "rugcheck_external_labels_full.csv"
DEFAULT_FULL_GROUND_TRUTH_OUT = DEFAULT_RESULTS_DIR / "rugcheck_ground_truth_labels_full.csv"
DEFAULT_PIPELINE_LOG = DEFAULT_RESULTS_DIR / "rugcheck_batch_pipeline_log.csv"


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def count_cached(tokens: list[str], cache_dir: Path) -> int:
    token_set = set(tokens)
    return sum(1 for token in cache_tokens(cache_dir) if token in token_set)


def cache_tokens(cache_dir: Path) -> set[str]:
    if not cache_dir.exists():
        return set()
    suffix = "_summary.json"
    tokens = set()
    for path in cache_dir.glob(f"*{suffix}"):
        name = path.name
        if name.endswith(suffix):
            tokens.add(name[: -len(suffix)])
    return tokens


def load_existing_rows(out_path: Path) -> list[dict[str, Any]]:
    if not out_path.exists():
        return []
    return pd.read_csv(out_path).to_dict("records")


def existing_output_tokens(existing_rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("token_address", "")).strip() for row in existing_rows if str(row.get("token_address", "")).strip()}


def merge_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        token = str(row.get("token_address", "")).strip()
        if token:
            merged[token] = row
    for row in new_rows:
        token = str(row.get("token_address", "")).strip()
        if token:
            merged[token] = row
    return list(merged.values())


def flatten_cached_token(mint: str, cache_dir: Path) -> dict[str, Any]:
    cache_path = cache_dir / f"{mint}_summary.json"
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    return flatten(mint, data, cache_dir)


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "unknown"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def fetch_and_flatten(mint: str, args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    was_cached = has_cached_summary(mint, args.cache_dir)
    data = fetch_summary(
        mint,
        args.cache_dir,
        cache_only_first=not args.no_cache_only_first,
        request_delay_seconds=args.delay,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    latency = time.perf_counter() - started_at
    return {
        "row": flatten(mint, data, args.cache_dir),
        "latency_seconds": latency,
        "api_status": data.get("_status_code"),
        "rugcheck_label": data.get("rugcheck_label"),
        "was_cached": was_cached,
    }


def run_crawl_batch(args: argparse.Namespace, tokens: list[str]) -> dict[str, Any]:
    batch_started_at = time.perf_counter()
    existing_rows = load_existing_rows(args.out)
    existing_tokens = existing_output_tokens(existing_rows)
    cached_tokens = cache_tokens(args.cache_dir)
    token_set = set(tokens)
    cached_tokens &= token_set
    done_tokens = existing_tokens | cached_tokens

    cached_but_missing_output = [token for token in tokens if token in cached_tokens and token not in existing_tokens]
    cached_rows = [flatten_cached_token(token, args.cache_dir) for token in cached_but_missing_output]
    if cached_rows:
        existing_rows = merge_rows(existing_rows, cached_rows)
        existing_tokens = existing_output_tokens(existing_rows)
        save_outputs(existing_rows, args.out, args.ground_truth_out)
        print(f"Hydrated {len(cached_rows)} cached tokens into output CSV before new requests.", flush=True)

    uncrawled_tokens = [token for token in tokens if token not in done_tokens]
    selected_tokens = uncrawled_tokens[: args.batch_size]

    print(
        f"Selected {len(selected_tokens)} uncrawled tokens for batch "
        f"({len(uncrawled_tokens)} uncrawled tokens remain before fetch).",
        flush=True,
    )
    print(f"ThreadPoolExecutor workers active: {args.workers}", flush=True)

    new_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    status_counts: Counter[str] = Counter()
    completed = 0
    total_selected = len(selected_tokens)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_and_flatten, token, args): token for token in selected_tokens}
        for future in as_completed(futures):
            token = futures[future]
            try:
                result = future.result()
                row = result["row"]
                latencies.append(float(result["latency_seconds"]))
                status_counts[str(result["api_status"])] += 1
            except Exception as exc:
                row = {
                    "token_address": token,
                    "api_status": None,
                    "api_ok": False,
                    "rugcheck_label": -1,
                    "label_reason": f"worker_error: {exc}",
                    "raw_json_path": str(args.cache_dir / f"{token}_summary.json"),
                }
                status_counts["worker_error"] += 1
            new_rows.append(row)
            completed += 1
            if completed % args.save_every == 0:
                rows = merge_rows(existing_rows, new_rows)
                save_outputs(rows, args.out, args.ground_truth_out)
                elapsed = time.perf_counter() - batch_started_at
                tokens_per_minute = completed / elapsed * 60 if elapsed > 0 else 0.0
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
                remaining_total = max(len(uncrawled_tokens) - completed, 0)
                eta_seconds = remaining_total / tokens_per_minute * 60 if tokens_per_minute > 0 else None
                print(
                    "Saved progress "
                    f"{completed}/{total_selected}; "
                    f"avg_latency={avg_latency:.2f}s; "
                    f"tokens_per_min={tokens_per_minute:.1f}; "
                    f"status_counts={dict(status_counts)}; "
                    f"ETA_remaining={format_duration(eta_seconds)}",
                    flush=True,
                )

    rows = merge_rows(existing_rows, new_rows)
    save_outputs(rows, args.out, args.ground_truth_out)
    labels = pd.DataFrame(rows)
    usable = labels["rugcheck_label"].isin([0, 1]) if "rugcheck_label" in labels else pd.Series(dtype=bool)
    elapsed = time.perf_counter() - batch_started_at
    tokens_per_minute = completed / elapsed * 60 if elapsed > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "processed_rows_this_output": int(len(labels)),
        "new_requests": int(len(selected_tokens)),
        "workers": int(args.workers),
        "avg_latency_seconds": float(avg_latency),
        "tokens_per_minute": float(tokens_per_minute),
        "api_status_counts": json.dumps(dict(status_counts), sort_keys=True),
        "uncrawled_remaining_after_batch": int(max(len(uncrawled_tokens) - len(selected_tokens), 0)),
        "usable_labels_in_output": int(usable.sum()) if not labels.empty else 0,
        "risky_in_output": int(labels["rugcheck_label"].eq(1).sum()) if "rugcheck_label" in labels else 0,
        "safe_in_output": int(labels["rugcheck_label"].eq(0).sum()) if "rugcheck_label" in labels else 0,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    eval_args = SimpleNamespace(
        features=args.features,
        rugcheck=args.out,
        graphsage=args.graphsage,
        window_hours=args.window_hours,
        master_out=args.master_out,
        existing_eval_out=args.existing_eval_out,
        calibration_out=args.calibration_out,
        coverage_out=args.coverage_out,
        weak_crosstab_out=args.weak_crosstab_out,
        split_dir=args.split_dir,
        summary_out=args.summary_out,
    )

    for path in [
        eval_args.master_out,
        eval_args.existing_eval_out,
        eval_args.calibration_out,
        eval_args.coverage_out,
        eval_args.weak_crosstab_out,
        eval_args.summary_out,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    master, model_rows = build_master(eval_args)
    master.to_csv(eval_args.master_out, index=False)

    coverage = coverage_table(master)
    coverage.to_csv(eval_args.coverage_out, index=False)
    weak = weak_crosstab(master)
    weak.to_csv(eval_args.weak_crosstab_out, index=False)
    existing_eval = evaluate_models(master)
    existing_eval.to_csv(eval_args.existing_eval_out, index=False)
    calibration = calibrate_thresholds(master)
    calibration.to_csv(eval_args.calibration_out, index=False)
    split_counts = write_benchmark_splits(master, eval_args.split_dir)

    usable = master[master["rugcheck_label"].isin([0, 1])]
    safe_count = int(usable["rugcheck_label"].eq(0).sum())
    risky_count = int(usable["rugcheck_label"].eq(1).sum())
    retrain_allowed = bool(len(usable) >= 500 and safe_count >= 100 and risky_count >= 100)
    summary = {
        "window_hours": eval_args.window_hours,
        "model_score_generation": model_rows,
        "coverage": coverage.to_dict("records"),
        "benchmark_split_counts": split_counts,
        "retrain_decision": {
            "usable_label_count": int(len(usable)),
            "rugcheck_safe_count": safe_count,
            "rugcheck_risky_count": risky_count,
            "meets_minimum_retrain_rule": retrain_allowed,
            "recommendation": "retrain_allowed" if retrain_allowed else "external_evaluation_only",
        },
        "outputs": {
            "master": str(eval_args.master_out),
            "existing_eval": str(eval_args.existing_eval_out),
            "threshold_calibration": str(eval_args.calibration_out),
            "coverage": str(eval_args.coverage_out),
            "weak_vs_rugcheck": str(eval_args.weak_crosstab_out),
            "split_dir": str(eval_args.split_dir),
        },
    }
    write_json(eval_args.summary_out, summary)
    return summary["retrain_decision"]


def append_log(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if log_path.exists():
        existing = pd.read_csv(log_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(log_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RugCheck crawl batches and rerun external evaluation after each batch.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS_DIR / "silver_labeled_token_features.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_FULL_OUT)
    parser.add_argument("--ground-truth-out", type=Path, default=DEFAULT_FULL_GROUND_TRUTH_OUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of uncached RugCheck API requests per batch.")
    parser.add_argument("--num-batches", type=int, default=1, help="How many crawl/evaluate cycles to run.")
    parser.add_argument("--workers", type=int, default=5, help="Parallel RugCheck request workers.")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--no-cache-only-first", action="store_true")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--graphsage", type=Path, default=DEFAULT_GRAPHSAGE)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--master-out", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--existing-eval-out", type=Path, default=DEFAULT_EXISTING_EVAL)
    parser.add_argument("--calibration-out", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--coverage-out", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--weak-crosstab-out", type=Path, default=DEFAULT_WEAK_CROSSTAB)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--pipeline-log", type=Path, default=DEFAULT_PIPELINE_LOG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.input, None)
    print(f"Token universe: {len(tokens)} unique tokens", flush=True)
    print(f"Cached before run: {count_cached(tokens, args.cache_dir)}", flush=True)

    for batch_index in range(1, args.num_batches + 1):
        args.current_batch = batch_index
        args.num_batches_display = args.num_batches
        before_cached = count_cached(tokens, args.cache_dir)
        print(f"Starting batch {batch_index}/{args.num_batches}; cached={before_cached}", flush=True)

        crawl_result = run_crawl_batch(args, tokens)
        after_cached = count_cached(tokens, args.cache_dir)
        print("Batch crawl complete. Running evaluation...", flush=True)
        retrain_decision = run_evaluation(args)

        log_row = {
            "batch": batch_index,
            "cached_before": before_cached,
            "cached_after": after_cached,
            **crawl_result,
            **retrain_decision,
        }
        append_log(args.pipeline_log, log_row)
        print(f"Batch {batch_index} done: {log_row}", flush=True)

        if after_cached >= len(tokens):
            print("All tokens have cached RugCheck summaries. Stopping.", flush=True)
            break


if __name__ == "__main__":
    main()
