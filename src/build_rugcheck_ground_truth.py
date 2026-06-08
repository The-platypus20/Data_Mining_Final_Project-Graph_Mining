from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "dune_publishable_2024_2025"
DEFAULT_INPUT = DEFAULT_RESULTS_DIR / "manual_validation_candidates_for_rugcheck_solscan.csv"
DEFAULT_OUT = DEFAULT_RESULTS_DIR / "rugcheck_external_labels.csv"
DEFAULT_GROUND_TRUTH_OUT = DEFAULT_RESULTS_DIR / "rugcheck_ground_truth_labels.csv"
DEFAULT_CACHE = DEFAULT_RESULTS_DIR / "rugcheck_cache"

TOKEN_COLUMNS = ["token_address", "mint", "token_mint", "token", "address"]
DANGER_LEVELS = {"danger", "critical", "high"}
WARN_LEVELS = {"warn", "warning", "medium"}


def find_token_col(df: pd.DataFrame) -> str:
    for col in TOKEN_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"No token column found. Columns: {df.columns.tolist()}")


def clean_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    token = str(value).strip()
    return token or None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def fetch_summary(
    mint: str,
    cache_dir: Path,
    *,
    cache_only_first: bool = True,
    request_delay_seconds: float = 0.5,
    timeout_seconds: int = 30,
    retries: int = 2,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{mint}_summary.json"
    if cache_path.exists():
        return read_json(cache_path)

    url = f"https://api.rugcheck.xyz/v1/tokens/{mint}/report/summary"
    attempts: list[dict[str, str] | None] = []
    if cache_only_first:
        attempts.append({"cacheOnly": "true"})
    attempts.append(None)

    last_response: requests.Response | None = None
    error_text: str | None = None

    for params in attempts:
        for retry_index in range(retries + 1):
            try:
                response = requests.get(url, params=params, timeout=timeout_seconds)
                last_response = response
            except requests.RequestException as exc:
                error_text = str(exc)
                if retry_index < retries:
                    time.sleep(min(20, 2 ** retry_index))
                continue

            if response.status_code == 429 and retry_index < retries:
                time.sleep(20)
                continue

            if response.status_code == 404 and params is not None:
                break

            data = response_to_payload(response)
            write_json(cache_path, data)
            time.sleep(request_delay_seconds)
            return data

    data = {
        "_status_code": last_response.status_code if last_response is not None else None,
        "_ok": False,
        "_text": last_response.text[:1000] if last_response is not None else error_text,
        "_request_error": error_text,
    }
    write_json(cache_path, data)
    time.sleep(request_delay_seconds)
    return data


def has_cached_summary(mint: str, cache_dir: Path) -> bool:
    return (cache_dir / f"{mint}_summary.json").exists()


def response_to_payload(response: requests.Response) -> dict[str, Any]:
    data: dict[str, Any] = {
        "_status_code": response.status_code,
        "_ok": response.ok,
        "_text": response.text[:1000] if not response.ok else None,
    }

    if response.ok:
        try:
            payload = response.json()
            if isinstance(payload, dict):
                data.update(payload)
            else:
                data["_parse_error"] = "RugCheck response is not a JSON object"
        except ValueError:
            data["_parse_error"] = True

    return data


def count_risks(risks: list[dict[str, Any]]) -> tuple[int, int]:
    danger_count = 0
    warn_count = 0

    for risk in risks:
        level = str(risk.get("level", "")).lower()
        if level in DANGER_LEVELS:
            danger_count += 1
        elif level in WARN_LEVELS:
            warn_count += 1

    return danger_count, warn_count


def make_label(data: dict[str, Any]) -> int:
    if not data.get("_ok"):
        return -1

    score = data.get("score")
    score_norm = data.get("score_normalised")
    risks = data.get("risks") or []
    danger_count, _ = count_risks(risks)

    if danger_count >= 1:
        return 1
    if score_norm is not None and score_norm >= 70:
        return 1
    if score is not None and score >= 10000:
        return 1

    if score_norm is not None and score_norm <= 40 and danger_count == 0:
        return 0
    if score is not None and danger_count == 0:
        return 0

    return -1


def label_reason(data: dict[str, Any]) -> str:
    if not data.get("_ok"):
        return "api_unavailable"

    score = data.get("score")
    score_norm = data.get("score_normalised")
    risks = data.get("risks") or []
    danger_count, _ = count_risks(risks)

    if danger_count >= 1:
        return "danger_or_high_risk"
    if score_norm is not None and score_norm >= 70:
        return "score_normalised_ge_70"
    if score is not None and score >= 10000:
        return "score_ge_10000"
    if score_norm is not None and score_norm <= 40:
        return "score_normalised_le_40_no_danger"
    if score is not None:
        return "score_available_no_danger"
    return "insufficient_signal"


def flatten(mint: str, data: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    risks = data.get("risks") or []
    danger_count, warn_count = count_risks(risks)

    return {
        "token_address": mint,
        "api_status": data.get("_status_code"),
        "api_ok": data.get("_ok"),
        "rugcheck_score": data.get("score"),
        "rugcheck_score_normalised": data.get("score_normalised"),
        "lpLockedPct": data.get("lpLockedPct"),
        "tokenProgram": data.get("tokenProgram"),
        "tokenType": data.get("tokenType"),
        "risk_count": len(risks),
        "danger_count": danger_count,
        "warn_count": warn_count,
        "risk_names": "|".join(str(risk.get("name", "")) for risk in risks),
        "risk_levels": "|".join(str(risk.get("level", "")) for risk in risks),
        "risk_scores": "|".join(str(risk.get("score", "")) for risk in risks),
        "rugcheck_label": make_label(data),
        "label_reason": label_reason(data),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "raw_json_path": str(cache_dir / f"{mint}_summary.json"),
    }


def load_tokens(input_path: Path, max_tokens: int | None) -> list[str]:
    df = pd.read_csv(input_path)
    token_col = find_token_col(df)

    tokens = [
        token
        for token in (clean_token(value) for value in df[token_col])
        if token is not None
    ]
    tokens = list(dict.fromkeys(tokens))
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def save_outputs(rows: list[dict[str, Any]], out_path: Path, ground_truth_out_path: Path) -> None:
    labels = pd.DataFrame(rows)
    labels.to_csv(out_path, index=False)

    if labels.empty:
        labels.to_csv(ground_truth_out_path, index=False)
        return

    ground_truth = labels[labels["rugcheck_label"].isin([0, 1])].copy()
    ground_truth = ground_truth.rename(columns={"rugcheck_label": "ground_truth_label"})
    ground_truth.to_csv(ground_truth_out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external RugCheck ground-truth labels for Solana token candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Candidate CSV with token address column.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Full RugCheck label output CSV.")
    parser.add_argument(
        "--ground-truth-out",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_OUT,
        help="Filtered output with only deterministic labels 0/1.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE, help="Directory for raw RugCheck JSON cache.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Optional cap on unique tokens to check.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based token offset after de-duplication.")
    parser.add_argument(
        "--max-new-requests",
        type=int,
        default=None,
        help="Stop after this many uncached RugCheck API requests; cached rows are still written.",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls in seconds.")
    parser.add_argument("--no-cache-only-first", action="store_true", help="Call live RugCheck report directly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(args.input, None)
    if args.start_index:
        tokens = tokens[args.start_index :]
    if args.max_tokens is not None:
        tokens = tokens[: args.max_tokens]
    rows: list[dict[str, Any]] = []
    new_request_count = 0

    for index, mint in enumerate(tokens, start=1):
        if args.max_new_requests is not None and not has_cached_summary(mint, args.cache_dir):
            if new_request_count >= args.max_new_requests:
                print(f"Reached --max-new-requests={args.max_new_requests}; stopping batch.", flush=True)
                break
            new_request_count += 1

        print(f"[{index}/{len(tokens)}] {mint}", flush=True)
        data = fetch_summary(
            mint,
            args.cache_dir,
            cache_only_first=not args.no_cache_only_first,
            request_delay_seconds=args.delay,
        )
        rows.append(flatten(mint, data, args.cache_dir))
        save_outputs(rows, args.out, args.ground_truth_out)

    save_outputs(rows, args.out, args.ground_truth_out)
    print(f"Saved full labels: {args.out}")
    print(f"Saved ground truth labels: {args.ground_truth_out}")


if __name__ == "__main__":
    main()
