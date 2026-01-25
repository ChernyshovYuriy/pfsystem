#!/usr/bin/env python3
"""
walk_universe_json.py

Walks through the universe-filter output JSON and produces
trade-oriented summaries.

Safe for Python 3.12+, no pandas truthiness issues.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from typing import Dict, List, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid JSON: root must be an object")

    if "rows" not in data or not isinstance(data["rows"], list):
        raise ValueError("Invalid JSON: missing 'rows' array")

    return data


def get_kept(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("status") == "keep"]


def summarize_exchanges(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for r in rows:
        ex = r.get("exchange", "UNKNOWN")
        counts[ex] += 1
    return dict(counts)


def summarize_rejects(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    reasons = Counter()
    for r in rows:
        if r.get("status") != "keep":
            reason_str = r.get("reasons", "")
            for reason in reason_str.split(","):
                if reason:
                    reasons[reason] += 1
    return dict(reasons)


def top_ranked(
        rows: List[Dict[str, Any]],
        limit: int = 20,
) -> List[Dict[str, Any]]:
    kept = get_kept(rows)
    kept_sorted = sorted(
        kept,
        key=lambda r: (r.get("score", 0.0), r.get("avg_dollar_vol_30d", 0.0)),
        reverse=True,
    )
    return kept_sorted[:limit]


def get_kept(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return rows with status == 'keep'.

    Defensive against:
    - missing keys
    - non-dict entries
    - unexpected status values
    """
    kept: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        if row.get("status") != "keep":
            continue

        # Minimal validation: ticker must exist
        ticker = row.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            continue

        kept.append(row)

    return kept


def print_overview(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    total = len(rows)
    kept = sum(1 for r in rows if r.get("status") == "keep")

    print("\n=== UNIVERSE OVERVIEW ===")
    print(f"Generated (UTC): {meta.get('generated_utc')}")
    print(f"Total tickers:   {total}")
    print(f"Tradable kept:   {kept}")
    print(f"Rejected:        {total - kept}")


def print_exchange_breakdown(rows: List[Dict[str, Any]]) -> None:
    counts = summarize_exchanges(rows)
    print("\n=== EXCHANGE BREAKDOWN (KEPT) ===")
    for ex, cnt in sorted(counts.items()):
        print(f"{ex:>4}: {cnt}")


def print_reject_reasons(rows: List[Dict[str, Any]], top_n: int = 10) -> None:
    reasons = summarize_rejects(rows)
    if not reasons:
        return

    print("\n=== TOP REJECTION REASONS ===")
    for reason, cnt in sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"{reason:30s} {cnt}")


def get_elite_universe(
        rows: list[dict],
        min_score: float = 90.0,
) -> list[dict]:
    """
    Return only top-tier tradable tickers.
    """
    elite = []

    for r in rows:
        if r.get("status") != "keep":
            continue

        score = r.get("score")
        if not isinstance(score, (int, float)):
            continue

        if score >= min_score:
            elite.append(r)

    return elite


def print_tradable_tickers(
        rows: List[Dict[str, Any]],
        limit: int | None = None,
) -> None:
    """
    Print tradable tickers (status == 'keep').

    Sorted by:
      1) score (desc)
      2) avg_dollar_vol_30d (desc)

    Args:
        rows: list of row dicts from results.json
        limit: optional max number of rows to print
    """
    kept = get_kept(rows)

    if not kept:
        print("No tradable tickers found.")
        return

    def sort_key(r: Dict[str, Any]) -> tuple:
        score = r.get("score")
        dvol = r.get("avg_dollar_vol_30d")

        score = float(score) if isinstance(score, (int, float)) else 0.0
        dvol = float(dvol) if isinstance(dvol, (int, float)) else 0.0

        return score, dvol

    kept_sorted = sorted(kept, key=sort_key, reverse=True)

    if limit is not None and limit > 0:
        kept_sorted = kept_sorted[:limit]

    print(f"\n=== TRADABLE TICKERS ({len(kept_sorted)}) ===")

    tickers = ""
    for i, r in enumerate(kept_sorted, start=1):
        ticker = r.get("ticker", "N/A")
        exchange = r.get("exchange", "??")

        tickers += ticker + ","

        score = r.get("score", 0.0)
        dvol = r.get("avg_dollar_vol_30d", 0.0)
        price = r.get("last_close")
        adr = r.get("adr_20d")

        try:
            score_s = f"{float(score):5.1f}"
        except Exception:
            score_s = "  n/a"

        try:
            dvol_s = f"${float(dvol):,.0f}"
        except Exception:
            dvol_s = "$0"

        price_s = f"{price:.2f}" if isinstance(price, (int, float)) else "n/a"
        adr_s = f"{adr:.2%}" if isinstance(adr, (int, float)) else "n/a"

        print(
            f"{i:>2}. {ticker:8s} "
            f"{exchange:>2} "
            f"score={score_s} "
            f"dvol30={dvol_s:>12s} "
            f"price={price_s:>7s} "
            f"adr20={adr_s:>7s}"
        )

    print(tickers)


def print_top_tickers(rows: List[Dict[str, Any]], limit: int) -> None:
    top = top_ranked(rows, limit)

    print(f"\n=== TOP {len(top)} TRADE CANDIDATES ===")
    for i, r in enumerate(top, start=1):
        print(
            f"{i:>2}. {r['ticker']:8s} "
            f"{r['exchange']:>2}  "
            f"score={r.get('score', 0):5.1f}  "
            f"dvol30=${r.get('avg_dollar_vol_30d', 0):,.0f}  "
            f"adr20={r.get('adr_20d', 0):.2%}  "
            f"rs_slope={r.get('rs_slope_63d', 0):+.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Walk and summarize universe JSON output")
    p.add_argument(
        "--json",
        default="res/can_tickers_filtered",
        help="Path to results.json"
    )
    p.add_argument(
        "--top",
        type=int,
        default=200,
        help="Show top N tickers (default: 20)"
    )
    args = p.parse_args()

    try:
        data = load_json(args.json)
    except Exception as e:
        print(f"Error loading JSON: {e}", file=sys.stderr)
        sys.exit(1)

    meta = data.get("meta", {})
    rows = data["rows"]

    # print_overview(meta, rows)
    # print_exchange_breakdown(get_kept(rows))
    # print_reject_reasons(rows)
    # print_top_tickers(rows, args.top)

    elite = get_elite_universe(rows, min_score=90.0)
    print_tradable_tickers(elite)


if __name__ == "__main__":
    main()
