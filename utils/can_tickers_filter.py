#!/usr/bin/env python3
"""
canada_universe_filter.py

Production-grade universe filter for Canadian tickers using yfinance.
Input: list of tickers (.TO / .V / .CN etc.)
Output: ranked "tradable" tickers plus diagnostics (liquidity/activity/volatility/RS).

Key practices implemented:
- Chunked downloads to reduce request overhead
- Retry with backoff on transient failures
- Optional on-disk caching of yfinance responses (via requests_cache)
- Defensive data validation and NaN handling
- Per-exchange thresholds (TO/V/CN) and optional scoring mode
- Robust logging and CLI interface

Usage examples:
  python canada_universe_filter.py --tickers tickers.txt --out results.json
  python canada_universe_filter.py --tickers "BNS.TO,SHOP.TO,ACQ.V,ABC.CN" --out results.csv --format csv
  python canada_universe_filter.py --tickers tickers.txt --mode score --top 80

Notes:
- yfinance data quality varies, especially for .CN. This script explicitly penalizes
  missing/irregular prints and low dollar-volume names.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance. Install with: pip install yfinance") from e


# -----------------------------
# Optional caching (recommended)
# -----------------------------
def _install_requests_cache(cache_dir: str, expire_seconds: int) -> None:
    """
    Install a requests-cache layer so repeated runs don't hammer Yahoo endpoints.
    This is optional: script works without it.
    """
    try:
        import requests_cache  # type: ignore
    except Exception:
        return

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "yfinance_http_cache")
    # Cache both GET/POST to be safe, but yfinance mainly uses GET.
    requests_cache.install_cache(
        cache_name=cache_path,
        backend="sqlite",
        expire_after=expire_seconds,
        allowable_methods=("GET", "POST"),
        stale_if_error=True,
    )


# -----------------------------
# Threshold configuration
# -----------------------------
@dataclass(frozen=True)
class ExchangeThresholds:
    min_price: float
    min_avg_dollar_vol_30d: float
    min_active_days_ratio: float
    max_volume_cv: float
    adr_min: float
    adr_max: float


DEFAULT_THRESHOLDS: Dict[str, ExchangeThresholds] = {
    # TSX
    "TO": ExchangeThresholds(
        min_price=1.00,
        min_avg_dollar_vol_30d=1_500_000.0,
        min_active_days_ratio=0.95,
        max_volume_cv=4.0,
        adr_min=0.02,
        adr_max=0.12,
    ),
    # TSX Venture
    "V": ExchangeThresholds(
        min_price=0.50,
        min_avg_dollar_vol_30d=750_000.0,
        min_active_days_ratio=0.95,
        max_volume_cv=4.0,
        adr_min=0.02,
        adr_max=0.12,
    ),
    # CSE (stricter / many are illiquid)
    "CN": ExchangeThresholds(
        min_price=0.75,
        min_avg_dollar_vol_30d=1_000_000.0,
        min_active_days_ratio=0.98,  # stricter for CN
        max_volume_cv=4.0,
        adr_min=0.02,
        adr_max=0.12,
    ),
    # Fallback for unknown suffixes
    "OTHER": ExchangeThresholds(
        min_price=1.00,
        min_avg_dollar_vol_30d=1_500_000.0,
        min_active_days_ratio=0.95,
        max_volume_cv=4.0,
        adr_min=0.02,
        adr_max=0.12,
    ),
}


# -----------------------------
# Utility helpers
# -----------------------------
def parse_tickers_arg(tickers_arg: str) -> List[str]:
    """
    Accept either:
    - a file path containing tickers (one per line), OR
    - a comma-separated string of tickers.
    """
    if os.path.exists(tickers_arg) and os.path.isfile(tickers_arg):
        with open(tickers_arg, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    else:
        tickers = [t.strip() for t in tickers_arg.split(",") if t.strip()]

    # Normalize: Yahoo expects dashes for some TSX symbols, but users often input dots.
    # We do not "fix" symbols aggressively; just strip whitespace.
    tickers = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    return tickers


def ticker_suffix(ticker: str) -> str:
    """
    Determine exchange suffix (.TO / .V / .CN).
    Returns: "TO", "V", "CN", or "OTHER"
    """
    t = ticker.upper()
    if t.endswith(".TO"):
        return "TO"
    if t.endswith(".V"):
        return "V"
    if t.endswith(".CN"):
        return "CN"
    return "OTHER"


def safe_float(x: object) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def linear_slope(y: np.ndarray) -> float:
    """
    Compute slope of y over index [0..n-1] using least squares.
    Returns NaN if insufficient data.
    """
    if y.size < 3 or np.all(~np.isfinite(y)):
        return float("nan")
    x = np.arange(y.size, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    # slope = cov(x,y)/var(x)
    vx = np.var(x)
    if vx == 0:
        return float("nan")
    return float(np.cov(x, y, ddof=0)[0, 1] / vx)


# -----------------------------
# yfinance download logic
# -----------------------------
def yf_download_chunk(
        tickers: List[str],
        period: str,
        interval: str,
        auto_adjust: bool,
        threads: bool,
) -> pd.DataFrame:
    """
    Wrap yf.download to keep consistent parameters.
    """
    return yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=auto_adjust,
        actions=False,
        prepost=False,
        threads=threads,
        progress=True,
        timeout=30,
    )


def download_with_retries(
        tickers: List[str],
        period: str = "9mo",
        interval: str = "1d",
        auto_adjust: bool = True,
        threads: bool = True,
        chunk_size: int = 80,
        max_retries: int = 4,
        base_sleep: float = 0.8,
) -> pd.DataFrame:
    """
    Download in chunks with retries/backoff.
    Returns a multi-index style dataframe (ticker -> OHLCV columns) from yfinance.
    """
    frames: List[pd.DataFrame] = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i: i + chunk_size]

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                df = yf_download_chunk(
                    tickers=chunk,
                    period=period,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    threads=threads,
                )
                if df is None or df.empty:
                    raise RuntimeError("Empty dataframe returned by yfinance")
                frames.append(df)
                break
            except Exception as e:
                last_exc = e
                sleep_s = base_sleep * (2 ** attempt) + random.random() * 0.25
                logging.warning(
                    "Download chunk failed (attempt %d/%d) for %d tickers: %s. Sleeping %.2fs",
                    attempt + 1,
                    max_retries,
                    len(chunk),
                    str(e),
                    sleep_s,
                )
                time.sleep(sleep_s)
        else:
            logging.error("Failed to download chunk after retries. First ticker=%s. Error=%s", chunk[0], last_exc)
            # Continue with next chunks; we prefer partial results over total failure.

        # Gentle pacing to avoid throttling patterns
        time.sleep(0.15 + random.random() * 0.1)

    if not frames:
        return pd.DataFrame()

    # If multiple frames, align columns/index via concat on columns (same date index)
    # yfinance often returns single-level columns when one ticker; normalize later.
    try:
        out = pd.concat(frames, axis=1)
    except Exception:
        # Fallback: if indexes differ, outer join then sort
        out = pd.concat(frames, axis=1, join="outer").sort_index()

    return out


def extract_ticker_frame(df_all: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    yfinance returns either:
    - MultiIndex columns: (Ticker, Field) when multiple tickers
    - Single level columns: Field when one ticker
    This function normalizes to a standard OHLCV dataframe for one ticker.
    """
    if df_all is None or df_all.empty:
        return None

    # MultiIndex columns
    if isinstance(df_all.columns, pd.MultiIndex):
        if ticker not in df_all.columns.get_level_values(0):
            return None
        sub = df_all[ticker].copy()
        # expected columns: Open, High, Low, Close, Volume (Adj Close if auto_adjust=False)
        return sub

    # Single ticker case: columns are fields
    cols = [c.lower() for c in df_all.columns.astype(str)]
    if "close" in cols and "volume" in cols:
        return df_all.copy()

    return None


# -----------------------------
# Feature computation
# -----------------------------
def compute_metrics(
        ticker_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        lookback_rs_days: int = 63,
        lookback_adr_days: int = 20,
        lookback_dvol_days: int = 30,
) -> Dict[str, float]:
    """
    Compute:
    - last_close
    - avg_dollar_vol_30d
    - active_days_ratio (volume>0)
    - volume_cv
    - adr_20d
    - rs_value (close/bench close last)
    - rs_slope (slope of RS over lookback_rs_days)
    - mom_1m / mom_3m / mom_6m (simple returns)
    """
    df = ticker_df.copy()
    df = df.dropna(how="all")
    if df.empty:
        return {}

    # Normalize columns
    df.columns = [str(c).title() for c in df.columns]
    for needed in ("Close", "High", "Low", "Volume"):
        if needed not in df.columns:
            return {}

    # Clean: volume may be missing; treat missing as 0 for activity checks
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)

    last_close = float(close.dropna().iloc[-1]) if close.dropna().size else float("nan")

    # Avg dollar volume (close * volume)
    dvol = (close * vol).replace([np.inf, -np.inf], np.nan)
    avg_dvol_30d = float(dvol.tail(lookback_dvol_days).dropna().mean()) if dvol.notna().any() else float("nan")

    # Active days ratio
    total_days = float(len(vol.tail(lookback_dvol_days)))
    active_days_ratio = float((vol.tail(lookback_dvol_days) > 0).sum() / total_days) if total_days > 0 else float("nan")

    # Volume coefficient of variation (std/mean)
    v_tail = vol.tail(lookback_dvol_days)
    v_mean = float(v_tail.mean()) if len(v_tail) else float("nan")
    v_std = float(v_tail.std(ddof=0)) if len(v_tail) else float("nan")
    volume_cv = float(v_std / v_mean) if v_mean and v_mean > 0 else float("nan")

    # ADR (Average Daily Range as %)
    rng_pct = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    adr_20d = float(rng_pct.tail(lookback_adr_days).dropna().mean()) if rng_pct.notna().any() else float("nan")

    # Benchmark alignment
    b = benchmark_df.copy()
    b = b.dropna(how="all")
    b.columns = [str(c).title() for c in b.columns]
    if "Close" not in b.columns:
        return {}

    # Align on dates (inner join)
    rs = pd.concat([close.rename("sym"), b["Close"].rename("bench")], axis=1, join="inner").dropna()
    rs = rs.replace([np.inf, -np.inf], np.nan).dropna()
    if rs.empty:
        rs_value = float("nan")
        rs_slope = float("nan")
    else:
        rs_series = rs["sym"] / rs["bench"]
        rs_value = float(rs_series.iloc[-1])
        rs_slice = rs_series.tail(lookback_rs_days).to_numpy(dtype=float)
        rs_slope = linear_slope(rs_slice)

    # Simple momentum returns (approx trading days: 21/63/126)
    def _ret(n: int) -> float:
        s = close.dropna()
        if len(s) <= n:
            return float("nan")
        return float(s.iloc[-1] / s.iloc[-(n + 1)] - 1.0)

    mom_1m = _ret(21)
    mom_3m = _ret(63)
    mom_6m = _ret(126)

    return {
        "last_close": last_close,
        "avg_dollar_vol_30d": avg_dvol_30d,
        "active_days_ratio_30d": active_days_ratio,
        "volume_cv_30d": volume_cv,
        "adr_20d": adr_20d,
        "rs_value": rs_value,
        "rs_slope_63d": rs_slope,
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "mom_6m": mom_6m,
    }


# -----------------------------
# Filtering / scoring
# -----------------------------
def passes_hard_filters(m: Dict[str, float], th: ExchangeThresholds) -> Tuple[bool, List[str]]:
    """
    Hard reject rules. Returns (pass, reasons_if_failed).
    """
    reasons: List[str] = []
    if not m:
        return False, ["no_data"]

    lc = m.get("last_close", float("nan"))
    if not np.isfinite(lc) or lc < th.min_price:
        reasons.append(f"price<{th.min_price:g}")

    dvol = m.get("avg_dollar_vol_30d", float("nan"))
    if not np.isfinite(dvol) or dvol < th.min_avg_dollar_vol_30d:
        reasons.append(f"dvol30<{th.min_avg_dollar_vol_30d:,.0f}")

    adr = m.get("adr_20d", float("nan"))
    if not np.isfinite(adr) or adr < th.adr_min or adr > th.adr_max:
        reasons.append(f"adr20_outside[{th.adr_min:.2%},{th.adr_max:.2%}]")

    act = m.get("active_days_ratio_30d", float("nan"))
    if not np.isfinite(act) or act < th.min_active_days_ratio:
        reasons.append(f"active_days<{th.min_active_days_ratio:.0%}")

    cv = m.get("volume_cv_30d", float("nan"))
    if not np.isfinite(cv) or cv > th.max_volume_cv:
        reasons.append(f"volume_cv>{th.max_volume_cv:g}")

    # RS sanity: prefer improving RS, but do not hard reject on missing (Yahoo gaps happen).
    rs_slope = m.get("rs_slope_63d", float("nan"))
    if np.isfinite(rs_slope) and rs_slope < 0:
        reasons.append("rs_slope_negative")

    return (len(reasons) == 0), reasons


def score_metrics(m: Dict[str, float], th: ExchangeThresholds) -> float:
    """
    Composite quality score (0..100) designed for swing-trading universes.

    Weighting:
      40% Liquidity
      20% Activity
      20% Volatility sanity (ADR sweet spot)
      20% Relative strength + momentum blend

    This is intentionally conservative; missing data reduces score.
    """
    if not m:
        return 0.0

    # Liquidity score: log-scaled between threshold and ~10x threshold
    dvol = m.get("avg_dollar_vol_30d", float("nan"))
    if not np.isfinite(dvol) or dvol <= 0:
        s_liq = 0.0
    else:
        lo = th.min_avg_dollar_vol_30d
        hi = lo * 10.0
        x = min(max(dvol, lo), hi)
        s_liq = 100.0 * (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))

    # Activity score
    act = m.get("active_days_ratio_30d", float("nan"))
    if not np.isfinite(act):
        s_act = 0.0
    else:
        # map [min_active..1.0] to [0..100]
        s_act = 100.0 * min(max((act - th.min_active_days_ratio) / (1.0 - th.min_active_days_ratio + 1e-9), 0.0), 1.0)

    # ADR sanity score: peak at ~6%, penalize extremes
    adr = m.get("adr_20d", float("nan"))
    if not np.isfinite(adr):
        s_adr = 0.0
    else:
        target = 0.06
        # triangular score within [adr_min, adr_max]
        if adr < th.adr_min or adr > th.adr_max:
            s_adr = 0.0
        else:
            span = max(target - th.adr_min, th.adr_max - target)
            s_adr = 100.0 * (1.0 - min(abs(adr - target) / (span + 1e-9), 1.0))

    # RS + momentum blend
    rs_slope = m.get("rs_slope_63d", float("nan"))
    mom = np.nanmean([m.get("mom_1m", float("nan")), m.get("mom_3m", float("nan")), m.get("mom_6m", float("nan"))])
    if not np.isfinite(mom):
        mom = float("nan")

    # Normalize: mom in [-0.3..+0.6] typical swing band
    if np.isfinite(mom):
        s_mom = 100.0 * min(max((mom + 0.30) / 0.90, 0.0), 1.0)
    else:
        s_mom = 0.0

    # RS slope: normalize around 0 (negative -> 0..40, positive -> 40..100)
    if np.isfinite(rs_slope):
        # typical slopes are small; scale factor is heuristic
        scaled = rs_slope * 200.0
        s_rs = 50.0 + 50.0 * min(max(scaled, -1.0), 1.0)
    else:
        s_rs = 0.0

    s_rs_mom = 0.6 * s_rs + 0.4 * s_mom

    score = 0.40 * s_liq + 0.20 * s_act + 0.20 * s_adr + 0.20 * s_rs_mom
    return float(min(max(score, 0.0), 100.0))


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(
        tickers: List[str],
        benchmark: str,
        mode: str,
        top_n: int,
        cache_dir: Optional[str],
        cache_expire_seconds: int,
        period: str,
        interval: str,
        threads: bool,
        chunk_size: int,
) -> pd.DataFrame:
    if cache_dir:
        _install_requests_cache(cache_dir, cache_expire_seconds)

    # Download benchmark first (single ticker)
    logging.info("Downloading benchmark: %s", benchmark)
    bench_all = download_with_retries(
        tickers=[benchmark],
        period=period,
        interval=interval,
        auto_adjust=True,
        threads=False,
        chunk_size=1,
        max_retries=5,
        base_sleep=0.8,
    )
    bench_df = extract_ticker_frame(bench_all, benchmark)
    if bench_df is None or bench_df.empty:
        bench_df = bench_all
    if bench_df is None or bench_df.empty:
        raise RuntimeError("Benchmark download failed; cannot compute RS filters.")

    # Download universe
    logging.info("Downloading %d tickers (period=%s interval=%s)", len(tickers), period, interval)
    all_df = download_with_retries(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        threads=threads,
        chunk_size=chunk_size,
        max_retries=4,
        base_sleep=0.8,
    )
    if all_df is None or all_df.empty:
        raise RuntimeError("Universe download returned empty data.")

    rows: List[Dict[str, object]] = []
    for t in tickers:
        tdf = extract_ticker_frame(all_df, t)
        if tdf is None or tdf.empty:
            rows.append(
                {
                    "ticker": t,
                    "exchange": ticker_suffix(t),
                    "status": "reject",
                    "reasons": "no_data",
                    "score": 0.0,
                }
            )
            continue

        ex = ticker_suffix(t)
        th = DEFAULT_THRESHOLDS.get(ex, DEFAULT_THRESHOLDS["OTHER"])

        m = compute_metrics(tdf, bench_df)
        ok, reasons = passes_hard_filters(m, th)

        sc = score_metrics(m, th)

        if mode == "hard":
            status = "keep" if ok else "reject"
        else:
            # score mode: keep if score >= 60; you can tune this
            status = "keep" if sc >= 60.0 else "reject"
            if not ok:
                # keep diagnostic reasons even in score mode
                reasons = reasons or ["failed_hard_rules"]

        row: Dict[str, object] = {
            "ticker": t,
            "exchange": ex,
            "status": status,
            "score": round(sc, 2),
            "reasons": ",".join(reasons) if reasons else "",
        }
        for k, v in m.items():
            row[k] = float(v) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else (
                None if not np.isfinite(v) else float(v))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Rank kept tickers by score (desc), then by dollar volume as tie-break
    df["avg_dollar_vol_30d"] = pd.to_numeric(df.get("avg_dollar_vol_30d"), errors="coerce")
    df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0.0)

    kept = df[df["status"] == "keep"].copy()
    kept = kept.sort_values(by=["score", "avg_dollar_vol_30d"], ascending=[False, False])

    if top_n > 0:
        kept = kept.head(top_n)

    # Return with rejects appended for visibility (optional)
    # For trading you usually want only `kept`; here we output both but put kept first.
    rejects = df[df["status"] != "keep"].copy()
    out = pd.concat([kept, rejects], axis=0, ignore_index=True)
    return out


def write_output(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kept_count": int((df["status"] == "keep").sum()),
        "total_count": int(len(df)),
    }

    if fmt.lower() == "csv":
        df.to_csv(out_path, index=False)
        # Also write a small sidecar meta JSON
        meta_path = os.path.splitext(out_path)[0] + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return

    # json
    payload = {
        "meta": meta,
        "rows": df.to_dict(orient="records"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter and rank Canadian tickers (yfinance).")
    p.add_argument(
        "--tickers",
        default="res/can_tickers",
        help="Path to tickers.txt or comma-separated tickers."
    )
    p.add_argument(
        "--out",
        default="res/can_tickers_filtered",
        help="Output file path (json or csv)."
    )
    p.add_argument(
        "--benchmark",
        default="XIU.TO",
        help="Benchmark ticker for RS (default: XIU.TO)."
    )
    p.add_argument(
        "--mode",
        choices=["hard", "score"],
        default="hard",
        help="hard=must pass filters; score=keep score>=60 (default: hard)."
    )
    p.add_argument("--top", type=int, default=0, help="Return top N kept tickers first (0 = no limit).")
    p.add_argument("--period", default="9mo", help="yfinance period (default: 9mo).")
    p.add_argument("--interval", default="1d", help="yfinance interval (default: 1d).")
    p.add_argument("--threads", action="store_true", help="Enable yfinance threading (faster, may be less stable).")
    p.add_argument("--chunk-size", type=int, default=80, help="Tickers per yfinance download chunk (default: 80).")
    p.add_argument("--cache-dir", default=".cache_yf", help="Enable HTTP cache in this dir (default: .cache_yf).")
    p.add_argument("--cache-expire", type=int, default=6 * 3600, help="Cache TTL seconds (default: 6 hours).")
    p.add_argument("--format", choices=["json", "csv"], default="json", help="Output format (default: json).")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING/ERROR).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    tickers = parse_tickers_arg(args.tickers)
    if not tickers:
        raise SystemExit("No tickers provided.")

    # Basic normalization: strip and upper, preserve suffix case-insensitive
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = list(dict.fromkeys(tickers))

    print(f"Process {len(tickers)} tickers:{tickers}")

    # Run
    df = run_pipeline(
        tickers=tickers,
        benchmark=args.benchmark.strip().upper(),
        mode=args.mode,
        top_n=args.top,
        cache_dir=args.cache_dir if args.cache_dir else None,
        cache_expire_seconds=int(args.cache_expire),
        period=args.period,
        interval=args.interval,
        threads=bool(args.threads),
        chunk_size=int(args.chunk_size),
    )

    # Write
    write_output(df, args.out, args.format)

    kept = int((df["status"] == "keep").sum())
    total = len(df)
    logging.info("Done. Kept %d/%d tickers. Output: %s", kept, total, args.out)

    # Also print a short console summary of the kept universe
    kept_df = df[df["status"] == "keep"].copy()
    if not kept_df.empty:
        cols = ["ticker", "exchange", "score", "avg_dollar_vol_30d", "last_close", "adr_20d", "rs_slope_63d"]
        cols = [c for c in cols if c in kept_df.columns]
        print("\n=== TOP KEPT TICKERS (preview) ===")
        print(kept_df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
