#!/usr/bin/env python3
"""
Universe Builder for swing trading (1–3 weeks) using yfinance.

Input:  text file with one ticker per line (Yahoo format: .TO / .V / .CN)
Output: ranked candidates + a plain ticker list for PFSystem

Run from IDE:
    from universe_builder import UniverseBuilderConfig, run_universe_builder
    run_universe_builder(UniverseBuilderConfig(...))
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class Thresholds:
    min_price: float = 1.0
    min_avg_dollar_vol_20: float = 1_000_000.0  # price*volume proxy
    max_atr_pct_14: float = 0.08  # 8% ATR
    max_one_day_drop_126: float = -0.15  # worst daily return must be >= -15%
    require_above_50d: bool = True  # hard gate for starter swings
    prefer_above_200d: bool = True  # ranking preference, not a hard gate


@dataclass
class UniverseBuilderConfig:
    tickers_path: str = "tickers.txt"
    out_prefix: str = "universe"
    period: str = "1y"
    interval: str = "1d"
    auto_adjust: bool = False
    batch_size: int = 80
    sleep_seconds: float = 1.0
    thresholds: Thresholds = field(default_factory=Thresholds)


def read_tickers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip().upper() for ln in f.readlines()]
    tickers = [t for t in lines if t and not t.startswith("#")]

    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def safe_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def slope_of_series(s: pd.Series, lookback: int = 10) -> float:
    s = s.dropna()
    if len(s) < lookback:
        return float("nan")

    y = s.iloc[-lookback:].values.astype(float)
    x = np.arange(len(y), dtype=float)

    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0:
        return float("nan")

    return float(((x - x.mean()) * (y - y.mean())).sum() / denom)


def score_row(row: Dict) -> float:
    """
    Higher is better: liquidity up, volatility down, trend up, avoid huge down days.
    Heuristic scoring; tune once you see outputs.
    """
    score = 0.0

    # Liquidity (log scale)
    adv = row.get("avg_dollar_vol_20", np.nan)
    if not np.isnan(adv) and adv > 0:
        score += min(5.0, math.log10(adv) - 5.0)  # 1e6=>~1, 1e8=>~3

    # Trend
    if row.get("above_50d", False):
        score += 1.0
    if row.get("above_200d", False):
        score += 0.8

    # Slope
    sma50_slope = row.get("sma50_slope", np.nan)
    if not np.isnan(sma50_slope) and sma50_slope > 0:
        score += 0.7

    # Volatility penalty
    atr_pct = row.get("atr_pct_14", np.nan)
    if not np.isnan(atr_pct):
        score -= min(2.0, atr_pct * 10.0)

    # Worst day penalty
    worst = row.get("worst_1d_ret_126", np.nan)
    if not np.isnan(worst):
        score -= min(2.5, abs(worst) * 10.0)

    return float(score)


def analyze_symbol(df: pd.DataFrame) -> Dict:
    out: Dict = {}

    close = df["Close"].dropna()
    vol = df["Volume"].dropna()
    if close.empty or vol.empty:
        out["error"] = "missing_close_or_volume"
        return out

    last_close = float(close.iloc[-1])
    out["last_close"] = last_close

    # Liquidity
    dv = (df["Close"] * df["Volume"]).dropna()
    out["avg_vol_20"] = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float("nan")
    out["avg_dollar_vol_20"] = float(dv.rolling(20).mean().iloc[-1]) if len(dv) >= 20 else float("nan")

    # Trend
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50"] = safe_last(sma50)
    out["sma200"] = safe_last(sma200)
    out["above_50d"] = bool(last_close > out["sma50"]) if not np.isnan(out["sma50"]) else False
    out["above_200d"] = bool(last_close > out["sma200"]) if not np.isnan(out["sma200"]) else False
    out["sma50_slope"] = slope_of_series(sma50, lookback=10)

    # ATR%
    atr = compute_atr(df, period=14)
    atr_last = safe_last(atr)
    out["atr_14"] = atr_last
    out["atr_pct_14"] = float(atr_last / last_close) if (not np.isnan(atr_last) and last_close > 0) else float("nan")

    # Worst 1-day return last ~6 months
    ret = close.pct_change()
    ret_126 = ret.dropna().iloc[-126:] if len(ret.dropna()) else ret.dropna()
    out["worst_1d_ret_126"] = float(ret_126.min()) if len(ret_126) else float("nan")

    return out


def pass_filters(row: Dict, th: Thresholds) -> Tuple[bool, List[str]]:
    reasons = []

    price = row.get("last_close", np.nan)
    if np.isnan(price) or price < th.min_price:
        reasons.append("price_too_low")

    adv = row.get("avg_dollar_vol_20", np.nan)
    if np.isnan(adv) or adv < th.min_avg_dollar_vol_20:
        reasons.append("low_dollar_volume")

    atr_pct = row.get("atr_pct_14", np.nan)
    if np.isnan(atr_pct) or atr_pct > th.max_atr_pct_14:
        reasons.append("too_volatile_atr")

    worst = row.get("worst_1d_ret_126", np.nan)
    if np.isnan(worst) or worst < th.max_one_day_drop_126:
        reasons.append("large_gap_risk")

    if th.require_above_50d and not row.get("above_50d", False):
        reasons.append("below_50d")

    return (len(reasons) == 0), reasons


def fetch_history_batch(tickers: List[str], period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        threads=True,
        progress=False
    )


def run_universe_builder(cfg: UniverseBuilderConfig) -> pd.DataFrame:
    tickers = read_tickers(cfg.tickers_path)
    print(f"Loaded {len(tickers)} tickers from {cfg.tickers_path}")

    rows: List[Dict] = []
    batches = chunked(tickers, cfg.batch_size)

    for i, batch in enumerate(batches, 1):
        print(f"Fetching batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            big = fetch_history_batch(batch, cfg.period, cfg.interval, cfg.auto_adjust)
        except Exception as e:
            print(f"Batch fetch error: {e}")
            time.sleep(cfg.sleep_seconds)
            continue

        if isinstance(big.columns, pd.MultiIndex):
            for sym in batch:
                if sym not in big.columns.get_level_values(0):
                    rows.append({"symbol": sym, "error": "no_data"})
                    continue

                sub = big[sym].dropna(how="all")
                if sub.empty:
                    rows.append({"symbol": sym, "error": "no_data"})
                    continue

                metrics = analyze_symbol(sub)
                metrics["symbol"] = sym
                ok, reasons = pass_filters(metrics, cfg.thresholds)
                metrics["tradable"] = ok
                metrics["reject_reasons"] = ",".join(reasons) if reasons else ""
                metrics["score"] = score_row(metrics) if ok else float("nan")
                rows.append(metrics)
        else:
            # single ticker case
            sym = batch[0]
            sub = big.dropna(how="all")
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if sub.empty:
                rows.append({"symbol": sym, "error": "no_data"})
            elif not needed.issubset(set(sub.columns)):
                rows.append({"symbol": sym, "error": "missing_ohlcv"})
            else:
                metrics = analyze_symbol(sub)
                metrics["symbol"] = sym
                ok, reasons = pass_filters(metrics, cfg.thresholds)
                metrics["tradable"] = ok
                metrics["reject_reasons"] = ",".join(reasons) if reasons else ""
                metrics["score"] = score_row(metrics) if ok else float("nan")
                rows.append(metrics)

        time.sleep(cfg.sleep_seconds)

    df = pd.DataFrame(rows)

    cols = [
        "symbol",
        "tradable",
        "score",
        "last_close",
        "avg_vol_20",
        "avg_dollar_vol_20",
        "atr_pct_14",
        "worst_1d_ret_126",
        "above_50d",
        "above_200d",
        "sma50_slope",
        "reject_reasons",
        "error",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    trad = df[df["tradable"] == True].copy()
    trad = trad.sort_values(["score", "avg_dollar_vol_20"], ascending=[False, False])

    out_json = f"{cfg.out_prefix}_candidates.json"
    out_txt = f"{cfg.out_prefix}_tickers.txt"
    out_one_line = f"{cfg.out_prefix}_tickers_oneline.txt"

    trad.to_json(out_json, orient="records", indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        for sym in trad["symbol"].tolist():
            f.write(sym + "\n")

    with open(out_one_line, "w", encoding="utf-8") as f:
        f.write(",".join(trad["symbol"].tolist()) + "\n")

    print("\nWrote:")
    print(f"  {out_json}  (tradables ranked)")
    print(f"  {out_txt}   (just tickers for PFSystem)")
    print(f"  {out_one_line}  (comma-separated tickers)")

    print("\nTop 20 tradable:")
    print(trad.head(20).to_string(index=False))

    return df


# IDE-friendly entry point
if __name__ == "__main__":
    config = UniverseBuilderConfig(
        tickers_path="res/can_tickers",
        out_prefix="cad_swing",
        period="1y",
        interval="1d",
        auto_adjust=False,
        batch_size=80,
        sleep_seconds=1.0,
        thresholds=Thresholds(
            min_price=1.0,
            min_avg_dollar_vol_20=1_000_000.0,
            max_atr_pct_14=0.08,
            max_one_day_drop_126=-0.15,
            require_above_50d=True,
            prefer_above_200d=True,
        )
    )
    run_universe_builder(config)
