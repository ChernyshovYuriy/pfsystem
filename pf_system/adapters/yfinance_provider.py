from __future__ import annotations

import os
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Sequence

import pandas as pd
import yfinance as yf

from pf_system.ports.data_provider import Bar, MarketDataProvider


class YFinanceMarketDataProvider(MarketDataProvider):
    """
    Yahoo Finance adapter (via yfinance).

    Design goals:
    - Works for EOD scanning (daily bars)
    - Safe-ish on rate limiting (basic retry/backoff)
    - Optional cache to accelerate repeated scans
    - Returns Bars in oldest->newest order
    """

    def __init__(
            self,
            *,
            auto_adjust: bool = True,
            cache_dir: Optional[str] = None,
            cache_ttl_seconds: int = 6 * 60 * 60,  # 6 hours
            max_retries: int = 3,
            base_backoff_seconds: float = 0.75,
    ) -> None:
        self._auto_adjust = auto_adjust
        self._cache_dir = cache_dir
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_retries = max_retries
        self._base_backoff_seconds = base_backoff_seconds

        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)

    def get_daily_bars(self, symbol: str, lookback_days: int) -> Sequence[Bar]:
        if lookback_days <= 0:
            return []

        # For "lookback trading days", request a bigger calendar window.
        # This avoids issues with weekends/holidays and still keeps requests bounded.
        start = (datetime.now(timezone.utc) - timedelta(days=int(lookback_days * 2.2))).date()
        end = (datetime.now(timezone.utc) + timedelta(days=1)).date()

        cache_key = self._cache_key(symbol, lookback_days, start, end)
        cached = self._try_load_cache(cache_key)
        if cached is not None:
            return cached

        df = self._download_with_retry(symbol=symbol, start=start, end=end)
        bars = self._to_bars(df, lookback_days)
        self._try_save_cache(cache_key, bars)
        print(f"Data for {symbol} processed")
        return bars

    # -------------------------
    # Internal helpers
    # -------------------------

    def _download_with_retry(self, *, symbol: str, start: date, end: date) -> pd.DataFrame:
        last_err: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                df = yf.download(
                    tickers=symbol,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval="1d",
                    auto_adjust=self._auto_adjust,
                    progress=True,
                    threads=False,
                )

                if df is None or df.empty:
                    raise RuntimeError(f"yfinance returned empty data for {symbol}")

                # Normalize columns: yfinance returns columns like Open/High/Low/Close/Volume
                # Sometimes columns are lowercase or have spaces depending on versions.
                df = df.copy()

                # Case 1: MultiIndex columns like (Field, Ticker)
                if isinstance(df.columns, pd.MultiIndex):
                    # yfinance commonly returns columns as (price_field, ticker)
                    # We want a single-ticker dataframe with flat columns.
                    if symbol in df.columns.get_level_values(-1):
                        df = df.xs(symbol, axis=1, level=-1)
                    elif symbol in df.columns.get_level_values(0):
                        # Uncommon layout; try the other level
                        df = df.xs(symbol, axis=1, level=0)
                    else:
                        raise RuntimeError(f"MultiIndex columns but symbol {symbol} not present: {df.columns}")

                    df.columns = [str(c).strip() for c in df.columns]

                # Case 2: Flat columns but may be non-standard strings
                else:
                    df.columns = [str(c).strip() for c in df.columns]

                    # Sometimes columns arrive as strings like "('Open', 'XIU.TO')" (object repr)
                    # If so, attempt to parse and reduce them.
                    if any("', '" in c and c.startswith("('") and c.endswith("')") for c in df.columns):
                        reduced = []
                        for c in df.columns:
                            # crude but effective: take first element inside tuple repr
                            # "('Open', 'XIU.TO')" -> "Open"
                            try:
                                inner = c[2:-2]  # remove (" and ")
                                first = inner.split("', '", 1)[0]
                                reduced.append(first)
                            except Exception:
                                reduced.append(c)
                        df.columns = reduced

                required = {"Open", "High", "Low", "Close", "Volume"}
                if not required.issubset(set(df.columns)):
                    raise RuntimeError(f"Unexpected columns for {symbol}: {list(df.columns)}")

                # Drop rows with NaNs in price columns
                df = df.dropna(subset=["Open", "High", "Low", "Close"])

                if df.empty:
                    raise RuntimeError(f"No usable OHLC rows after cleaning for {symbol}")

                return df

            except Exception as e:
                print(f"Download with retry exception {e}")
                last_err = e
                if attempt == self._max_retries:
                    break

                # Exponential-ish backoff
                sleep_s = self._base_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_s)

        raise RuntimeError(f"Failed to download {symbol} after {self._max_retries} attempts: {last_err}")

    def _to_bars(self, df: pd.DataFrame, lookback_days: int) -> List[Bar]:
        # Ensure index is datetime-like
        if not isinstance(df.index, pd.DatetimeIndex):
            # Some edge cases: try coercion
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.dropna(subset=[df.index.name] if df.index.name else None)

        # Keep only the most recent lookback_days rows
        df = df.sort_index()
        if len(df) > lookback_days:
            df = df.iloc[-lookback_days:]

        bars: List[Bar] = []
        for ts, row in df.iterrows():
            # Convert timestamp to date (local exchange timezone is not critical for EOD bars)
            d = ts.date()

            bars.append(
                Bar(
                    d=d,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume", 0.0)),
                )
            )

        return bars

    # -------------------------
    # Cache
    # -------------------------

    def _cache_key(self, symbol: str, lookback_days: int, start: date, end: date) -> str:
        # Safe filename key
        safe_symbol = symbol.replace("/", "_").replace(":", "_").replace("^", "")
        adj = "adj1" if self._auto_adjust else "adj0"
        return f"{safe_symbol}__lb{lookback_days}__{start.isoformat()}__{end.isoformat()}__{adj}"

    def _cache_path(self, key: str) -> Optional[str]:
        if not self._cache_dir:
            return None
        return os.path.join(self._cache_dir, f"{key}.json")

    def _try_load_cache(self, key: str) -> Optional[List[Bar]]:
        path = self._cache_path(key)
        if not path or not os.path.exists(path):
            return None

        # TTL check
        age = time.time() - os.path.getmtime(path)
        if age > self._cache_ttl_seconds:
            return None

        try:
            payload = pd.read_json(path, orient="records")
            # Expected columns: d, open, high, low, close, volume
            bars: List[Bar] = []
            for _, r in payload.iterrows():
                bars.append(
                    Bar(
                        d=pd.to_datetime(r["d"]).date(),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["volume"]),
                    )
                )
            return bars
        except Exception:
            # Corrupt cache; ignore
            return None

    def _try_save_cache(self, key: str, bars: List[Bar]) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            df = pd.DataFrame([asdict(b) for b in bars])
            # Write atomically: write temp then replace
            tmp = f"{path}.tmp"
            df.to_json(tmp, orient="records", date_format="iso")
            os.replace(tmp, path)
        except Exception:
            return
