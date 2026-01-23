from __future__ import annotations

from typing import List, Sequence

from pf_system.domain.models import ScanResultRow
from pf_system.ports.data_provider import MarketDataProvider


class DomainScanner:
    """
    Domain service.
    Today: stubbed regime/score.
    Later: build P&F, signals, trendlines, RS confirmation, scoring.
    """

    def __init__(self, data_provider: MarketDataProvider) -> None:
        self._data = data_provider

    def scan(self, symbols: Sequence[str], lookback_days: int) -> List[ScanResultRow]:
        out: List[ScanResultRow] = []

        num_if_tickers = len(symbols)
        for i, sym in enumerate(symbols):
            print(f"Scan {i}:{num_if_tickers}")
            # In the stub, we still call the provider to validate plumbing.
            _bars = self._data.get_daily_bars(sym, lookback_days)
            _ = _bars[-1].close if _bars else 0.0

            # Deterministic stub buckets
            if i % 3 == 0:
                out.append(ScanResultRow(sym, "BULLISH", 80.0 - i, "stub: bullish bucket"))
            elif i % 3 == 1:
                out.append(ScanResultRow(sym, "NEUTRAL", 50.0 - i, "stub: neutral bucket"))
            else:
                out.append(ScanResultRow(sym, "BEARISH", 20.0 - i, "stub: bearish bucket"))

        out.sort(key=lambda r: r.score, reverse=True)
        return out
