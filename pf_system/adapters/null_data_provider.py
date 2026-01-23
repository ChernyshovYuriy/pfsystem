from __future__ import annotations

from datetime import date, timedelta
from typing import List, Sequence

from pf_system.ports.data_provider import Bar, MarketDataProvider


class NullMarketDataProvider(MarketDataProvider):
    """
    Stub provider for early GUI + plumbing.
    Returns synthetic, deterministic-ish bars so domain layer can run.
    """

    def get_daily_bars(self, symbol: str, lookback_days: int) -> Sequence[Bar]:
        today = date.today()
        bars: List[Bar] = []

        # Deterministic base per symbol for stable UI during dev
        seed = abs(hash(symbol)) % 10_000
        price = 10.0 + (seed % 1000) / 100.0

        for i in range(lookback_days):
            d = today - timedelta(days=(lookback_days - i))
            # pseudo movement
            price += ((seed % 7) - 3) * 0.01

            bars.append(
                Bar(
                    d=d,
                    open=price * 0.995,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    volume=1_000_000.0,
                )
            )

        return bars
