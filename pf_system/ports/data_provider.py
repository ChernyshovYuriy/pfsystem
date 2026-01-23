from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class Bar:
    d: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(ABC):
    @abstractmethod
    def get_daily_bars(self, symbol: str, lookback_days: int) -> Sequence[Bar]:
        """Return daily bars (oldest -> newest)."""
        raise NotImplementedError
