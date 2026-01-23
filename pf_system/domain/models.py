from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanResultRow:
    symbol: str
    regime: str  # "BULLISH" | "BEARISH" | "NEUTRAL"
    score: float
    note: str
