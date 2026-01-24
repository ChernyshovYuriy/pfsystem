from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScanResultRow:
    symbol: str
    regime: str
    score: float

    # PF summary for table + details panel
    pf_active_name: str  # e.g. "BUY_DOUBLE_TOP" or "none"
    pf_active_trigger: Optional[float]
    pf_active_cs: Optional[int]
    pf_buy_trigger: Optional[float]
    pf_buy_cs: Optional[int]
    pf_sell_trigger: Optional[float]
    pf_sell_cs: Optional[int]
    pf_cur_type: str
    pf_cur_low: float
    pf_cur_high: float
