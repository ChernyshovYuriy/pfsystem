from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pf_system.domain.point_and_figure import PFChart


@dataclass(frozen=True)
class PFSignal:
    name: str  # e.g. "BUY_DOUBLE_TOP", "SELL_DOUBLE_BOTTOM"
    trigger: float  # breakout/breakdown level
    column_index: int  # index in chart.columns where it triggered


def last_double_top_buy(chart: PFChart, *, include_current: bool = False) -> Optional[PFSignal]:
    """
    Double Top Buy:
      An X column exceeds the high of the previous X column.

    To avoid 'always fresh' signals, default behavior excludes the last (current) column and
    evaluates only completed columns.
    """
    cols = chart.columns
    if len(cols) < 3:
        return None

    last_index = len(cols) - 1
    max_i = last_index if include_current else last_index - 1
    if max_i < 2:
        return None

    for i in range(max_i, 1, -1):
        if cols[i].col_type != "X":
            continue

        j = i - 2
        if j < 0 or cols[j].col_type != "X":
            continue

        prev_high = cols[j].high
        if cols[i].high > prev_high:
            return PFSignal(name="BUY_DOUBLE_TOP", trigger=prev_high, column_index=i)

    return None


def last_double_bottom_sell(chart: PFChart, *, include_current: bool = False) -> Optional[PFSignal]:
    """
    Double Bottom Sell:
      An O column falls below the low of the previous O column.

    Default behavior excludes the last (current) column and evaluates only completed columns.
    """
    cols = chart.columns
    if len(cols) < 3:
        return None

    last_index = len(cols) - 1
    max_i = last_index if include_current else last_index - 1
    if max_i < 2:
        return None

    for i in range(max_i, 1, -1):
        if cols[i].col_type != "O":
            continue

        j = i - 2
        if j < 0 or cols[j].col_type != "O":
            continue

        prev_low = cols[j].low
        if cols[i].low < prev_low:
            return PFSignal(name="SELL_DOUBLE_BOTTOM", trigger=prev_low, column_index=i)

    return None
