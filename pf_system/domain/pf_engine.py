from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pf_system.domain.point_and_figure import PFChart
from pf_system.domain.signals import PFSignal, last_double_top_buy, last_double_bottom_sell


@dataclass(frozen=True)
class PFRegimeResult:
    regime: str
    score: float
    signal: Optional[PFSignal]
    cols_since: Optional[int]
    buy_cs: Optional[int]
    sell_cs: Optional[int]
    note: str


def evaluate_pf(
        chart: PFChart,
        *,
        last_price: float,
        box_value: float,
) -> PFRegimeResult:
    if not chart.columns:
        return PFRegimeResult("NEUTRAL", 0.0, None, None, None, None, "pf: empty")

    buy = last_double_top_buy(chart)
    sell = last_double_bottom_sell(chart)

    cur_idx = len(chart.columns) - 1
    buy_cs = (cur_idx - buy.column_index) if buy is not None else None
    sell_cs = (cur_idx - sell.column_index) if sell is not None else None

    sig = _pick_active_signal(chart, buy, sell)

    cur_type = chart.current.col_type
    cols_since = None
    if sig is not None:
        cols_since = (len(chart.columns) - 1) - sig.column_index

    regime = _regime_from_signal(cur_type, sig)

    # If signal doesn't match current column -> neutral posture; don't score off that signal.
    sig_for_score = sig if regime != "NEUTRAL" else None
    cols_since_for_score = cols_since if regime != "NEUTRAL" else None

    score = _pf_score(
        regime=regime,
        sig=sig_for_score,
        cols_since=cols_since_for_score,
        last_price=last_price,
        box_value=box_value,
    )

    note = _pf_note(chart, regime, sig, cols_since, last_price, buy, sell)

    return PFRegimeResult(regime, score, sig, cols_since, buy_cs, sell_cs, note)


def _pick_active_signal(chart: PFChart, buy: Optional[PFSignal], sell: Optional[PFSignal]) -> Optional[PFSignal]:
    cur = chart.current.col_type

    if cur == "X":
        return buy if buy is not None else sell
    else:  # "O"
        return sell if sell is not None else buy


def _regime_from_signal(cur_type: str, sig: Optional[PFSignal]) -> str:
    if sig is None:
        return "NEUTRAL"
    if sig.name.startswith("BUY") and cur_type == "X":
        return "BULLISH"
    if sig.name.startswith("SELL") and cur_type == "O":
        return "BEARISH"
    return "NEUTRAL"


def _pf_score(
        *,
        regime: str,
        sig: Optional[PFSignal],
        cols_since: Optional[int],
        last_price: float,
        box_value: float,
) -> float:
    if regime == "NEUTRAL":
        return 0.0

    base = 60.0 if regime == "BULLISH" else -60.0

    fresh = 0.0
    if cols_since is not None:
        fresh = max(0.0, 12.0 - 2.0 * float(cols_since))  # 12,10,8,...,0

    penalty = 0.0
    magnitude = 0.0

    if sig is not None and sig.trigger > 0 and box_value > 0:
        if sig.name.startswith("BUY"):
            distance_pct = (last_price / sig.trigger) - 1.0
        else:
            distance_pct = (sig.trigger / last_price) - 1.0

        distance_boxes = distance_pct / box_value

        # Reward up to 5 boxes post-trigger (0..15 points)
        magnitude = min(5.0, max(0.0, distance_boxes)) * 3.0

        # Penalize overextension beyond 2 boxes (0..25 points)
        if distance_boxes > 2.0:
            penalty = min(25.0, (distance_boxes - 2.0) * 3.0)

    if regime == "BULLISH":
        return base + 3.0 * fresh + magnitude - penalty
    else:
        return base - 3.0 * fresh - magnitude + penalty


def _pf_note(
        chart: PFChart,
        regime: str,
        active: Optional[PFSignal],
        cols_since: Optional[int],
        last_price: float,
        buy: Optional[PFSignal],
        sell: Optional[PFSignal],
) -> str:
    cur = chart.current

    def fmt(s: Optional[PFSignal]) -> str:
        if s is None:
            return "na"
        cs = (len(chart.columns) - 1) - s.column_index
        return f"{s.name}@{s.trigger:.2f}(cs={cs})"

    active_s = "none" if active is None else f"{active.name}@{active.trigger:.2f}(cs={cols_since})"

    return (
        f"pf: regime={regime} active={active_s} "
        f"buy={fmt(buy)} sell={fmt(sell)} "
        f"| cur={cur.col_type} {cur.low:.2f}->{cur.high:.2f} last={last_price:.2f}"
    )
