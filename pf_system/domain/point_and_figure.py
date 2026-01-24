from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PFColumn:
    col_type: str  # "X" or "O"
    boxes: List[float]  # filled box price levels (ascending for X, descending for O)

    @property
    def high(self) -> float:
        return max(self.boxes)

    @property
    def low(self) -> float:
        return min(self.boxes)


@dataclass(frozen=True)
class PFChart:
    box_mode: str  # "percent" or "fixed"
    box_value: float  # percent (e.g., 0.015) or fixed (e.g., 0.25)
    reversal: int
    columns: List[PFColumn]

    @property
    def current(self) -> PFColumn:
        return self.columns[-1]


def _round_to_box(price: float, box_mode: str, box_value: float, direction: str) -> float:
    """
    Map a price to the nearest box boundary (for initialization).
    direction: "down" or "up" (bias rounding).
    """
    if box_mode == "fixed":
        step = box_value
        if step <= 0:
            raise ValueError("box_value must be > 0 for fixed mode")
        q = price / step
        if direction == "down":
            return (int(q)) * step
        else:
            return (int(q) + (0 if abs(q - int(q)) < 1e-12 else 1)) * step

    # percent mode: boxes are multiplicative
    if box_value <= 0:
        raise ValueError("box_value must be > 0 for percent mode")
    # We approximate by using logarithms to find nearest step in multiplicative grid
    import math
    base = 1.0 + box_value
    k = math.log(price) / math.log(base)
    if direction == "down":
        return base ** math.floor(k)
    else:
        return base ** math.ceil(k)


def _next_box(price: float, box_mode: str, box_value: float, steps: int, direction: str) -> float:
    """Move `steps` boxes from a reference price."""
    if steps <= 0:
        return price

    if box_mode == "fixed":
        delta = box_value * steps
        return price + delta if direction == "up" else price - delta

    # percent
    base = 1.0 + box_value
    return price * (base ** steps) if direction == "up" else price / (base ** steps)


def build_pf_from_closes(
        closes: List[float],
        *,
        box_mode: str = "percent",
        box_value: float = 0.015,
        reversal: int = 3,
) -> PFChart:
    """
    Close-only P&F chart.

    Rules (standard):
    - Columns alternate X (up) and O (down)
    - Add boxes only when price moves >= 1 box beyond last box
    - Reverse only when price moves >= reversal boxes in opposite direction
    """
    if reversal < 1:
        raise ValueError("reversal must be >= 1")
    closes = [float(c) for c in closes if c and c > 0]
    if len(closes) < 2:
        return PFChart(box_mode, box_value, reversal, columns=[])

    # Initialize first box level around first close
    first = closes[0]
    # anchor = _round_to_box(first, box_mode, box_value, direction="down")

    # Determine initial direction from first meaningful move
    col_type: Optional[str] = None
    col_boxes: List[float] = []

    last_box = first

    def fill_up(from_price: float, to_price: float) -> List[float]:
        boxes = []
        p = from_price
        while True:
            nxt = _next_box(p, box_mode, box_value, 1, "up")
            if nxt <= to_price + 1e-12:
                boxes.append(nxt)
                p = nxt
            else:
                break
        return boxes

    def fill_down(from_price: float, to_price: float) -> List[float]:
        boxes = []
        p = from_price
        while True:
            nxt = _next_box(p, box_mode, box_value, 1, "down")
            if nxt >= to_price - 1e-12:
                boxes.append(nxt)
                p = nxt
            else:
                break
        return boxes

    # Find first column direction
    for px in closes[1:]:
        up1 = _next_box(last_box, box_mode, box_value, 1, "up")
        dn1 = _next_box(last_box, box_mode, box_value, 1, "down")

        if px >= up1:
            col_type = "X"
            col_boxes = [last_box] + fill_up(last_box, px)
            last_box = col_boxes[-1]
            break
        if px <= dn1:
            col_type = "O"
            col_boxes = [last_box] + fill_down(last_box, px)
            last_box = col_boxes[-1]
            break

    if col_type is None:
        # No movement large enough to form even one box
        return PFChart(box_mode, box_value, reversal, columns=[])

    columns: List[PFColumn] = [PFColumn(col_type, col_boxes)]

    # Process remaining prices
    for px in closes[1:]:
        cur = columns[-1]
        if cur.col_type == "X":
            # extend up?
            up1 = _next_box(cur.high, box_mode, box_value, 1, "up")
            if px >= up1:
                extra = fill_up(cur.high, px)
                if extra:
                    columns[-1] = PFColumn("X", cur.boxes + extra)
                continue

            # reverse down?
            rev_level = _next_box(cur.high, box_mode, box_value, reversal, "down")
            if px <= rev_level:
                # start new O column from one box below current high
                start = _next_box(cur.high, box_mode, box_value, 1, "down")
                new_boxes = [start] + fill_down(start, px)
                columns.append(PFColumn("O", new_boxes))
                continue

        else:  # "O"
            # extend down?
            dn1 = _next_box(cur.low, box_mode, box_value, 1, "down")
            if px <= dn1:
                extra = fill_down(cur.low, px)
                if extra:
                    columns[-1] = PFColumn("O", cur.boxes + extra)
                continue

            # reverse up?
            rev_level = _next_box(cur.low, box_mode, box_value, reversal, "up")
            if px >= rev_level:
                start = _next_box(cur.low, box_mode, box_value, 1, "up")
                new_boxes = [start] + fill_up(start, px)
                columns.append(PFColumn("X", new_boxes))
                continue

    return PFChart(box_mode, box_value, reversal, columns=columns)
