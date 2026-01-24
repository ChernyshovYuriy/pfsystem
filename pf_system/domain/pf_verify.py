from __future__ import annotations

from typing import List, Tuple

from pf_system.domain.point_and_figure import PFChart


def verify_pf_chart(chart: PFChart) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    if not chart.columns:
        problems.append("chart has no columns")
        return False, problems

    # 1) Alternation
    for i in range(1, len(chart.columns)):
        if chart.columns[i].col_type == chart.columns[i - 1].col_type:
            problems.append(f"columns do not alternate at index {i - 1}->{i}")

    # 2) Monotonic boxes + reasonable length
    for i, col in enumerate(chart.columns):
        if len(col.boxes) < 2:
            problems.append(f"column {i} has <2 boxes (unexpected for most liquid symbols)")

        if col.col_type == "X":
            for j in range(1, len(col.boxes)):
                if not (col.boxes[j] > col.boxes[j - 1]):
                    problems.append(f"column {i} X is not strictly increasing at box {j}")
                    break
        elif col.col_type == "O":
            for j in range(1, len(col.boxes)):
                if not (col.boxes[j] < col.boxes[j - 1]):
                    problems.append(f"column {i} O is not strictly decreasing at box {j}")
                    break
        else:
            problems.append(f"column {i} has invalid type {col.col_type}")

    return (len(problems) == 0), problems


def summarize_pf(chart: PFChart, last_n: int = 8) -> str:
    cols = chart.columns[-last_n:] if chart.columns else []
    parts = []
    for c in cols:
        parts.append(f"{c.col_type}:{c.low:.2f}->{c.high:.2f}({len(c.boxes)})")
    return " | ".join(parts)
