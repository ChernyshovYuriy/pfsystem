from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Scanning defaults
    default_lookback_days: int = 252

    # Placeholder for later (P&F defaults)
    box_mode: str = "percent"  # "percent" or "fixed"
    box_value: float = 0.02  # 1.5% (if percent) or e.g. 0.25 (if fixed)
    reversal: int = 3

    # Universe / benchmark placeholders
    default_benchmark: str = "XIU.TO"
