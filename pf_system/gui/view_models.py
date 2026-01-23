from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pf_system.domain.models import ScanResultRow


@dataclass
class ScanViewState:
    rows: List[ScanResultRow]
    status: str = "Ready."
    is_busy: bool = False
