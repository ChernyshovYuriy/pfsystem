from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from pf_system.domain.models import ScanResultRow


@dataclass(frozen=True)
class ScanRequest:
    symbols: List[str]
    lookback_days: int


@dataclass(frozen=True)
class ScanResponse:
    rows: List[ScanResultRow]
    closes_cache: Dict[str, List[float]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "rows": [asdict(r) for r in self.rows],
        }
