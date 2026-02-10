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
    # Cache used by the GUI to open a chart on double-click.
    # Kept intentionally flexible because we may store tuples/dicts (date+close)
    # in addition to the original List[float] shape.
    closes_cache: Dict[str, List[Any]]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "rows": [asdict(r) for r in self.rows],
        }
