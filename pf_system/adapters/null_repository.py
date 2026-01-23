from __future__ import annotations

from typing import Any, Dict, Optional

from pf_system.ports.repository import ScanRepository


class NullScanRepository(ScanRepository):
    def save_scan(self, scan_payload: Dict[str, Any]) -> None:
        return

    def load_last_scan(self) -> Optional[Dict[str, Any]]:
        return None
