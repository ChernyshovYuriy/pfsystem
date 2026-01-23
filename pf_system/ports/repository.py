from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ScanRepository(ABC):
    @abstractmethod
    def save_scan(self, scan_payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_last_scan(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError
