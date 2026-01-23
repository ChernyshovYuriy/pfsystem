from __future__ import annotations

from dataclasses import dataclass

from pf_system.app.use_cases import ScanMarketUseCase
from pf_system.config.settings import Settings


@dataclass(frozen=True)
class AppServices:
    scan_market: ScanMarketUseCase
    settings: Settings
