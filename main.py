from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from pf_system.adapters.null_repository import NullScanRepository
from pf_system.adapters.yfinance_provider import YFinanceMarketDataProvider
from pf_system.app.services import AppServices
from pf_system.app.use_cases import ScanMarketUseCase
from pf_system.config.settings import Settings
from pf_system.domain.scanner import DomainScanner
from pf_system.gui.main_window import MainWindow


def build_app(settings: Settings) -> MainWindow:
    # Ports/adapters
    data_provider = YFinanceMarketDataProvider(
        auto_adjust=True,
        cache_dir=".cache/yf",
        cache_ttl_seconds=6 * 60 * 60,
    )
    repo = NullScanRepository()

    # Domain
    scanner = DomainScanner(
        data_provider=data_provider,
        benchmark=settings.default_benchmark,
        enable_audit=False
    )

    # Use cases
    scan_uc = ScanMarketUseCase(scanner=scanner, repo=repo)

    # App service registry (optional but useful once you have multiple use cases)
    services = AppServices(scan_market=scan_uc, settings=settings)

    # GUI
    return MainWindow(services=services)


def main() -> int:
    settings = Settings()
    app = QApplication(sys.argv)
    window = build_app(settings)
    window.resize(1200, 720)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
