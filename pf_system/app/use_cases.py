from __future__ import annotations

from pf_system.app.dto import ScanRequest, ScanResponse
from pf_system.domain.scanner import DomainScanner
from pf_system.ports.repository import ScanRepository


class ScanMarketUseCase:
    def __init__(self, scanner: DomainScanner, repo: ScanRepository) -> None:
        self._scanner = scanner
        self._repo = repo

    def execute(self, req: ScanRequest) -> ScanResponse:
        rows, closes_cache = self._scanner.scan(req.symbols, req.lookback_days)
        resp = ScanResponse(rows=rows, closes_cache=closes_cache)
        self._repo.save_scan(resp.to_payload())  # no-op for now
        return resp
