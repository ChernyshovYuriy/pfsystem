from __future__ import annotations

from urllib.parse import quote_plus

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


def stockcharts_pf_url(symbol: str, *, box_pct: float, reversal: int) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        return "https://stockcharts.com/"

    # params = {
    #     "symbol": normalized,
    #     "chart": "PNF",
    #     "boxmode": "percent",
    #     "box": f"{box_pct * 100:.2f}",
    #     "reversal": str(reversal),
    # }
    return "https://stockcharts.com/freecharts/pnf.php?c=" + normalized + "%2Cp"


def stockcharts_symbol_url(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        return "https://stockcharts.com/"
    return f"https://stockcharts.com/h-sc/ui?s={quote_plus(normalized)}"


def open_stockcharts_url(primary_url: str, fallback_url: str | None = None) -> bool:
    if QDesktopServices.openUrl(QUrl(primary_url)):
        return True
    if fallback_url:
        return QDesktopServices.openUrl(QUrl(fallback_url))
    return False
