from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pf_system.config.settings import Settings
from pf_system.domain.models import ScanResultRow
from pf_system.gui.pf_chart_widget import PFChartWidget
from pf_system.gui.stockcharts import open_stockcharts_url, stockcharts_pf_url, stockcharts_symbol_url


class PFChartDialog(QDialog):
    def __init__(
        self,
        row: ScanResultRow,
        closes: List[float],
        lookback_days: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle(f"{row.symbol} – Point & Figure")
        self.setMinimumSize(840, 520)

        title = QLabel(f"{row.symbol} – Point & Figure")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(close_btn)

        chart = PFChartWidget(closes)
        scroll = QScrollArea()
        scroll.setWidget(chart)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("background: transparent;")

        info_panel = self._build_info_panel(row, lookback_days)
        info_panel.setFixedWidth(360)

        body = QHBoxLayout()
        body.addWidget(scroll, 1)
        body.addWidget(info_panel, 0)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(body, 1)

    def _build_info_panel(self, row: ScanResultRow, lookback_days: int) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        heading = QLabel("Info")
        heading.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(heading)

        details = QFormLayout()
        details.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        details.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        details.setHorizontalSpacing(12)
        details.setVerticalSpacing(6)

        details.addRow("Symbol", QLabel(row.symbol))
        details.addRow("Regime", QLabel(row.regime))
        details.addRow("Score", QLabel(f"{row.score:.2f}"))
        details.addRow("Lookback", QLabel(f"{lookback_days} days"))

        active_trigger = "—" if row.pf_active_trigger is None else f"{row.pf_active_trigger:.2f}"
        active_cs = "—" if row.pf_active_cs is None else str(row.pf_active_cs)
        active_summary = QLabel(f"{row.pf_active_name} (trigger {active_trigger}, CS {active_cs})")
        active_summary.setWordWrap(True)
        details.addRow("Active signal", active_summary)

        cur_range = QLabel(f"{row.pf_cur_type} {row.pf_cur_low:.2f} → {row.pf_cur_high:.2f}")
        details.addRow("Current column", cur_range)

        layout.addLayout(details)

        layout.addWidget(self._build_divider())

        settings = Settings()
        primary_url = stockcharts_pf_url(
            row.symbol,
            box_pct=settings.box_value,
            reversal=settings.reversal,
        )
        fallback_url = stockcharts_symbol_url(row.symbol)

        url_label = QLabel(primary_url)
        url_label.setWordWrap(True)
        url_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)

        open_btn = QPushButton("Open original chart")
        open_btn.clicked.connect(lambda: open_stockcharts_url(primary_url, fallback_url))

        copy_btn = QPushButton("Copy URL")
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(primary_url))

        layout.addWidget(open_btn)
        layout.addWidget(QLabel("StockCharts URL"))
        layout.addWidget(url_label)
        layout.addWidget(copy_btn)

        layout.addWidget(self._build_divider())

        notes_title = QLabel("Notes")
        notes_title.setStyleSheet("font-weight: 600;")
        notes_body = QLabel("Add notes or signals here (coming soon).")
        notes_body.setWordWrap(True)

        layout.addWidget(notes_title)
        layout.addWidget(notes_body)
        layout.addStretch(1)

        return panel

    @staticmethod
    def _build_divider() -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet("color: #333;")
        return divider
