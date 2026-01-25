from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from pf_system.gui.pf_chart_widget import PFChartWidget


class PFChartDialog(QDialog):
    def __init__(self, symbol: str, closes: List[float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle(f"{symbol} – Point & Figure")
        self.setMinimumSize(840, 520)

        title = QLabel(f"{symbol} – Point & Figure")
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

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(scroll, 1)
