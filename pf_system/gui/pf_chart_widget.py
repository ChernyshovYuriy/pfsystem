from __future__ import annotations

from typing import List

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from pf_system.domain.point_and_figure import PFChart, build_pf_from_closes


class PFChartWidget(QWidget):
    BOX_SIZE = 14
    COL_SPACING = 8
    MARGIN = 20

    def __init__(self, closes: List[float], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._chart: PFChart = build_pf_from_closes(
            closes,
            box_mode="percent",
            box_value=0.015,
            reversal=3,
        )
        self._levels: List[float] = self._build_levels()
        self._level_index = {level: idx for idx, level in enumerate(self._levels)}
        self._update_geometry()

    def _build_levels(self) -> List[float]:
        levels = {box for col in self._chart.columns for box in col.boxes}
        return sorted(levels)

    def _update_geometry(self) -> None:
        if not self._levels or not self._chart.columns:
            self._chart_width = 480
            self._chart_height = 320
            return

        col_width = self.BOX_SIZE + self.COL_SPACING
        self._chart_width = (len(self._chart.columns) * col_width) + (self.MARGIN * 2)
        self._chart_height = (len(self._levels) * self.BOX_SIZE) + (self.MARGIN * 2)

    def sizeHint(self) -> QSize:  # pragma: no cover - Qt sizing
        return QSize(self._chart_width, self._chart_height)

    def paintEvent(self, event) -> None:  # pragma: no cover - Qt paint
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if not self._chart.columns or not self._levels:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "No chart data")
            return

        max_level = len(self._levels) - 1
        col_width = self.BOX_SIZE + self.COL_SPACING

        x_pen = QPen(QColor(74, 163, 255))
        x_pen.setWidth(2)
        o_pen = QPen(QColor(255, 120, 120))
        o_pen.setWidth(2)

        for col_idx, col in enumerate(self._chart.columns):
            x = self.MARGIN + (col_idx * col_width)
            for box in col.boxes:
                idx = self._level_index.get(box)
                if idx is None:
                    continue
                y = self.MARGIN + ((max_level - idx) * self.BOX_SIZE)
                rect = QRectF(x, y, self.BOX_SIZE, self.BOX_SIZE)
                if col.col_type == "X":
                    painter.setPen(x_pen)
                    painter.drawLine(rect.topLeft(), rect.bottomRight())
                    painter.drawLine(rect.bottomLeft(), rect.topRight())
                else:
                    painter.setPen(o_pen)
                    painter.drawEllipse(rect)

        painter.setPen(QColor(120, 120, 120))
        painter.drawRect(
            QRectF(
                self.MARGIN / 2,
                self.MARGIN / 2,
                self._chart_width - self.MARGIN,
                self._chart_height - self.MARGIN,
            )
        )
