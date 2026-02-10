from __future__ import annotations

import math
from datetime import date, datetime
from typing import List, Optional, Tuple

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from pf_system.domain.point_and_figure import PFChart, build_pf_from_closes


class PFChartWidget(QWidget):
    BOX_SIZE = 14
    COL_SPACING = 8
    MARGIN = 20

    AXIS_HEIGHT = 46
    AXIS_TICK = 6
    MIN_HEIGHT = 240

    def __init__(self, closes: List[float], dates: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self._closes = closes or []
        self._dates = dates or []
        self._chart: PFChart = build_pf_from_closes(self._closes) if self._closes else PFChart(
            box_mode="percent", box_value=0.015, reversal=3, columns=[]
        )
        self.setMinimumHeight(self.MIN_HEIGHT)

        # Cached vertical mapping
        self._v_min: Optional[float] = None
        self._v_max: Optional[float] = None
        self._v_count: int = 0  # number of box-rows on the y-axis

    # --------- sizing ---------

    def sizeHint(self) -> QSize:
        cols = max(1, len(self._chart.columns))
        width = self.MARGIN * 2 + cols * self.BOX_SIZE + (cols - 1) * self.COL_SPACING

        if not self._chart.columns:
            return QSize(width, self.MIN_HEIGHT)

        v_min, v_max, v_count = self._compute_vertical_grid()
        # add some padding rows
        rows = max(16, v_count + 6)
        height = self.MARGIN * 2 + rows * self.BOX_SIZE + self.AXIS_HEIGHT
        return QSize(width, max(self.MIN_HEIGHT, height))

    # --------- paint ---------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if not self._chart.columns:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignCenter, "No data")
            return

        cols = len(self._chart.columns)
        col_width = self.BOX_SIZE + self.COL_SPACING

        # plot region (reserve bottom for x-axis)
        plot_left = self.MARGIN
        plot_top = self.MARGIN
        plot_right = self.width() - self.MARGIN
        plot_bottom = self.height() - self.MARGIN - self.AXIS_HEIGHT

        v_min, v_max, v_count = self._compute_vertical_grid()

        # grid
        painter.setPen(QColor(60, 60, 60))
        for c in range(cols):
            x = plot_left + c * col_width
            painter.drawLine(int(x), int(plot_top), int(x), int(plot_bottom))

        # horizontal grid lines (one per row)
        for r in range(v_count + 1):
            y = plot_bottom - r * self.BOX_SIZE
            painter.drawLine(int(plot_left), int(y), int(plot_right), int(y))

        # draw symbols
        for c_idx, col in enumerate(self._chart.columns):
            x0 = plot_left + c_idx * col_width
            is_x = (col.col_type == "X")

            for box_price in col.boxes:
                row = self._price_to_row(box_price, v_min)
                y = plot_bottom - row * self.BOX_SIZE
                if is_x:
                    self._draw_x(painter, x0, y)
                else:
                    self._draw_o(painter, x0, y)

        # x-axis
        self._draw_x_axis(painter, plot_bottom, col_width)

    # --------- vertical mapping (core fix) ---------

    def _compute_vertical_grid(self) -> Tuple[float, float, int]:
        """Compute min/max and number of box-rows for PF chart."""
        if self._v_min is not None:
            return self._v_min, self._v_max, self._v_count  # type: ignore

        all_boxes: List[float] = []
        for c in self._chart.columns:
            all_boxes.extend(c.boxes)

        v_min = min(all_boxes)
        v_max = max(all_boxes)

        # For percent mode, grid is multiplicative, so use logarithmic count
        if self._chart.box_mode == "fixed":
            step = float(self._chart.box_value)
            if step <= 0:
                step = 0.01
            count = int(math.ceil((v_max - v_min) / step)) + 1
        else:
            # percent: each step is * (1 + box_value)
            base = 1.0 + float(self._chart.box_value)
            if base <= 1.0:
                base = 1.015
            # how many multiplicative steps from v_min to v_max
            count = int(math.ceil(math.log(v_max / v_min) / math.log(base))) + 1

        self._v_min, self._v_max, self._v_count = v_min, v_max, max(1, count)
        return self._v_min, self._v_max, self._v_count

    def _price_to_row(self, price: float, v_min: float) -> int:
        """Convert a box price level into an integer row index (0 at v_min)."""
        if self._chart.box_mode == "fixed":
            step = float(self._chart.box_value)
            if step <= 0:
                step = 0.01
            return int(round((price - v_min) / step))

        base = 1.0 + float(self._chart.box_value)
        if base <= 1.0:
            base = 1.015
        # rows are log steps from v_min
        return int(round(math.log(price / v_min) / math.log(base)))

    # --------- symbol drawing ---------

    def _draw_x(self, painter: QPainter, x0: float, y: float) -> None:
        pen = QPen(QColor(60, 180, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        s = self.BOX_SIZE * 0.75
        cx = x0 + self.BOX_SIZE / 2
        cy = y - self.BOX_SIZE / 2
        painter.drawLine(int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))
        painter.drawLine(int(cx - s / 2), int(cy + s / 2), int(cx + s / 2), int(cy - s / 2))

    def _draw_o(self, painter: QPainter, x0: float, y: float) -> None:
        pen = QPen(QColor(255, 90, 90))
        pen.setWidth(2)
        painter.setPen(pen)
        s = self.BOX_SIZE * 0.75
        cx = x0 + self.BOX_SIZE / 2
        cy = y - self.BOX_SIZE / 2
        painter.drawEllipse(int(cx - s / 2), int(cy - s / 2), int(s), int(s))

    # --------- x-axis (label collision fix) ---------

    def _draw_x_axis(self, painter: QPainter, plot_bottom: float, col_width: int) -> None:
        axis_y = plot_bottom + self.MARGIN / 2
        left_x = self.MARGIN
        right_x = self.MARGIN + (len(self._chart.columns) - 1) * col_width + self.BOX_SIZE

        painter.setPen(QColor(140, 140, 140))
        painter.drawLine(int(left_x), int(axis_y), int(right_x), int(axis_y))

        n_cols = max(1, len(self._chart.columns))

        def _parse_date(s: str) -> Optional[date]:
            if not s:
                return None
            s = s.split("T")[0].split(" ")[0]
            for fmt in ("%Y-%m-%d", "%y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt).date()
                except Exception:
                    pass
            return None

        parsed: List[date] = []
        for d in (self._dates or []):
            dd = _parse_date(str(d))
            if dd is not None:
                parsed.append(dd)

        # estimate labels by pixel capacity (prevents overlap / prevents "none drawn")
        available_px = max(1.0, float(right_x - left_x))
        est_label_px = 80.0  # safe width for 'Feb 03'
        max_labels = int(available_px // est_label_px)
        max_labels = max(2, min(10, max_labels))

        if len(parsed) >= 2:
            # month boundaries + end
            month_idxs: List[int] = [0]
            for i in range(1, len(parsed)):
                if (parsed[i].year, parsed[i].month) != (parsed[i - 1].year, parsed[i - 1].month):
                    month_idxs.append(i)
            if month_idxs[-1] != len(parsed) - 1:
                month_idxs.append(len(parsed) - 1)

            # thin to fit
            if len(month_idxs) > max_labels:
                stride = int(math.ceil(len(month_idxs) / float(max_labels)))
                month_idxs = month_idxs[::stride]
                if month_idxs[-1] != len(parsed) - 1:
                    month_idxs.append(len(parsed) - 1)

            n_dates = len(parsed)

            painter.setPen(QColor(180, 180, 180))
            for idx in month_idxs:
                col_idx = round(idx * (n_cols - 1) / (n_dates - 1)) if n_cols > 1 else 0
                col_idx = min(max(col_idx, 0), n_cols - 1)

                x = left_x + col_idx * col_width + (self.BOX_SIZE / 2)
                painter.drawLine(int(x), int(axis_y), int(x), int(axis_y + self.AXIS_TICK))

                d = parsed[idx]
                label = f"{d.strftime('%b')} {d.day:02d}"
                rect = QRectF(x - 45, axis_y + self.AXIS_TICK + 4, 90, 20)
                painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, label)
            return

        # fallback: sparse column indices
        painter.setPen(QColor(180, 180, 180))
        step = max(1, int(math.ceil(n_cols / float(max_labels))))
        for col_idx in range(0, n_cols, step):
            x = left_x + col_idx * col_width + (self.BOX_SIZE / 2)
            painter.drawLine(int(x), int(axis_y), int(x), int(axis_y + self.AXIS_TICK))
            rect = QRectF(x - 20, axis_y + self.AXIS_TICK + 4, 40, 20)
            painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, str(col_idx + 1))
