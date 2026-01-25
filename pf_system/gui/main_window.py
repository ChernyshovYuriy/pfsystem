from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QGraphicsBlurEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pf_system.app.dto import ScanRequest
from pf_system.app.services import AppServices
from pf_system.domain.models import ScanResultRow
from pf_system.gui.pf_dialog import PFChartDialog


class ScanResultsTableModel(QAbstractTableModel):
    HEADERS = [
        "Symbol",
        "Regime",
        "Score",
        "Active",
        "CS",
        "Trigger",
        "Cur",
        "Range",
        "Buy CS",
        "Sell CS",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[ScanResultRow] = []

    def set_rows(self, rows: list[ScanResultRow]) -> None:
        self.beginResetModel()
        # Default sort: highest score first
        self._rows = sorted(rows, key=lambda r: r.score, reverse=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return row.symbol
            if col == 1:
                return row.regime
            if col == 2:
                return f"{row.score:.2f}"
            if col == 3:
                return row.pf_active_name
            if col == 4:
                return "" if row.pf_active_cs is None else str(row.pf_active_cs)
            if col == 5:
                return "" if row.pf_active_trigger is None else f"{row.pf_active_trigger:.2f}"
            if col == 6:
                return row.pf_cur_type
            if col == 7:
                return f"{row.pf_cur_low:.2f} → {row.pf_cur_high:.2f}"
            if col == 8:
                return "" if row.pf_buy_cs is None else str(row.pf_buy_cs)
            if col == 9:
                return "" if row.pf_sell_cs is None else str(row.pf_sell_cs)

        if role == Qt.TextAlignmentRole:
            # right align numeric-ish columns
            if col in (2, 4, 5, 8, 9):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)

        return None


class _ScanWorker(QObject):
    finished = Signal(object)  # ScanResponse
    failed = Signal(str)

    def __init__(self, services: AppServices, req: ScanRequest) -> None:
        super().__init__()
        self._services = services
        self._req = req

    def run(self) -> None:
        try:
            resp = self._services.scan_market.execute(self._req)
            self.finished.emit(resp)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, services: AppServices) -> None:
        super().__init__()
        self._services = services
        self._thread: QThread | None = None
        self._worker = None
        self._closes_cache: dict[str, list[float]] = {}
        self._modal_overlay: QWidget | None = None
        self._blur_effect: QGraphicsBlurEffect | None = None

        self.setWindowTitle("P&F Market Scanner (Skeleton)")

        root = QWidget()
        self.setCentralWidget(root)

        self._symbols = QLineEdit()
        self._symbols.setPlaceholderText("Comma-separated symbols, e.g. XIU.TO, CNQ.TO, SHOP.TO")

        self._lookback = QSpinBox()
        self._lookback.setMinimum(30)
        self._lookback.setMaximum(2000)
        self._lookback.setValue(self._services.settings.default_lookback_days)

        self._scan_btn = QPushButton("Scan Market")
        self._scan_btn.clicked.connect(self._on_scan)

        self._status = QLabel("Ready.")

        self._table_model = ScanResultsTableModel()
        self._table = QTableView()
        self._table.setModel(self._table_model)
        self._table.setSortingEnabled(False)
        self._table.doubleClicked.connect(self._on_row_double_clicked)

        top = QHBoxLayout()
        top.addWidget(QLabel("Symbols:"))
        top.addWidget(self._symbols, 1)
        top.addWidget(QLabel("Lookback:"))
        top.addWidget(self._lookback)
        top.addWidget(self._scan_btn)

        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._status)

        self._details = QLabel("Select a row to see details.")
        layout.addWidget(self._details)
        self._table.selectionModel().selectionChanged.connect(self._on_row_selected)

        # Default symbols so it runs instantly
        self._symbols.setText("XIU.TO, CNQ.TO, SU.TO, SHOP.TO, BNS.TO, AEM.TO, WCN.TO, ATD.TO, TRI.TO")

    def _on_row_selected(self, selected, _deselected) -> None:
        if not selected.indexes():
            self._details.setText("Select a row to see details.")
            return

        idx = selected.indexes()[0]
        row = self._table_model._rows[idx.row()]

        self._details.setText(
            f"{row.symbol} | {row.regime} | score={row.score:.2f} | "
            f"active={row.pf_active_name}@{'' if row.pf_active_trigger is None else f'{row.pf_active_trigger:.2f}'} "
            f"(cs={row.pf_active_cs}) | cur={row.pf_cur_type} {row.pf_cur_low:.2f}->{row.pf_cur_high:.2f} | "
            f"buy_cs={row.pf_buy_cs} sell_cs={row.pf_sell_cs}"
        )

    def _on_scan(self) -> None:
        symbols = [s.strip() for s in self._symbols.text().split(",") if s.strip()]
        if not symbols:
            self._status.setText("Enter at least one symbol.")
            return

        req = ScanRequest(symbols=symbols, lookback_days=int(self._lookback.value()))

        self._scan_btn.setEnabled(False)
        self._status.setText(f"Scanning {len(symbols)} symbols...")

        self._thread = QThread(self)
        self._worker = _ScanWorker(self._services, req)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_scan_done)
        self._worker.failed.connect(self._on_scan_failed)

        # Cleanup
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_scan_done(self, resp) -> None:
        self._table_model.set_rows(resp.rows)
        self._closes_cache = resp.closes_cache
        self._status.setText(f"Done. {len(resp.rows)} results.")
        self._scan_btn.setEnabled(True)
        if self._thread:
            self._thread.quit()

    def _on_scan_failed(self, msg: str) -> None:
        self._status.setText(f"Scan failed: {msg}")
        self._scan_btn.setEnabled(True)
        if self._thread:
            self._thread.quit()

    def _on_row_double_clicked(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        row = self._table_model._rows[index.row()]
        closes = self._closes_cache.get(row.symbol)
        if not closes:
            self._status.setText(f"No cached closes available for {row.symbol}.")
            return

        dialog = PFChartDialog(row.symbol, closes, parent=self)
        self._show_modal_overlay()
        try:
            dialog.exec()
        finally:
            self._hide_modal_overlay()

    def _show_modal_overlay(self) -> None:
        if self._modal_overlay is None:
            self._modal_overlay = QWidget(self)
            self._modal_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 160);")
        self._modal_overlay.setGeometry(self.centralWidget().geometry())
        self._modal_overlay.show()
        self._modal_overlay.raise_()

        if self._blur_effect is None:
            self._blur_effect = QGraphicsBlurEffect()
            self._blur_effect.setBlurRadius(6)
        self.centralWidget().setGraphicsEffect(self._blur_effect)

    def _hide_modal_overlay(self) -> None:
        if self._modal_overlay is not None:
            self._modal_overlay.hide()
        self.centralWidget().setGraphicsEffect(None)

    def resizeEvent(self, event) -> None:  # pragma: no cover - Qt event
        super().resizeEvent(event)
        if self._modal_overlay is not None and self._modal_overlay.isVisible():
            self._modal_overlay.setGeometry(self.centralWidget().geometry())
