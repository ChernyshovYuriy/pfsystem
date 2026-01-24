from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
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


class ScanResultsTableModel(QAbstractTableModel):
    HEADERS = ["Symbol", "Regime", "Score", "Note"]

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[ScanResultRow] = []

    def set_rows(self, rows: list[ScanResultRow]) -> None:
        self.beginResetModel()
        self._rows = rows
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
            # if col == 3:
            #     return row.note

        if role == Qt.TextAlignmentRole and col == 2:
            return int(Qt.AlignRight | Qt.AlignVCenter)

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

        # Default symbols so it runs instantly
        self._symbols.setText("XIU.TO, CNQ.TO, SU.TO, SHOP.TO, BNS.TO, AEM.TO, WCN.TO, ATD.TO, TRI.TO")

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
        self._status.setText(f"Done. {len(resp.rows)} results.")
        self._scan_btn.setEnabled(True)
        if self._thread:
            self._thread.quit()

    def _on_scan_failed(self, msg: str) -> None:
        self._status.setText(f"Scan failed: {msg}")
        self._scan_btn.setEnabled(True)
        if self._thread:
            self._thread.quit()
