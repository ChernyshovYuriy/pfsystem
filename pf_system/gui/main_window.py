from __future__ import annotations

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
)
from PySide6.QtWidgets import (
    QGraphicsBlurEffect,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pf_system.app.dto import ScanRequest, ScanResponse
from pf_system.app.services import AppServices
from pf_system.domain.models import ScanResultRow
from pf_system.gui.pf_dialog import PFChartDialog


class ScanResultsTableModel(QAbstractTableModel):
    # Column order is IMPORTANT: proxy relies on these indices.
    # 0 Symbol, 1 Regime, 2 Score, 3 Active, 4 CS, 5 Trigger, 6 Cur, 7 Range, 8 Phase, 9 Entry, 10 Buy CS, 11 Sell CS
    HEADERS = [
        "Symbol",
        "Regime",
        "Score",
        "Active",
        "CS",
        "Trigger",
        "Cur",
        "Range",
        "Phase",
        "Entry",
        "Buy CS",
        "Sell CS",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[ScanResultRow] = []

    def set_rows(self, rows: list[ScanResultRow]) -> None:
        self.beginResetModel()
        # Keep insertion order; sorting is handled by the proxy.
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.HEADERS)

    def get_row(self, index: int) -> ScanResultRow | None:
        if index < 0 or index >= len(self._rows):
            return None
        return self._rows[index]

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
            return None
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
                return row.pf_entry_phase
            if col == 9:
                return "" if row.pf_entry_score is None else f"{row.pf_entry_score:.0f}"
            if col == 10:
                return "" if row.pf_buy_cs is None else str(row.pf_buy_cs)
            if col == 11:
                return "" if row.pf_sell_cs is None else str(row.pf_sell_cs)

        if role == Qt.TextAlignmentRole:
            # Center everything except Symbol (and optionally Active).
            if col == 0:
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            # Uncomment if you prefer Active left-aligned:
            # if col == 3:
            #     return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            return int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        return None


class StrongRuleProxy(QSortFilterProxyModel):
    """
    Sorts rows so that "strong rule" candidates float to the top,
    without filtering the rest out.

    Strong rule:
      Phase == FRESH
      Regime == BULLISH and Score >= 70   (longs)
      OR Regime == BEARISH and Score <= -70 (shorts)
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.min_abs_score = 70.0

        # Ranking maps
        self._phase_rank = {"FRESH": 0, "OK": 1, "EXTENDED": 2, "na": 3}
        self._regime_rank = {"BULLISH": 0, "BEARISH": 1, "NEUTRAL": 2}

    @staticmethod
    def _to_float(x) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _is_strong_candidate(self, phase: str, regime: str, score: float) -> bool:
        if phase != "FRESH":
            return False
        if regime == "BULLISH":
            return score >= self.min_abs_score
        if regime == "BEARISH":
            return score <= -self.min_abs_score
        return False

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        m = self.sourceModel()
        if m is None:
            return super().lessThan(left, right)

        # Column indices depend on ScanResultsTableModel.HEADERS:
        # 0 Symbol, 1 Regime, 2 Score, 8 Phase, 9 Entry
        l_reg = m.index(left.row(), 1).data()
        r_reg = m.index(right.row(), 1).data()

        l_score = self._to_float(m.index(left.row(), 2).data())
        r_score = self._to_float(m.index(right.row(), 2).data())

        l_phase = m.index(left.row(), 8).data()
        r_phase = m.index(right.row(), 8).data()

        l_entry = self._to_float(m.index(left.row(), 9).data())
        r_entry = self._to_float(m.index(right.row(), 9).data())

        # 1) Strong-rule candidates first
        l_strong = self._is_strong_candidate(l_phase, l_reg, l_score)
        r_strong = self._is_strong_candidate(r_phase, r_reg, r_score)
        if l_strong != r_strong:
            # We want True to mean "left < right" in ascending sort,
            # so if left is NOT strong and right IS strong, left should be "greater".
            return (not l_strong) and r_strong

        # 2) Phase (FRESH, OK, EXTENDED, na)
        lp = self._phase_rank.get(str(l_phase), 99)
        rp = self._phase_rank.get(str(r_phase), 99)
        if lp != rp:
            return lp > rp  # reverse so smaller rank appears first

        # 3) Regime (BULLISH then BEARISH then NEUTRAL)
        lr = self._regime_rank.get(str(l_reg), 99)
        rr = self._regime_rank.get(str(r_reg), 99)
        if lr != rr:
            return lr > rr  # reverse so bullish first

        # 4) Strength: abs(score) descending
        la = abs(l_score) if l_score == l_score else -1.0
        ra = abs(r_score) if r_score == r_score else -1.0
        if la != ra:
            return la < ra  # normal comparator: smaller is "less" => this yields descending in practice via inversion above

        # 5) Entry score descending (where available)
        if l_entry != r_entry:
            return l_entry < r_entry

        # 6) Symbol tie-break
        l_sym = m.index(left.row(), 0).data()
        r_sym = m.index(right.row(), 0).data()
        return str(l_sym) < str(r_sym)


class _ScanWorker(QObject):
    finished = Signal(ScanResponse)
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
        self._worker: _ScanWorker | None = None
        self._last_scan_response: ScanResponse | None = None
        self._last_scan_lookback: int | None = None
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

        # Source model + proxy
        self._table_model = ScanResultsTableModel()
        self._proxy = StrongRuleProxy(self)
        self._proxy.setSourceModel(self._table_model)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.doubleClicked.connect(self._on_row_double_clicked)

        # Resize columns to content
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

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

        # Default sort using proxy’s comparator (any column triggers it)
        self._table.sortByColumn(2, Qt.SortOrder.DescendingOrder)

    def _map_to_source_row(self, proxy_index: QModelIndex) -> ScanResultRow | None:
        if not proxy_index.isValid():
            return None
        src_index = self._proxy.mapToSource(proxy_index)
        if not src_index.isValid():
            return None
        return self._table_model.get_row(src_index.row())

    def _on_row_selected(self, selected, _deselected) -> None:
        if not selected.indexes():
            self._details.setText("Select a row to see details.")
            return

        proxy_idx = selected.indexes()[0]
        row = self._map_to_source_row(proxy_idx)
        if row is None:
            self._details.setText("Select a row to see details.")
            return

        self._details.setText(
            f"{row.symbol} | {row.regime} | score={row.score:.2f} | "
            f"active={row.pf_active_name}@{'' if row.pf_active_trigger is None else f'{row.pf_active_trigger:.2f}'} "
            f"(cs={row.pf_active_cs}) | phase={row.pf_entry_phase} entry={'' if row.pf_entry_score is None else f'{row.pf_entry_score:.0f}'} | "
            f"cur={row.pf_cur_type} {row.pf_cur_low:.2f}->{row.pf_cur_high:.2f} | "
            f"buy_cs={row.pf_buy_cs} sell_cs={row.pf_sell_cs}"
        )

    def _on_scan(self) -> None:
        symbols = [s.strip() for s in self._symbols.text().split(",") if s.strip()]
        if not symbols:
            self._status.setText("Enter at least one symbol.")
            return
        if self._thread is not None and self._thread.isRunning():
            self._status.setText("Scan already in progress.")
            return

        req = ScanRequest(symbols=symbols, lookback_days=int(self._lookback.value()))
        self._last_scan_lookback = req.lookback_days

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

    def _on_scan_done(self, resp: ScanResponse) -> None:
        self._table_model.set_rows(resp.rows)
        self._last_scan_response = resp
        self._status.setText(f"Done. {len(resp.rows)} results.")
        self._scan_btn.setEnabled(True)

        # Re-apply proxy sorting
        self._table.sortByColumn(2, Qt.SortOrder.DescendingOrder)

        self._cleanup_scan_thread()

    def _on_scan_failed(self, msg: str) -> None:
        self._status.setText(f"Scan failed: {msg}")
        self._scan_btn.setEnabled(True)
        self._cleanup_scan_thread()

    def _cleanup_scan_thread(self) -> None:
        if self._thread is None:
            return
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None

    def _on_row_double_clicked(self, index: QModelIndex) -> None:
        row = self._map_to_source_row(index)
        if row is None:
            return

        closes_cache = self._last_scan_response.closes_cache if self._last_scan_response else {}
        closes = closes_cache.get(row.symbol)
        if not closes:
            QMessageBox.warning(self, "Missing cached data", f"No cached data for {row.symbol}. Re-run scan.")
            return

        lookback_days = self._last_scan_lookback or int(self._lookback.value())
        dialog = PFChartDialog(row, closes, lookback_days, parent=self)
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
        self._blur_effect = None

    def resizeEvent(self, event) -> None:  # pragma: no cover - Qt event
        super().resizeEvent(event)
        if self._modal_overlay is not None and self._modal_overlay.isVisible():
            self._modal_overlay.setGeometry(self.centralWidget().geometry())
