from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    QThread,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
from pf_system.gui.watchlist_store import WatchlistStore


class ScanResultsTableModel(QAbstractTableModel):
    # Column order is IMPORTANT: proxy relies on these indices.
    # 0 Symbol, 1 Regime, 2 Score, 3 Active, 4 CS, 5 Trigger, 6 Cur, 7 Range,
    # 8 Phase, 9 Entry, 10 Buy CS, 11 Sell CS, 12 Track
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
        "Track",
    ]

    trackedChanged = Signal(set)

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[ScanResultRow] = []
        self._tracked: set[str] = set()

    def set_tracked_symbols(self, tracked: set[str]) -> None:
        """Sets the global tracked symbols used by the Track checkbox column."""
        self._tracked = {str(s).upper() for s in tracked}
        if self.rowCount() > 0:
            # Refresh the Track column.
            col = self.columnCount() - 1
            top_left = self.index(0, col)
            bottom_right = self.index(self.rowCount() - 1, col)
            self.dataChanged.emit(top_left, bottom_right, [Qt.CheckStateRole])

    def tracked_symbols(self) -> set[str]:
        return set(self._tracked)

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

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        col = index.column()
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

        # Track column is checkable.
        if col == (len(self.HEADERS) - 1):
            flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable

        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if not index.isValid():
            return False

        col = index.column()
        if col != (len(self.HEADERS) - 1):
            return False

        if role == Qt.CheckStateRole:
            row = self._rows[index.row()]
            sym = row.symbol.upper()

            if value == Qt.CheckState.Checked:
                self._tracked.add(sym)
            else:
                self._tracked.discard(sym)

            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            self.trackedChanged.emit(set(self._tracked))
            return True

        return False

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.CheckStateRole:
            if col == (len(self.HEADERS) - 1):
                sym = row.symbol.upper()
                return Qt.CheckState.Checked if sym in self._tracked else Qt.CheckState.Unchecked

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
            if col == 12:
                return ""

        if role == Qt.TextAlignmentRole:
            if col == 0:
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        return None


class StrongRuleProxy(QSortFilterProxyModel):
    """Custom ranking proxy.

    Goals:
      1) Tracked tickers are pinned to the top
      2) Strong rule candidates float to the top of the remaining set
      3) Sorting never becomes "inverted" due to Qt's DescendingOrder reversal.

    We treat the header's sort indicator as a *preference* and apply it ourselves,
    while forcing the underlying Qt sort to AscendingOrder.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.min_abs_score = 70.0

        self._phase_rank = {"FRESH": 0, "OK": 1, "EXTENDED": 2, "na": 3}
        self._regime_rank = {"BULLISH": 0, "BEARISH": 1, "NEUTRAL": 2}

        self._tracked: set[str] = set()
        self._tracked_only: bool = False

        # Track the user's last requested sort.
        self._user_sort_col: int = 2
        self._user_sort_order: Qt.SortOrder = Qt.SortOrder.DescendingOrder

    def set_tracked_symbols(self, tracked: set[str]) -> None:
        self._tracked = {str(s).upper() for s in tracked}
        self.invalidate()
        self.invalidateFilter()

    def set_tracked_only(self, enabled: bool) -> None:
        self._tracked_only = bool(enabled)
        self.invalidateFilter()

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        # Store the user's preference...
        self._user_sort_col = int(column)
        self._user_sort_order = order
        # ...but force the underlying Qt sort to ASC so it doesn't invert lessThan().
        super().sort(column, Qt.SortOrder.AscendingOrder)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if not self._tracked_only:
            return True

        m = self.sourceModel()
        if m is None:
            return True

        sym = m.index(source_row, 0, source_parent).data()
        return str(sym).upper() in self._tracked

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

        # Symbols are used for tracked pinning and tie-breaks.
        l_sym = str(m.index(left.row(), 0).data()).upper()
        r_sym = str(m.index(right.row(), 0).data()).upper()

        # 0) Tracked tickers always float to the top.
        l_tr = l_sym in self._tracked
        r_tr = r_sym in self._tracked
        if l_tr != r_tr:
            return l_tr

        # Column indices depend on ScanResultsTableModel.HEADERS:
        # 0 Symbol, 1 Regime, 2 Score, 8 Phase, 9 Entry
        l_reg = str(m.index(left.row(), 1).data())
        r_reg = str(m.index(right.row(), 1).data())

        l_score = self._to_float(m.index(left.row(), 2).data())
        r_score = self._to_float(m.index(right.row(), 2).data())

        l_phase = str(m.index(left.row(), 8).data())
        r_phase = str(m.index(right.row(), 8).data())

        l_entry = self._to_float(m.index(left.row(), 9).data())
        r_entry = self._to_float(m.index(right.row(), 9).data())

        # 1) Strong-rule candidates first.
        l_strong = self._is_strong_candidate(l_phase, l_reg, l_score)
        r_strong = self._is_strong_candidate(r_phase, r_reg, r_score)
        if l_strong != r_strong:
            return l_strong

        # 2) Phase (FRESH, OK, EXTENDED, na)
        lp = self._phase_rank.get(l_phase, 99)
        rp = self._phase_rank.get(r_phase, 99)
        if lp != rp:
            return lp < rp

        # 3) Regime (BULLISH then BEARISH then NEUTRAL)
        lr = self._regime_rank.get(l_reg, 99)
        rr = self._regime_rank.get(r_reg, 99)
        if lr != rr:
            return lr < rr

        # 4) Apply the user's chosen sort column/order *within* the ranked bucket.
        #    This keeps your "tracked/strong/phase/regime" logic stable, while still
        #    letting you sort by Score/Symbol/etc.
        order = self._user_sort_order

        if self._user_sort_col == 0:  # Symbol
            if l_sym != r_sym:
                return (l_sym < r_sym) if order == Qt.SortOrder.AscendingOrder else (l_sym > r_sym)

        elif self._user_sort_col == 2:  # Score
            if l_score != r_score:
                # For BEARISH rows, a "stronger" score is *more negative*.
                # Interpreting DescendingOrder as "stronger first" gives:
                #   BULLISH: higher score first
                #   BEARISH: lower score first
                if order == Qt.SortOrder.DescendingOrder:
                    if l_reg == "BEARISH" and r_reg == "BEARISH":
                        return l_score < r_score
                    return l_score > r_score
                else:
                    if l_reg == "BEARISH" and r_reg == "BEARISH":
                        return l_score > r_score
                    return l_score < r_score

        elif self._user_sort_col == 9:  # Entry
            if l_entry != r_entry:
                return (l_entry < r_entry) if order == Qt.SortOrder.AscendingOrder else (l_entry > r_entry)

        # 5) Default strength: abs(score) descending (bigger first)
        la = abs(l_score) if l_score == l_score else -1.0
        ra = abs(r_score) if r_score == r_score else -1.0
        if la != ra:
            return la > ra

        # 6) Entry score descending (where available)
        if l_entry != r_entry:
            return l_entry > r_entry

        # 7) Symbol tie-break
        return l_sym < r_sym


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

        # Global watchlist (tracked tickers)
        self._watchlist_store = WatchlistStore(Path.home() / ".pfsystem" / "watchlist.txt")
        self._tracked: set[str] = self._watchlist_store.load()

        # Remember current sort indicator (so tracked toggles can re-sort consistently).
        self._sort_col: int = 2
        self._sort_order: Qt.SortOrder = Qt.SortOrder.DescendingOrder

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

        self._tracked_only = QCheckBox("Tracked only")
        self._tracked_only.setChecked(False)

        self._status = QLabel("Ready.")

        # Source model + proxy
        self._table_model = ScanResultsTableModel()
        self._proxy = StrongRuleProxy(self)
        self._proxy.setSourceModel(self._table_model)

        # Wire watchlist state into model + proxy
        self._table_model.set_tracked_symbols(self._tracked)
        self._proxy.set_tracked_symbols(self._tracked)
        self._tracked_only.toggled.connect(self._proxy.set_tracked_only)
        self._table_model.trackedChanged.connect(self._on_tracked_changed)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)

        # Make the table behave like a list: select whole row, single selection.
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        # Important: checkbox toggling is handled explicitly (more reliable than relying
        # on edit triggers, especially with SelectRows).
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.clicked.connect(self._on_table_clicked)

        # Open chart on double click, except Track column.
        self._table.doubleClicked.connect(self._on_row_double_clicked)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.sortIndicatorChanged.connect(self._on_sort_indicator_changed)

        top = QHBoxLayout()
        top.addWidget(QLabel("Symbols:"))
        top.addWidget(self._symbols, 1)
        top.addWidget(QLabel("Lookback:"))
        top.addWidget(self._lookback)
        top.addWidget(self._scan_btn)
        top.addWidget(self._tracked_only)

        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._status)

        self._details = QLabel("Select a row to see details.")
        layout.addWidget(self._details)

        self._table.selectionModel().selectionChanged.connect(self._on_row_selected)

        # Default symbols so it runs instantly
        if self._tracked:
            self._symbols.setText(", ".join(sorted(self._tracked)))
        else:
            self._symbols.setText("XIU.TO, CNQ.TO, SU.TO, SHOP.TO, BNS.TO, AEM.TO, WCN.TO, ATD.TO, TRI.TO")

        # Default sort: Score (descending)
        self._table.sortByColumn(self._sort_col, self._sort_order)

    @property
    def _track_col(self) -> int:
        return len(ScanResultsTableModel.HEADERS) - 1

    def _on_sort_indicator_changed(self, section: int, order: Qt.SortOrder) -> None:
        self._sort_col = int(section)
        self._sort_order = order

    def _map_to_source_row(self, proxy_index: QModelIndex) -> ScanResultRow | None:
        if not proxy_index.isValid():
            return None
        src_index = self._proxy.mapToSource(proxy_index)
        if not src_index.isValid():
            return None
        return self._table_model.get_row(src_index.row())

    def _on_table_clicked(self, index: QModelIndex) -> None:
        # Toggle checkbox reliably.
        if not index.isValid() or index.column() != self._track_col:
            return

        cur = self._proxy.data(index, Qt.CheckStateRole)
        new_state = Qt.CheckState.Unchecked if cur == Qt.CheckState.Checked else Qt.CheckState.Checked
        self._proxy.setData(index, new_state, Qt.CheckStateRole)

    def _on_tracked_changed(self, tracked: set) -> None:
        """Persists global watchlist and refreshes proxy ordering/filter."""
        self._tracked = {str(s).upper() for s in tracked}
        self._watchlist_store.save(self._tracked)

        # Update both model and proxy.
        self._table_model.set_tracked_symbols(self._tracked)
        self._proxy.set_tracked_symbols(self._tracked)

        # Re-apply user's current sort indicator.
        self._table.sortByColumn(self._sort_col, self._sort_order)

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

        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_scan_done(self, resp: ScanResponse) -> None:
        self._table_model.set_rows(resp.rows)
        self._last_scan_response = resp
        self._status.setText(f"Done. {len(resp.rows)} results.")
        self._scan_btn.setEnabled(True)

        # Re-apply sort preference after data reset.
        self._table.sortByColumn(self._sort_col, self._sort_order)
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
        if not index.isValid() or index.column() == self._track_col:
            return

        row = self._map_to_source_row(index)
        if row is None:
            return

        closes_cache = self._last_scan_response.closes_cache if self._last_scan_response else {}
        raw_series = closes_cache.get(row.symbol)
        closes, dates = self._extract_closes_and_dates(raw_series)
        if not closes:
            QMessageBox.warning(self, "Missing cached data", f"No cached data for {row.symbol}. Re-run scan.")
            return

        lookback_days = self._last_scan_lookback or int(self._lookback.value())
        dialog = PFChartDialog(row, closes, dates, lookback_days, parent=self)
        self._show_modal_overlay()
        try:
            dialog.exec()
        finally:
            self._hide_modal_overlay()

    @staticmethod
    def _extract_closes_and_dates(raw_series) -> tuple[list[float], list[str]]:
        """Best-effort extraction for several common cache formats.

        Supported shapes:
          - List[float] -> closes only
          - List[(date_str, close)]
          - List[{"date": ..., "close": ...}] (or "time" instead of "date")
        """
        if not raw_series:
            return [], []

        # List[(date, close)]
        first = raw_series[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            dates = [str(x[0]) for x in raw_series]
            closes = [float(x[1]) for x in raw_series]
            return closes, dates

        # List[dict]
        if isinstance(first, dict):
            date_key = "date" if "date" in first else ("time" if "time" in first else None)
            close_key = "close" if "close" in first else ("c" if "c" in first else None)
            if close_key is not None:
                closes = [float(x.get(close_key)) for x in raw_series]
                dates = [str(x.get(date_key, "")) for x in raw_series] if date_key else []
                return closes, dates

        # Assume List[float]
        try:
            return [float(x) for x in raw_series], []
        except Exception:
            return [], []

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
