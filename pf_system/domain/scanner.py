from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from pf_system.config.settings import Settings
from pf_system.domain.models import ScanResultRow
from pf_system.domain.pf_engine import evaluate_pf
from pf_system.domain.pf_verify import summarize_pf, verify_pf_chart
from pf_system.domain.point_and_figure import build_pf_from_closes
from pf_system.ports.data_provider import MarketDataProvider


def _safe_signal_trigger(sig) -> Optional[float]:
    if sig is None:
        return None
    trig = getattr(sig, "trigger", None)
    return float(trig) if isinstance(trig, (int, float)) else None


def _safe_cols_since(value) -> Optional[int]:
    return int(value) if isinstance(value, int) else None


def _entry_phase(cols_since: Optional[int], boxes_from_trigger: Optional[float]) -> str:
    if cols_since is None or boxes_from_trigger is None:
        return "na"
    # “Fresh” = very recent and not stretched
    if cols_since <= 1 and boxes_from_trigger <= 2.0:
        return "FRESH"
    # “OK” = still reasonable continuation (but not first print)
    if cols_since <= 3 and boxes_from_trigger <= 6.0:
        return "OK"
    return "EXTENDED"


def _entry_score(cols_since: Optional[int], boxes_from_trigger: Optional[float]) -> Optional[float]:
    if cols_since is None or boxes_from_trigger is None:
        return None

    # Age penalty: older signal = worse entry timing
    age_pen = min(40.0, 8.0 * float(cols_since))  # 0..40

    # Extension penalty: stretched beyond 2 boxes = worse entry timing
    ext_pen = min(60.0, 20.0 * max(0.0, boxes_from_trigger - 2.0))  # 0..60

    return max(0.0, 100.0 - age_pen - ext_pen)


class DomainScanner:
    """Domain service orchestrating data fetch, P&F evaluation, and scoring."""

    def __init__(
            self,
            data_provider: MarketDataProvider,
            benchmark: str = "XIU.TO",
            *,
            enable_audit: bool = False,
    ) -> None:
        self.settings = Settings()
        self._data = data_provider
        self._benchmark = benchmark
        self._enable_audit = enable_audit

    def scan(
            self,
            symbols: Sequence[str],
            lookback_days: int,
    ) -> Tuple[List[ScanResultRow], dict[str, list[float]]]:
        out: List[ScanResultRow] = []
        closes_cache: dict[str, list[float]] = {}

        bench_closes = self._try_get_benchmark_closes(lookback_days)

        for sym in symbols:
            try:
                series = self._safe_fetch_series(sym, lookback_days)
                if series is not None:
                    closes_cache[sym] = series[0]
                row = self._scan_symbol(sym, lookback_days, bench_closes, series)
            except Exception as exc:  # per-symbol guardrail
                print(f"{sym}: scan error: {exc!r}")
                row = self._error_row(sym, f"error: {exc}")
            out.append(row)

        out.sort(key=lambda r: r.score, reverse=True)
        return out, closes_cache

    def _try_get_benchmark_closes(self, lookback_days: int) -> Optional[List[float]]:
        try:
            bench_series = self._safe_fetch_series(self._benchmark, lookback_days)
        except Exception as exc:
            print(f"{self._benchmark}: benchmark fetch error: {exc!r}")
            return None

        if bench_series is None:
            print(f"{self._benchmark}: benchmark unavailable")
            return None

        bench_closes = bench_series[0]
        if len(bench_closes) < 130:
            print(f"{self._benchmark}: benchmark insufficient ({len(bench_closes)} bars)")
            return None
        return bench_closes

    def _scan_symbol(
            self,
            sym: str,
            lookback_days: int,
            bench_closes: Optional[List[float]],
            series: Optional[Tuple[List[float], List[float]]] = None,
    ) -> ScanResultRow:
        if series is None:
            print(f"{sym}: insufficient data (no usable bars)")
            return self._error_row(sym, "insufficient data")

        closes, vols = series
        if len(closes) < 210:
            print(f"{sym}: insufficient data ({len(closes)} bars)")
            return self._error_row(sym, "insufficient data")

        last = float(closes[-1])

        # --- P&F evaluation (Option B driver) ---
        pf_chart = build_pf_from_closes(
            closes,
            box_mode="percent",
            box_value=self.settings.box_value,
            reversal=3,
        )
        pf_res = evaluate_pf(
            pf_chart,
            last_price=last,
            box_value=self.settings.box_value,
        )

        if self._enable_audit:
            # Audit uses already-fetched series; no refetch inside audit.
            self._audit_symbol(sym, closes, lookback_days, bench_closes)

        # --- Returns ---
        r6m = self._return_over(closes, 126)

        # --- Trend (SMA) ---
        sma200 = float(self._sma(closes, 200))

        # --- Relative strength vs benchmark (return differential) ---
        rs6m: Optional[float] = None
        if bench_closes is not None and len(bench_closes) >= 127 and len(closes) >= 127:
            br6m = self._return_over(bench_closes, 126)
            rs6m = r6m - br6m

        # --- Liquidity ---
        dv20 = float(self._avg_dollar_vol(closes, vols, 20))
        liq = float(self._liquidity_score(dv20))

        # --- Option B: start from PF regime/score ---
        regime = str(getattr(pf_res, "regime", "NEUTRAL"))
        base_score = float(getattr(pf_res, "score", 0.0))

        # --- Context multipliers (applied first) ---
        score = base_score
        mult = 1.0
        if regime == "BULLISH":
            if rs6m is not None and rs6m < 0:
                mult *= 0.6
            if last <= sma200:
                mult *= 0.5
        elif regime == "BEARISH":
            if rs6m is not None and rs6m > 0:
                mult *= 0.7
            if last >= sma200:
                mult *= 0.7
        score *= mult

        # --- Whipsaw damping (signal proximity after context) ---
        buy_cs = _safe_cols_since(getattr(pf_res, "buy_cs", None))
        sell_cs = _safe_cols_since(getattr(pf_res, "sell_cs", None))

        if regime == "BULLISH" and sell_cs is not None:
            if sell_cs <= 2:
                score *= 0.75
            if buy_cs is not None and abs(buy_cs - sell_cs) <= 1:
                score *= 0.9
        elif regime == "BEARISH" and buy_cs is not None:
            if buy_cs <= 2:
                score *= 0.75
            if sell_cs is not None and abs(buy_cs - sell_cs) <= 1:
                score *= 0.9

        # --- Liquidity add-on last (only non-neutral regimes) ---
        if regime != "NEUTRAL":
            score += 5.0 * liq

        # --- PF summary fields (contract-safe) ---
        signal = getattr(pf_res, "signal", None)
        active_name = getattr(signal, "name", "none") if signal else "none"
        active_trigger = _safe_signal_trigger(signal)
        active_cs = _safe_cols_since(getattr(pf_res, "cols_since", None))

        buy_sig = getattr(pf_res, "buy_sig", None)
        sell_sig = getattr(pf_res, "sell_sig", None)
        buy_trigger = _safe_signal_trigger(buy_sig)
        sell_trigger = _safe_signal_trigger(sell_sig)

        cur_type = str(getattr(pf_res, "cur_type", "?"))
        cur_low = float(getattr(pf_res, "cur_low", 0.0))
        cur_high = float(getattr(pf_res, "cur_high", 0.0))

        # --- Entry/timing layer (separate from structural score) ---
        boxes_from_trigger: Optional[float] = None
        entry_score: Optional[float] = None
        entry_phase: str = "na"

        if regime != "NEUTRAL" and active_trigger is not None and active_trigger > 0:
            # Use last-price distance (simple + consistent with pf_engine scoring)
            if active_name.startswith("BUY"):
                distance_pct = (last / active_trigger) - 1.0
            else:
                distance_pct = (active_trigger / last) - 1.0

            if distance_pct >= 0 and pf_res is not None:
                # IMPORTANT: uses the SAME box_value you used to build PF
                boxes_from_trigger = distance_pct / self.settings.box_value
                entry_phase = _entry_phase(active_cs, boxes_from_trigger)
                entry_score = _entry_score(active_cs, boxes_from_trigger)

                # Make “Active” label truthy but not misleading
                if active_name != "none":
                    active_name = f"{active_name}_{entry_phase}"

        # UX rule: neutral posture should not advertise an active signal.
        if regime == "NEUTRAL":
            active_name = "none"
            active_trigger = None
            active_cs = None
            entry_phase = "na"
            entry_score = None
            boxes_from_trigger = None

        result = ScanResultRow(
            symbol=sym,
            regime=regime,
            score=float(score),
            pf_active_name=active_name,
            pf_active_trigger=active_trigger,
            pf_active_cs=active_cs,
            pf_buy_trigger=buy_trigger,
            pf_buy_cs=buy_cs,
            pf_sell_trigger=sell_trigger,
            pf_sell_cs=sell_cs,
            pf_cur_type=cur_type,
            pf_cur_low=cur_low,
            pf_cur_high=cur_high,
            pf_entry_phase=entry_phase,
            pf_entry_score=entry_score,
            pf_boxes_from_trigger=boxes_from_trigger
        )

        print(f"{sym}: regime={regime} base={base_score:.2f} score={score:.2f} row={result}")
        return result

    # -------------------------
    # Audit (console-only)
    # -------------------------

    def _audit_symbol(
            self,
            symbol: str,
            closes: List[float],
            lookback_days: int,
            bench_closes: Optional[List[float]],
    ) -> None:
        if len(closes) < 210:
            print("=" * 60)
            print("SYMBOL:", symbol)
            print("INSUFFICIENT DATA:", len(closes))
            return

        last = closes[-1]
        r1m = self._return_over(closes, 21)
        r3m = self._return_over(closes, 63)
        r6m = self._return_over(closes, 126)
        sma50 = self._sma(closes, 50)
        sma200 = self._sma(closes, 200)

        # Build P&F from THESE closes (no refetch)
        chart = build_pf_from_closes(
            closes,
            box_mode="percent",
            box_value=self.settings.box_value,
            reversal=3,
        )
        ok, problems = verify_pf_chart(chart)

        print("=" * 60)
        print("SYMBOL:", symbol)
        print("BARS:", len(closes))
        print(f"LAST CLOSE: {last:.2f}")
        print("1M START:", closes[-22], "END:", last, "RET:", f"{r1m * 100:.2f}%")
        print("3M START:", closes[-64], "END:", last, "RET:", f"{r3m * 100:.2f}%")
        print("6M START:", closes[-127], "END:", last, "RET:", f"{r6m * 100:.2f}%")
        print(f"SMA50:  {sma50:.2f}")
        print(f"SMA200: {sma200:.2f}")

        # RS sanity (correct unpack / correct series)
        if symbol == self._benchmark:
            print("RS3M: 0.00%  (benchmark)")
            print("RS6M: 0.00%  (benchmark)")
        elif bench_closes is not None:
            if len(bench_closes) >= 64:
                br3m = self._return_over(bench_closes, 63)
                print(f"RS3M: {(r3m - br3m) * 100:.2f}%")
            else:
                print("RS3M: na (benchmark insufficient)")

            if len(bench_closes) >= 127:
                br6m = self._return_over(bench_closes, 126)
                print(f"RS6M: {(r6m - br6m) * 100:.2f}%")
            else:
                print("RS6M: na (benchmark insufficient)")
        else:
            print("RS3M: na (benchmark unavailable)")
            print("RS6M: na (benchmark unavailable)")

        # P&F output (symbol-prefixed to avoid log confusion)
        print(f"{symbol} P&F COLUMNS:", len(chart.columns))
        if chart.columns:
            cur = chart.current
            if last < cur.low or last > cur.high * (1.0 + 2 * self.settings.box_value):
                print(
                    f"{symbol} P&F SANITY FAIL: last close {last:.2f} not compatible with "
                    f"current column {cur.low:.2f}..{cur.high:.2f}"
                )
            print(
                f"{symbol} P&F CURRENT:",
                cur.col_type,
                "LOW:",
                f"{cur.low:.2f}",
                "HIGH:",
                f"{cur.high:.2f}",
                "BOXES:",
                len(cur.boxes),
            )
            print(f"{symbol} P&F TAPE:", summarize_pf(chart, last_n=10))

            # Basic price sanity: last close should be near current column range
            # (close-only chart: last close can be between boxes; we allow 2 boxes tolerance)
            # We don't compute box size directly here, but this catches egregious mismatches.
            if not (cur.low * 0.85 <= last <= cur.high * 1.15):
                print(
                    f"{symbol} P&F PRICE SANITY: WARN "
                    f"(last {last:.2f} outside approx range {cur.low:.2f}..{cur.high:.2f})"
                )

        print(f"{symbol} P&F VERIFY:", "PASS" if ok else "FAIL")
        if not ok:
            for p in problems:
                print(" -", p)

    # -------------------------
    # Data fetch
    # -------------------------

    def _safe_fetch_series(self, symbol: str, lookback_days: int) -> Optional[Tuple[List[float], List[float]]]:
        bars = self._data.get_daily_bars(symbol, lookback_days)
        if not bars:
            return None

        closes: List[float] = []
        vols: List[float] = []

        for b in bars:
            if b.close is None or b.close <= 0:
                continue
            v = b.volume if (b.volume is not None and b.volume >= 0) else 0.0
            closes.append(float(b.close))
            vols.append(float(v))

        if not closes or len(closes) != len(vols):
            return None
        return closes, vols

    # -------------------------
    # Liquidity scoring
    # -------------------------

    def _avg_dollar_vol(self, closes: List[float], vols: List[float], n: int = 20) -> float:
        if len(closes) < n or len(vols) < n:
            return 0.0
        total = 0.0
        for c, v in zip(closes[-n:], vols[-n:]):
            total += c * v
        return total / float(n)

    def _liquidity_score(self, dv20: float) -> float:
        """
        Map dollar volume to 0..1 using log scaling.
        Reference points (CAD):
          1M/day   -> low
          10M/day  -> OK
          50M/day  -> strong
          200M/day -> very strong
        """
        if dv20 <= 0:
            return 0.0
        x = math.log10(dv20)  # e.g. 7 = 10M, 8 = 100M
        return max(0.0, min(1.0, (x - 6.0) / (8.3 - 6.0)))

    # -------------------------
    # Math helpers
    # -------------------------

    def _return_over(self, closes: List[float], n: int) -> float:
        if len(closes) <= n:
            return 0.0
        start = closes[-(n + 1)]
        end = closes[-1]
        if start <= 0:
            return 0.0
        return (end / start) - 1.0

    def _sma(self, closes: List[float], n: int) -> float:
        if len(closes) < n:
            return closes[-1]
        window = closes[-n:]
        return sum(window) / float(n)

    def _clip(self, x: float, lo: float = -0.5, hi: float = 0.5) -> float:
        return max(lo, min(hi, x))

    # -------------------------
    # Legacy helpers (kept for reference)
    # -------------------------

    def _regime(
            self,
            *,
            last: float,
            r6m: float,
            sma50: float,
            sma200: float,
            rs6m: Optional[float],
            rs3m: Optional[float],
    ) -> str:
        above_50 = last > sma50
        above_200 = last > sma200
        ma_bull = sma50 > sma200

        rs_ok = True
        if rs6m is not None or rs3m is not None:
            rs_ok = (rs6m is not None and rs6m > 0) or (rs3m is not None and rs3m > 0)

        if r6m > 0 and above_50 and (ma_bull or above_200) and rs_ok:
            return "BULLISH"

        rs_bad = False
        if rs6m is not None and rs3m is not None:
            rs_bad = rs6m < 0 and rs3m < 0

        if r6m < 0 and (not above_50) and (not above_200) and rs_bad:
            return "BEARISH"

        return "NEUTRAL"

    def _score(
            self,
            *,
            r1m: float,
            r3m: float,
            r6m: float,
            rs6m: Optional[float],
            liquidity_score: float,
    ) -> float:
        c1 = self._clip(r1m, -0.3, 0.3)
        c3 = self._clip(r3m, -0.5, 0.5)
        c6 = self._clip(r6m, -0.6, 0.6)
        crs = self._clip(rs6m, -0.4, 0.4) if rs6m is not None else 0.0
        return (40.0 * c6) + (25.0 * c3) + (10.0 * c1) + (20.0 * crs) + (5.0 * liquidity_score)

    def _note(
            self,
            r1m: float,
            r3m: float,
            r6m: float,
            sma50: float,
            sma200: float,
            rs3m: Optional[float],
            rs6m: Optional[float],
            dv20: float,
    ) -> str:
        def pct(x: Optional[float]) -> str:
            if x is None:
                return "na"
            return f"{x * 100:.1f}%"

        def dv(x: float) -> str:
            if x >= 1_000_000_000:
                return f"{x / 1_000_000_000:.2f}B"
            if x >= 1_000_000:
                return f"{x / 1_000_000:.2f}M"
            if x >= 1_000:
                return f"{x / 1_000:.2f}K"
            return f"{x:.0f}"

        return (
            f"1M={pct(r1m)} 3M={pct(r3m)} 6M={pct(r6m)} "
            f"SMA50={sma50:.2f} SMA200={sma200:.2f} "
            f"RS3M={pct(rs3m)} RS6M={pct(rs6m)} "
            f"DV20={dv(dv20)}"
        )

    def _error_row(self, sym: str, msg: str) -> ScanResultRow:
        # Log message so failures are not silent, even though the UI contract has no note field.
        print(f"{sym}: {msg}")
        return ScanResultRow(
            symbol=sym,
            regime="ERROR",
            score=-9999.0,
            pf_active_name="none",
            pf_active_trigger=None,
            pf_active_cs=None,
            pf_buy_trigger=None,
            pf_buy_cs=None,
            pf_sell_trigger=None,
            pf_sell_cs=None,
            pf_cur_type="?",
            pf_cur_low=0.0,
            pf_cur_high=0.0,
            pf_entry_phase="n/a",
            pf_entry_score=0.0,
            pf_boxes_from_trigger=0.0
        )


def run_self_test(data_provider: MarketDataProvider, symbol: str, lookback_days: int = 400) -> ScanResultRow:
    """Minimal unit-style sanity check for the scanner contract and PF column type."""
    scanner = DomainScanner(data_provider=data_provider, enable_audit=False)

    closes_vols = scanner._safe_fetch_series(symbol, lookback_days)
    if closes_vols is None:
        raise RuntimeError(f"self-test: no data for {symbol}")

    closes, _vols = closes_vols
    chart = build_pf_from_closes(closes, box_mode="percent", box_value=0.015, reversal=3)
    if chart.columns and chart.current.col_type not in {"X", "O"}:
        raise RuntimeError(f"self-test: unexpected PFColumn col_type={chart.current.col_type!r}")

    rows, _closes_cache = scanner.scan([symbol], lookback_days)
    if not rows:
        raise RuntimeError("self-test: scanner returned no rows")

    row = rows[0]
    required_strs = (row.symbol, row.regime, row.pf_active_name, row.pf_cur_type)
    if any(not isinstance(v, str) or not v for v in required_strs):
        raise RuntimeError(f"self-test: required string fields missing: {row}")
    if not isinstance(row.score, float):
        raise RuntimeError(f"self-test: score is not float: {row.score!r}")
    return row
