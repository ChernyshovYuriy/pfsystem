from __future__ import annotations

import math
from typing import List, Sequence, Optional

from pf_system.domain.models import ScanResultRow
from pf_system.ports.data_provider import MarketDataProvider


class DomainScanner:
    """
    Domain service.
    V1 implementation: momentum + trend + relative strength vs benchmark.
    Later: replace with true P&F regime + signals while keeping the same output shape.
    """

    def __init__(self, data_provider: MarketDataProvider, benchmark: str = "XIU.TO") -> None:
        self._data = data_provider
        self._benchmark = benchmark

    def scan(self, symbols: Sequence[str], lookback_days: int) -> List[ScanResultRow]:
        out: List[ScanResultRow] = []

        # Fetch benchmark once (for RS)
        bench_series = self._safe_fetch_series(self._benchmark, lookback_days)
        bench_closes = bench_series[0] if bench_series else None
        if bench_closes is None or len(bench_closes) < 130:
            bench_closes = None

        if bench_closes is None or len(bench_closes) < 130:
            # If benchmark fails, we still scan but RS is disabled.
            bench_closes = None

        for sym in symbols:
            try:
                series = self._safe_fetch_series(sym, lookback_days)
                if series is None:
                    out.append(ScanResultRow(sym, "ERROR", -9999.0, "insufficient data"))
                    continue
                closes, vols = series
                if len(closes) < 210:
                    out.append(ScanResultRow(sym, "ERROR", -9999.0, "insufficient data"))
                    continue

                last = closes[-1]

                # For audit only.
                # self._audit_symbol(sym, 252)

                # Returns
                r1m = self._return_over(closes, 21)
                r3m = self._return_over(closes, 63)
                r6m = self._return_over(closes, 126)

                # Trend (SMA)
                sma50 = self._sma(closes, 50)
                sma200 = self._sma(closes, 200)

                # Relative strength vs benchmark (return differential)
                rs3m = None
                rs6m = None
                if bench_closes is not None and len(bench_closes) >= len(closes):
                    br3m = self._return_over(bench_closes, 63)
                    br6m = self._return_over(bench_closes, 126)
                    rs3m = r3m - br3m
                    rs6m = r6m - br6m

                dv20 = self._avg_dollar_vol(closes, vols, 20)
                liq = self._liquidity_score(dv20)

                regime = self._regime(
                    last=last,
                    r6m=r6m,
                    sma50=sma50,
                    sma200=sma200,
                    rs6m=rs6m,
                    rs3m=rs3m,
                )

                score = self._score(
                    r1m=r1m,
                    r3m=r3m,
                    r6m=r6m,
                    rs6m=rs6m,
                    liquidity_score=liq,
                )

                note = self._note(r1m, r3m, r6m, sma50, sma200, rs3m, rs6m, dv20)

                out.append(ScanResultRow(sym, regime, score, note))

            except Exception as e:
                out.append(ScanResultRow(sym, "ERROR", -9999.0, f"error: {e}"))

        out.sort(key=lambda r: r.score, reverse=True)
        return out

    # -------------------------
    # Helpers
    # -------------------------

    def _audit_symbol(self, symbol: str, lookback_days: int) -> None:
        closes = self._safe_fetch_series(symbol, lookback_days)
        if closes is None or len(closes) < 210:
            print(symbol, "INSUFFICIENT DATA:", 0 if closes is None else len(closes))
            return

        last = closes[-1]

        r1m = self._return_over(closes, 21)
        r3m = self._return_over(closes, 63)
        r6m = self._return_over(closes, 126)

        sma50 = self._sma(closes, 50)
        sma200 = self._sma(closes, 200)

        print("=" * 60)
        print("SYMBOL:", symbol)
        print("BARS:", len(closes))
        print(f"LAST CLOSE: {last:.2f}")

        print("1M START:", closes[-22], "END:", last, "RET:", f"{r1m * 100:.2f}%")
        print("3M START:", closes[-64], "END:", last, "RET:", f"{r3m * 100:.2f}%")
        print("6M START:", closes[-127], "END:", last, "RET:", f"{r6m * 100:.2f}%")

        print(f"SMA50:  {sma50:.2f}")
        print(f"SMA200: {sma200:.2f}")

        if symbol != self._benchmark:
            bench = self._safe_fetch_series(self._benchmark, lookback_days)
            if bench:
                br3m = self._return_over(bench, 63)
                br6m = self._return_over(bench, 126)
                print(f"RS3M: {(r3m - br3m) * 100:.2f}%")
                print(f"RS6M: {(r6m - br6m) * 100:.2f}%")

    def _safe_fetch_series(self, symbol: str, lookback_days: int) -> Optional[tuple[list[float], list[float]]]:
        bars = self._data.get_daily_bars(symbol, lookback_days)
        if not bars:
            return None

        closes: list[float] = []
        vols: list[float] = []
        for b in bars:
            if b.close is None or b.close <= 0:
                continue
            v = b.volume if b.volume is not None and b.volume >= 0 else 0.0
            closes.append(float(b.close))
            vols.append(float(v))

        if not closes or len(closes) != len(vols):
            return None
        return closes, vols

    def _safe_fetch_closes(self, symbol: str, lookback_days: int) -> Optional[List[float]]:
        bars = self._data.get_daily_bars(symbol, lookback_days)
        if not bars:
            return None
        closes = [b.close for b in bars if b.close is not None and b.close > 0]
        return closes if closes else None

    def _avg_dollar_vol(self, closes: list[float], vols: list[float], n: int = 20) -> float:
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
        # Scale: 6.0 (1M) -> 0, 8.3 (~200M) -> 1
        return max(0.0, min(1.0, (x - 6.0) / (8.3 - 6.0)))

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
            # require at least one positive RS measure
            rs_ok = ((rs6m is not None and rs6m > 0) or (rs3m is not None and rs3m > 0))

        if r6m > 0 and above_50 and (ma_bull or above_200) and rs_ok:
            return "BULLISH"

        rs_bad = False
        if rs6m is not None or rs3m is not None:
            rs_bad = ((rs6m is not None and rs6m < 0) and (rs3m is not None and rs3m < 0))

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
        # Clip to keep outliers from dominating
        c1 = self._clip(r1m, -0.3, 0.3)
        c3 = self._clip(r3m, -0.5, 0.5)
        c6 = self._clip(r6m, -0.6, 0.6)
        crs = self._clip(rs6m, -0.4, 0.4) if rs6m is not None else 0.0

        # Scale to a “nice” range ~ 0..100-ish but can go outside a bit
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
            # Pretty-print as M/B
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
