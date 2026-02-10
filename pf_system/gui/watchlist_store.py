from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class WatchlistStore:
    """
    Global tracked symbols persistence.

    Disk format: single line comma-separated.
    Loader accepts comma OR newline-separated.
    """

    path: Path

    def load(self) -> set[str]:
        if not self.path.exists():
            return set()

        txt = self.path.read_text(encoding="utf-8").strip()
        if not txt:
            return set()

        parts = [p.strip().upper() for p in txt.replace("\n", ",").split(",")]
        return {p for p in parts if p}

    def save(self, symbols: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = ",".join(sorted(symbols))
        self.path.write_text(line + "\n", encoding="utf-8")
