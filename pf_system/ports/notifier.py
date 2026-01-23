from __future__ import annotations

from abc import ABC, abstractmethod


class Notifier(ABC):
    @abstractmethod
    def notify(self, title: str, message: str) -> None:
        raise NotImplementedError
