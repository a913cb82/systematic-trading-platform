from typing import Any
from abc import ABC, abstractmethod
from datetime import datetime
from .types import Bar, Event, Trade


class MarketDataProvider(ABC):
    @abstractmethod
    def write_bars(self, data: list[Bar]) -> None:
        pass

    @abstractmethod
    def get_bars(
        self,
        internal_ids: list[int],
        start: datetime,
        end: datetime,
        adjustment: str,
        as_of: datetime | None = None,
    ) -> list[Bar]:
        pass

    @abstractmethod
    def subscribe_bars(self, internal_ids: list[int], on_bar: Any) -> None:
        pass

    @abstractmethod
    def get_universe(self, date: datetime) -> list[int]:
        pass


class AlphaModel(ABC):
    @abstractmethod
    def generate_forecasts(self, timestamp: datetime) -> dict[int, float]:
        pass


class RiskModel(ABC):
    @abstractmethod
    def get_covariance_matrix(
        self, date: datetime, internal_ids: list[int]
    ) -> list[list[float]]:
        pass

    @abstractmethod
    def get_factor_exposures(
        self, date: datetime, internal_ids: list[int]
    ) -> dict[int, dict[str, float]]:
        pass


class PortfolioOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        timestamp: datetime,
        forecasts: dict[int, float],
        current_weights: dict[int, float] | None = None,
    ) -> dict[int, float]:
        pass


class ExecutionEngine(ABC):
    @abstractmethod
    def execute(
        self, timestamp: datetime, target_weights: dict[int, float]
    ) -> None:
        pass

    @abstractmethod
    def report_fill(self, fill: Trade) -> None:
        pass

    @abstractmethod
    def get_fills(
        self, start_time: datetime, end_time: datetime
    ) -> list[Trade]:
        pass

    @abstractmethod
    def subscribe_fills(self, on_fill: Any) -> None:
        pass
