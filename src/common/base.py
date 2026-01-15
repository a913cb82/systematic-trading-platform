from abc import ABC, abstractmethod
from datetime import datetime
from .types import Bar, Event, Trade

class MarketDataProvider(ABC):
    @abstractmethod
    def get_bars(self, internal_ids: list[int], start: datetime, end: datetime, adjustment: str, as_of: datetime | None = None) -> list[Bar]:
        pass

    @abstractmethod
    def get_universe(self, date: datetime) -> list[int]:
        pass

class AlphaModel(ABC):
    @abstractmethod
    def generate_forecasts(self, timestamp: datetime) -> dict[int, float]:
        pass

class PortfolioOptimizer(ABC):
    @abstractmethod
    def optimize(self, timestamp: datetime, forecasts: dict[int, float]) -> dict[int, float]:
        pass

class ExecutionEngine(ABC):
    @abstractmethod
    def execute(self, timestamp: datetime, target_weights: dict[int, float]) -> None:
        pass
