from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .types import Bar, ModelState, Trade


class MarketDataProvider(ABC):
    @abstractmethod
    def write_bars(self, data: list[Bar], fill_gaps: bool = False) -> None:
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

    @abstractmethod
    def get_returns(
        self,
        internal_ids: list[int],
        date_range: tuple[datetime, datetime],
        type: str = "RAW",
        as_of: datetime | None = None,
        risk_model: Any = None,
    ) -> Any:
        pass


class AlphaModel(ABC):
    @abstractmethod
    def generate_forecasts(self, timestamp: datetime) -> dict[int, float]:
        pass

    @abstractmethod
    def on_cycle(
        self, timestamp: datetime, state: ModelState
    ) -> dict[int, float]:
        """
        Maximally simple API for users to override.
        """
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


class BaseAlphaModel(AlphaModel):
    """
    Standard wrapper that handles state construction.
    """

    def __init__(
        self,
        market_data: MarketDataProvider,
        internal_ids: list[int],
        risk_model: RiskModel | None = None,
    ) -> None:
        self.market_data = market_data
        self.internal_ids = internal_ids
        self.risk_model = risk_model

    def generate_forecasts(self, timestamp: datetime) -> dict[int, float]:
        # 1. Build Model State
        from datetime import timedelta

        import pandas as pd

        start = timestamp - timedelta(days=5)
        bars_list = self.market_data.get_bars(
            self.internal_ids, start, timestamp, adjustment="RATIO"
        )
        bars_df = pd.DataFrame(bars_list)

        raw_returns = self.market_data.get_returns(
            self.internal_ids, (start, timestamp), type="RAW"
        )

        residual_returns = self.market_data.get_returns(
            self.internal_ids,
            (start, timestamp),
            type="RESIDUAL",
            risk_model=self.risk_model,
        )

        state: ModelState = {
            "timestamp": timestamp,
            "universe": self.internal_ids,
            "bars": bars_df,
            "returns": raw_returns,
            "residuals": residual_returns,
            "positions": {},
            "metadata": {},
        }

        # 2. Delegate to on_cycle
        return self.on_cycle(timestamp, state)

    def on_cycle(
        self, timestamp: datetime, state: ModelState
    ) -> dict[int, float]:
        return {}


class PortfolioOptimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        timestamp: datetime,
        forecasts: dict[int, float],
        current_weights: dict[int, float] | None = None,
        factor_returns: dict[str, float] | None = None,
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
