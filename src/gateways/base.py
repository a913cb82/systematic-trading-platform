from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List

import pandas as pd

from src.core.types import Bar, Timeframe


class BarProvider(ABC):
    @abstractmethod
    def fetch_bars(
        self,
        tickers: List[str],
        start: datetime,
        end: datetime,
        timeframe: Timeframe = Timeframe.DAY,
    ) -> pd.DataFrame:
        """Fetch historical bars."""
        pass


class CorporateActionProvider(ABC):
    @abstractmethod
    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch historical splits and dividends."""
        pass


class EventProvider(ABC):
    @abstractmethod
    def fetch_events(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch historical fundamental or other events."""
        pass


class StreamProvider(ABC):
    @abstractmethod
    def subscribe(
        self, tickers: List[str], handler: Callable[[Bar], None]
    ) -> None:
        """Subscribes to live data for given tickers."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Starts the realtime event loop."""
        pass


class ExecutionBackend(ABC):
    @abstractmethod
    def submit_order(self, ticker: str, quantity: float, side: str) -> bool:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Returns ticker -> quantity"""
        pass

    @abstractmethod
    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        pass
