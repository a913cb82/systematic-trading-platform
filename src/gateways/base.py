from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

import pandas as pd


class DataProvider(ABC):
    @abstractmethod
    def fetch_bars(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch bars for given tickers and range."""
        pass

    @abstractmethod
    def fetch_corporate_actions(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Returns a DataFrame with columns: [ticker, ex_date, type, ratio]"""
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
