from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from .data_platform import DataPlatform

# Global Feature Registry
FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def feature(name: str) -> Callable[[Any], Any]:
    def decorator(func: Callable[[pd.DataFrame], pd.Series]) -> Callable:
        FEATURES[name] = func
        return func

    return decorator


class SignalProcessor:
    @staticmethod
    def zscore(signals: Dict[int, float]) -> Dict[int, float]:
        if not signals:
            return {}
        vals = np.array(list(signals.values()))
        mu, std = vals.mean(), vals.std()
        if std == 0:
            return {k: 0.0 for k in signals}
        return {k: float((v - mu) / std) for k, v in signals.items()}

    @staticmethod
    def winsorize(
        signals: Dict[int, float], limit: float = 3.0
    ) -> Dict[int, float]:
        return {
            k: float(np.clip(v, -limit, limit)) for k, v in signals.items()
        }

    @staticmethod
    def apply_decay(
        signal: float,
        initial_ts: datetime,
        current_ts: datetime,
        half_life_mins: float,
    ) -> float:
        delta = (current_ts - initial_ts).total_seconds() / 60.0
        return float(signal * np.exp(-np.log(2) * delta / half_life_mins))

    @staticmethod
    def apply_linear_decay(
        signal: float,
        initial_ts: datetime,
        current_ts: datetime,
        duration_mins: float,
    ) -> float:
        delta = (current_ts - initial_ts).total_seconds() / 60.0
        if delta >= duration_mins:
            return 0.0
        return float(signal * (1 - delta / duration_mins))

    @staticmethod
    def rank_transform(signals: Dict[int, float]) -> Dict[int, float]:
        if not signals:
            return {}
        s = pd.Series(signals).rank(pct=True)
        return cast(Dict[int, float], s.to_dict())


class AlphaModel:
    def __init__(self, data: DataPlatform, features: List[str]):
        self.data = data
        self.feature_names = features

    def generate_forecasts(
        self,
        internal_ids: List[int],
        timestamp: datetime,
        lookback_days: int = 30,
    ) -> Dict[int, float]:
        start = timestamp - pd.Timedelta(days=lookback_days)
        df = self.data.get_bars(internal_ids, start, timestamp)

        # Hydrate Features
        for f in self.feature_names:
            if f in FEATURES:
                df[f] = FEATURES[f](df)

        latest = df[df["timestamp"] == timestamp].set_index("internal_id")

        # Model Logic (passing latest state and full hydrated history)
        return self.compute_signals(latest, df)

    def compute_signals(
        self, latest: pd.DataFrame, history: pd.DataFrame
    ) -> Dict[int, float]:
        raise NotImplementedError


class SignalCombiner:
    """
    Combines multiple signals using fixed linear weights.
    """

    @staticmethod
    def combine(
        signals_list: List[Dict[int, float]],
        weights: Optional[List[float]] = None,
    ) -> Dict[int, float]:
        if not signals_list:
            return {}
        if weights is None:
            weights = [1.0 / len(signals_list)] * len(signals_list)

        combined: Dict[int, float] = {}
        for signals, weight in zip(signals_list, weights):
            for iid, val in signals.items():
                combined[iid] = combined.get(iid, 0.0) + val * weight
        return combined


class BacktestValidator:
    """
    Implements Sliding and Expanding Window validation frameworks.
    """

    @staticmethod
    def get_windows(
        start_date: datetime,
        end_date: datetime,
        window_size_days: int,
        step_days: int,
        expanding: bool = False,
    ) -> List[tuple[datetime, datetime, datetime]]:
        """
        Returns a list of (train_start, train_end, test_end)
        """
        windows = []
        current_train_end = start_date + pd.Timedelta(days=window_size_days)

        while current_train_end + pd.Timedelta(days=step_days) <= end_date:
            train_start = (
                start_date
                if expanding
                else current_train_end - pd.Timedelta(days=window_size_days)
            )
            test_end = current_train_end + pd.Timedelta(days=step_days)
            windows.append((train_start, current_train_end, test_end))
            current_train_end += pd.Timedelta(days=step_days)

        return windows
