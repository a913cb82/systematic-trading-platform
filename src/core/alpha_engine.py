from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_platform import DataPlatform

# Global Feature Registry
FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
FEATURE_DEPS: Dict[str, List[str]] = {}


def feature(
    name: str, dependencies: Optional[List[str]] = None
) -> Callable[[Any], Any]:
    def decorator(func: Callable[[pd.DataFrame], pd.Series]) -> Callable:
        FEATURES[name] = func
        FEATURE_DEPS[name] = dependencies or []
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


class AlphaModel:
    def __init__(self, data: DataPlatform, features: List[str]):
        self.data = data
        self.feature_names = features
        # Ensure all features are registered
        import src.alpha_library.features  # noqa: F401

    def _hydrate_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        seen: Optional[set[str]] = None,
    ) -> None:
        if seen is None:
            seen = set()

        for f in feature_names:
            if f in seen or f not in FEATURES:
                continue

            # Recursively hydrate dependencies first
            deps = FEATURE_DEPS.get(f, [])
            self._hydrate_features(df, deps, seen)

            # Compute feature
            df[f] = FEATURES[f](df)
            seen.add(f)

    def generate_forecasts(
        self,
        internal_ids: List[int],
        timestamp: datetime,
        lookback_days: int = 30,
    ) -> Dict[int, float]:
        start = timestamp - pd.Timedelta(days=lookback_days)
        df = self.data.get_bars(internal_ids, start, timestamp)

        # Hydrate requested features and all their dependencies
        self._hydrate_features(df, self.feature_names)

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
