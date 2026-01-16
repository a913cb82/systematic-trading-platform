from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_platform import BAR_COLS, DataPlatform

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
    def __init__(self) -> None:
        self.feature_names: List[str] = []

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        raise NotImplementedError


class AlphaEngine:
    @staticmethod
    def _hydrate_features(
        df: pd.DataFrame,
        feature_names: List[str],
        seen: Optional[set[str]] = None,
    ) -> None:
        if seen is None:
            seen = set()

        for f in feature_names:
            if f in seen or f not in FEATURES:
                continue

            deps = FEATURE_DEPS.get(f, [])
            AlphaEngine._hydrate_features(df, deps, seen)
            df[f] = FEATURES[f](df)
            seen.add(f)

    @staticmethod
    def run_model(
        data: DataPlatform,
        model: AlphaModel,
        internal_ids: List[int],
        timestamp: datetime,
        lookback_days: int = 30,
    ) -> Dict[int, float]:
        # Ensure all features are registered
        import src.alpha_library.features  # noqa: F401

        start = timestamp - pd.Timedelta(days=lookback_days)
        df = data.get_bars(internal_ids, start, timestamp)

        AlphaEngine._hydrate_features(df, model.feature_names)

        available_features = [
            f for f in model.feature_names if f in df.columns
        ]
        df_filtered = df[BAR_COLS + available_features]

        latest = df_filtered[df_filtered["timestamp"] == timestamp].set_index(
            "internal_id"
        )

        return model.compute_signals(latest)


class SignalCombiner:
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
