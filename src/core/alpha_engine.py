from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .data_platform import DataPlatform, Event, QueryConfig

# Global Feature Registry
FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
FEATURE_DEPS: Dict[str, List[str]] = {}


@dataclass
class ModelRunConfig:
    timestamp: datetime
    timeframe: str = "1D"
    lookback_days: int = 30


def feature(
    name: str, dependencies: Optional[List[str]] = None
) -> Callable[
    [Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.Series]
]:
    def decorator(
        func: Callable[[pd.DataFrame], pd.Series],
    ) -> Callable[[pd.DataFrame], pd.Series]:
        FEATURES[name] = func
        FEATURE_DEPS[name] = dependencies or []
        return func

    return decorator


def multi_tf_feature(
    name: str, timeframes: List[str], dependencies: Optional[List[str]] = None
) -> Callable[
    [Callable[[pd.DataFrame, str], pd.Series]],
    Callable[[pd.DataFrame, str], pd.Series],
]:
    def decorator(
        func: Callable[[pd.DataFrame, str], pd.Series],
    ) -> Callable[[pd.DataFrame, str], pd.Series]:
        for tf in timeframes:
            full_name = f"{name}_{tf}"
            tf_deps = [f"{d}_{tf}" for d in (dependencies or [])]

            def make_wrapper(
                current_tf: str,
            ) -> Callable[[pd.DataFrame], pd.Series]:
                return lambda df: func(df, current_tf)

            feature(full_name, dependencies=tf_deps)(make_wrapper(tf))
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


class AlphaModel(ABC):
    _context_data: Optional[DataPlatform] = None
    _context_as_of: Optional[datetime] = None

    def __init__(self) -> None:
        self.feature_names: List[str] = []

    @staticmethod
    def get_events(
        iids: List[int],
        types: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Event]:
        """
        Static access to events during model execution.
        Automatically respects the 'as_of' time of the current run.
        """
        if AlphaModel._context_data is None:
            raise RuntimeError("get_events called outside of model execution.")
        return AlphaModel._context_data.get_events(
            iids,
            types=types,
            start=start,
            end=end,
            as_of=AlphaModel._context_as_of,
        )

    @abstractmethod
    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        """
        Calculates alpha scores for all securities in the universe.
        'latest' contains features hydrated for the target timestamp.
        """
        pass


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
        config: ModelRunConfig,
    ) -> Dict[int, float]:
        # Ensure all features are registered
        import src.alpha_library.features  # noqa: F401, PLC0415

        # 1. Fetch Bars
        start = config.timestamp - pd.Timedelta(days=config.lookback_days)
        df = data.get_bars(
            internal_ids,
            QueryConfig(
                start=start, end=config.timestamp, timeframe=config.timeframe
            ),
        )
        if df.empty:
            return {}

        # 2. Hydrate requested features
        AlphaEngine._hydrate_features(df, model.feature_names)

        # 3. Slice for latest timestamp
        latest = df[df["timestamp"] == config.timestamp].set_index(
            "internal_id"
        )

        # 4. Model execution with context
        AlphaModel._context_data = data
        AlphaModel._context_as_of = config.timestamp
        try:
            return model.compute_signals(latest)
        finally:
            AlphaModel._context_data = None
            AlphaModel._context_as_of = None


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
