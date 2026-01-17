from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Generator, List, Optional

import numpy as np
import pandas as pd

from .data_platform import DataPlatform, Event, QueryConfig

# Global Feature Registry
FEATURES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
FEATURE_DEPS: Dict[str, List[str]] = {}

# Thread-safe context variables
_context_data: ContextVar[Optional[DataPlatform]] = ContextVar(
    "context_data", default=None
)
_context_as_of: ContextVar[Optional[datetime]] = ContextVar(
    "context_as_of", default=None
)


@contextmanager
def alpha_context(
    data: DataPlatform, timestamp: datetime
) -> Generator[None, None, None]:
    """Manages the execution context for AlphaModels."""
    t1 = _context_data.set(data)
    t2 = _context_as_of.set(timestamp)
    try:
        yield
    finally:
        _context_data.reset(t1)
        _context_as_of.reset(t2)


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
    def __init__(self) -> None:
        self.feature_names: List[str] = []

    @property
    def context_as_of(self) -> Optional[datetime]:
        return _context_as_of.get()

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
        data = _context_data.get()
        as_of = _context_as_of.get()
        if data is None:
            raise RuntimeError("get_events called outside of model execution.")
        return data.get_events(
            iids,
            types=types,
            start=start,
            end=end,
            as_of=as_of,
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
        with alpha_context(data, config.timestamp):
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
