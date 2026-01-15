from datetime import datetime
from typing import Any, Callable, Dict, List, cast

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


@feature("sma_10")
def sma_10(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        df.groupby("internal_id")["close"].transform(
            lambda x: x.rolling(10).mean()
        ),
    )


@feature("rsi_14")
def rsi_14(df: pd.DataFrame) -> pd.Series:
    def _rsi(x: pd.Series) -> pd.Series:
        delta = x.diff()
        u, d = delta.clip(lower=0), -delta.clip(upper=0)
        ma_u, ma_d = u.rolling(14).mean(), d.rolling(14).mean()
        return cast(pd.Series, 100 - (100 / (1 + ma_u / ma_d)))

    return cast(pd.Series, df.groupby("internal_id")["close"].transform(_rsi))


@feature("macd")
def macd(df: pd.DataFrame) -> pd.Series:
    def _macd(x: pd.Series) -> pd.Series:
        return cast(pd.Series, x.ewm(span=12).mean() - x.ewm(span=26).mean())

    return cast(pd.Series, df.groupby("internal_id")["close"].transform(_macd))


@feature("ofi")
def ofi(df: pd.DataFrame) -> pd.Series:
    # Order Flow Imbalance (Simplified: Buy Vol - Sell Vol)
    # Note: Requires buy_volume/sell_volume in Bar.
    return cast(
        pd.Series, df["volume"] * np.where(df["close"] >= df["open"], 1, -1)
    )


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
        # 1. Get Residual Returns (Hedge Fund Guide mandates this)
        returns = self.data.get_returns(internal_ids, start, timestamp)

        # 2. Hydrate Features
        df = self.data.get_bars(internal_ids, start, timestamp)
        for f in self.feature_names:
            df[f] = FEATURES[f](df)

        latest = df[df["timestamp"] == timestamp].set_index("internal_id")

        # 3. Model Logic (to be overridden)
        return self.compute_signals(latest, returns)

    def compute_signals(
        self, latest_features: pd.DataFrame, returns: pd.DataFrame
    ) -> Dict[int, float]:
        raise NotImplementedError
