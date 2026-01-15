from datetime import datetime
from typing import Any, Callable, Dict, List, cast

import pandas as pd

from ..data.event_store import EventStore
from ..data.market_data import MarketDataEngine

# Registry for feature calculation functions
# Key: feature_name, Value: Callable[[pd.DataFrame], pd.Series]
CYCLE_FEATURE_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
EVENT_FEATURE_REGISTRY: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}


def register_cycle_feature(name: str) -> Callable[[Any], Any]:
    def decorator(
        func: Callable[[pd.DataFrame], pd.Series],
    ) -> Callable[[pd.DataFrame], pd.Series]:
        CYCLE_FEATURE_REGISTRY[name] = func
        return func

    return decorator


def register_event_feature(name: str) -> Callable[[Any], Any]:
    def decorator(
        func: Callable[[pd.DataFrame], pd.Series],
    ) -> Callable[[pd.DataFrame], pd.Series]:
        EVENT_FEATURE_REGISTRY[name] = func
        return func

    return decorator


class FeatureStore:
    def __init__(self, market_data: MarketDataEngine, event_store: EventStore):
        self.market_data = market_data
        self.event_store = event_store

    def calculate_cycle_feature(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        feature_name: str,
    ) -> pd.DataFrame:
        """
        Calculates a feature based on cycle data (bars).
        """
        if feature_name not in CYCLE_FEATURE_REGISTRY:
            raise ValueError(f"Unknown cycle feature: {feature_name}")

        bars = self.market_data.get_bars(internal_ids, start, end)
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["internal_id", "timestamp"])

        calc_func = CYCLE_FEATURE_REGISTRY[feature_name]
        df["feature"] = calc_func(df)

        return df[["internal_id", "timestamp", "feature"]]

    def calculate_event_feature(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        feature_name: str,
    ) -> pd.DataFrame:
        """
        Calculates a feature based on event data.
        """
        if feature_name not in EVENT_FEATURE_REGISTRY:
            raise ValueError(f"Unknown event feature: {feature_name}")

        # This assumes the feature knows which event types it needs
        # In a real system, we'd store required event types in the registry.
        event_type = "EARNINGS" if feature_name == "earnings_surprise" else ""
        events = self.event_store.get_events(
            [event_type], internal_ids, start, end
        )
        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        calc_func = EVENT_FEATURE_REGISTRY[feature_name]
        df["feature"] = calc_func(df)

        return df[["internal_id", "timestamp_event", "feature"]].rename(
            columns={"timestamp_event": "timestamp"}
        )


# --- Default Features ---


@register_cycle_feature("returns_1d")
def calc_returns_1d(df: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, df.groupby("internal_id")["close"].pct_change())


@register_cycle_feature("sma_5")
def calc_sma_5(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        df.groupby("internal_id")["close"].transform(
            lambda x: x.rolling(5).mean()
        ),
    )


@register_cycle_feature("rsi_14")
def calc_rsi_14(df: pd.DataFrame) -> pd.Series:
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return cast(pd.Series, 100 - (100 / (1 + rs)))

    return cast(pd.Series, df.groupby("internal_id")["close"].transform(_rsi))


@register_cycle_feature("macd")
def calc_macd(df: pd.DataFrame) -> pd.Series:
    def _macd(series: pd.Series) -> pd.Series:
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        return cast(pd.Series, exp1 - exp2)

    return cast(pd.Series, df.groupby("internal_id")["close"].transform(_macd))


@register_cycle_feature("ofi")
def calc_ofi(df: pd.DataFrame) -> pd.Series:
    return cast(pd.Series, df["buy_volume"] - df["sell_volume"])


@register_event_feature("earnings_surprise")
def calc_earnings_surprise(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        df["value"].apply(
            lambda x: x.get("surprise", 0) if isinstance(x, dict) else 0
        ),
    )
