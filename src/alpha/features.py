import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from ..data.market_data import MarketDataEngine
from ..data.event_store import EventStore


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
        Example: 'returns_5d', 'volatility_20d', 'ma_50'
        """
        # Fetch enough data for lookback if needed
        # For simplicity in PoC, just fetch start to end
        bars = self.market_data.get_bars(internal_ids, start, end)
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["internal_id", "timestamp"])

        if feature_name == "returns_1d":
            df["feature"] = df.groupby("internal_id")["close"].pct_change()
        elif feature_name == "sma_5":
            df["feature"] = df.groupby("internal_id")["close"].transform(
                lambda x: x.rolling(5).mean()
            )
        elif feature_name == "rsi_14":

            def calc_rsi(series, period=14):
                delta = series.diff()
                gain = (
                    (delta.where(delta > 0, 0)).rolling(window=period).mean()
                )
                loss = (
                    (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                )
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df["feature"] = df.groupby("internal_id")["close"].transform(
                calc_rsi
            )
        elif feature_name == "macd":

            def calc_macd(series):
                exp1 = series.ewm(span=12, adjust=False).mean()
                exp2 = series.ewm(span=26, adjust=False).mean()
                return exp1 - exp2

            df["feature"] = df.groupby("internal_id")["close"].transform(
                calc_macd
            )
        elif feature_name == "ofi":
            # Order Flow Imbalance proxy using buy/sell volume
            df["feature"] = df["buy_volume"] - df["sell_volume"]
        else:
            raise ValueError(f"Unknown feature: {feature_name}")

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
        Example: 'earnings_surprise', 'sentiment_score'
        """
        if feature_name == "earnings_surprise":
            events = self.event_store.get_events(
                ["EARNINGS"], internal_ids, start, end
            )
            if not events:
                return pd.DataFrame()
            df = pd.DataFrame(events)
            # Assume value is dict with 'surprise'
            df["feature"] = df["value"].apply(
                lambda x: x.get("surprise", 0) if isinstance(x, dict) else 0
            )
            return df[["internal_id", "timestamp_event", "feature"]].rename(
                columns={"timestamp_event": "timestamp"}
            )
        else:
            raise ValueError(f"Unknown event feature: {feature_name}")
