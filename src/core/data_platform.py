from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from src.gateways.base import DataProvider


@dataclass
class Bar:
    internal_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp_knowledge: datetime = field(default_factory=datetime.now)


@dataclass
class Event:
    internal_id: int
    timestamp: datetime
    event_type: str
    value: Any
    timestamp_knowledge: datetime = field(default_factory=datetime.now)


@dataclass
class CorporateAction:
    internal_id: int
    ex_date: datetime
    type: str  # 'SPLIT' or 'DIVIDEND'
    ratio: float


class DataPlatform:
    def __init__(self, provider: Optional[DataProvider] = None) -> None:
        self.provider = provider
        self.sec_df = pd.DataFrame(
            columns=["internal_id", "ticker", "start", "end", "extra"]
        ).astype(
            {
                "internal_id": "int64",
                "ticker": "string",
                "start": "datetime64[ns]",
                "end": "datetime64[ns]",
            }
        )
        self.bars: List[Bar] = []
        self.events: List[Event] = []
        self.ca_df = pd.DataFrame(
            columns=["internal_id", "ex_date", "type", "ratio"]
        ).astype(
            {
                "internal_id": "int64",
                "ex_date": "datetime64[ns]",
                "type": "string",
                "ratio": "float64",
            }
        )
        self._id_counter = 1000

    def register_security(
        self,
        ticker: str,
        internal_id: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs: Any,
    ) -> int:
        start_dt = start or datetime(1900, 1, 1)
        end_dt = end or datetime(2100, 1, 1)
        if internal_id is None:
            mask = (
                (self.sec_df["ticker"] == ticker)
                & (self.sec_df["start"] <= end_dt)
                & (self.sec_df["end"] >= start_dt)
            )
            existing = self.sec_df[mask]
            if not existing.empty:
                return int(existing.iloc[0]["internal_id"])
            internal_id = self._id_counter
            self._id_counter += 1
        new_row = pd.DataFrame(
            [
                {
                    "internal_id": internal_id,
                    "ticker": ticker,
                    "start": start_dt,
                    "end": end_dt,
                    "extra": kwargs,
                }
            ]
        )
        self.sec_df = pd.concat([self.sec_df, new_row], ignore_index=True)
        return internal_id

    def get_internal_id(
        self, ticker: str, date: Optional[datetime] = None
    ) -> int:
        date = date or datetime.now()
        mask = (
            (self.sec_df["ticker"] == ticker)
            & (self.sec_df["start"] <= date)
            & (self.sec_df["end"] >= date)
        )
        res = self.sec_df[mask]
        if res.empty:
            return self.register_security(ticker, start=date)
        return int(res.iloc[0]["internal_id"])

    @property
    def reverse_ism(self) -> Dict[int, str]:
        if self.sec_df.empty:
            return {}
        return cast(
            Dict[int, str],
            self.sec_df.sort_values("start")
            .groupby("internal_id")["ticker"]
            .last()
            .to_dict(),
        )

    def get_universe(self, date: datetime) -> List[int]:
        if self.sec_df.empty:
            return []
        mask = (self.sec_df["start"] <= date) & (self.sec_df["end"] >= date)
        return cast(
            List[int], self.sec_df[mask]["internal_id"].unique().tolist()
        )

    def add_events(self, events: List[Event]) -> None:
        self.events.extend(events)

    def get_events(
        self,
        internal_ids: List[int],
        event_types: Optional[List[str]] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        as_of = as_of or datetime.now()
        return [
            e
            for e in self.events
            if e.internal_id in internal_ids
            and (event_types is None or e.event_type in event_types)
            and e.timestamp_knowledge <= as_of
        ]

    def add_bars(self, bars: List[Bar]) -> None:
        self.bars.extend(bars)

    def add_ca(self, ca: CorporateAction) -> None:
        new_row = pd.DataFrame([ca.__dict__])
        self.ca_df = pd.concat([self.ca_df, new_row]).drop_duplicates()

    def sync_data(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> None:
        if not self.provider:
            return
        raw_bars = self.provider.fetch_bars(tickers, start, end)
        now = datetime.now()
        for _, row in raw_bars.iterrows():
            iid = self.get_internal_id(row["ticker"], row["timestamp"])
            self.bars.append(
                Bar(
                    iid,
                    row["timestamp"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    now,
                )
            )
        raw_ca = self.provider.fetch_corporate_actions(tickers, start, end)
        if not raw_ca.empty:
            for _, row in raw_ca.iterrows():
                iid = self.get_internal_id(row["ticker"], row["ex_date"])
                self.add_ca(
                    CorporateAction(
                        iid, row["ex_date"], row["type"], row["ratio"]
                    )
                )

    def get_bars(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
        adjust: bool = True,
    ) -> pd.DataFrame:
        as_of = as_of or datetime.now()
        valid = [
            b
            for b in self.bars
            if b.internal_id in internal_ids
            and start <= b.timestamp <= end
            and b.timestamp_knowledge <= as_of
        ]
        if not valid:
            return pd.DataFrame()
        df = (
            pd.DataFrame(valid)
            .sort_values("timestamp_knowledge")
            .groupby(["internal_id", "timestamp"])
            .last()
            .reset_index()
        )
        if adjust and not self.ca_df.empty:
            for iid in internal_ids:
                iid_ca = self.ca_df[
                    (self.ca_df["internal_id"] == iid)
                    & (self.ca_df["ex_date"] <= end)
                ]
                for _, ca in iid_ca.sort_values(
                    "ex_date", ascending=False
                ).iterrows():
                    mask = (df["internal_id"] == iid) & (
                        df["timestamp"] < ca["ex_date"]
                    )
                    if ca["type"] == "SPLIT":
                        df.loc[mask, ["open", "high", "low", "close"]] /= ca[
                            "ratio"
                        ]
                    elif ca["type"] == "DIVIDEND":
                        df.loc[mask, ["open", "high", "low", "close"]] *= ca[
                            "ratio"
                        ]
        return df
