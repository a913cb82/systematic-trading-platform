import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, cast

import pandas as pd
from arcticdb import Arctic, QueryBuilder

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
    timeframe: str = "1D"
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
    type: str
    ratio: float


@dataclass
class QueryConfig:
    start: datetime
    end: datetime
    timeframe: str = "1D"
    as_of: Optional[datetime] = None
    adjust: bool = True


_ARCTIC_CACHE: Dict[str, Arctic] = {}


def get_arctic(db_path: str) -> Arctic:
    if db_path not in _ARCTIC_CACHE:
        _ARCTIC_CACHE[db_path] = Arctic(f"lmdb://{db_path}")
    return _ARCTIC_CACHE[db_path]


class DataPlatform:
    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        db_path: str = "./.arctic_db",
        clear: bool = False,
    ) -> None:
        self.provider, self.arctic = provider, get_arctic(db_path)
        if clear and "platform" in self.arctic.list_libraries():
            self.arctic.delete_library("platform")
        self.lib = self.arctic.get_library("platform", create_if_missing=True)
        self._ensure_symbols()
        self._load_metadata()
        self._id_counter = (
            int(self.sec_df["internal_id"].max() + 1)
            if not self.sec_df.empty
            else 1000
        )

    def _ensure_symbols(self) -> None:
        type_map = {int: "int64", float: "float64", datetime: "datetime64[ns]"}
        for symbol, cls in {
            "sec_df": None,
            "ca_df": CorporateAction,
            "bars": Bar,
            "events": Event,
        }.items():
            if not self.lib.has_symbol(symbol):
                if cls:
                    df = pd.DataFrame(
                        {
                            f.name: pd.Series(
                                dtype=type_map.get(
                                    cast(Type, f.type), "object"
                                )
                            )
                            for f in fields(cls)
                        }
                    )
                    if symbol in ["bars", "events"]:
                        df = df.set_index("timestamp")
                else:
                    df = pd.DataFrame(
                        columns=[
                            "internal_id",
                            "ticker",
                            "start",
                            "end",
                            "extra",
                        ]
                    ).astype({"internal_id": "int64"})
                self.lib.write(symbol, df)

    def _load_metadata(self) -> None:
        self.sec_df = self.lib.read("sec_df").data.copy()
        self.sec_df["extra"] = self.sec_df["extra"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        self.ca_df = self.lib.read("ca_df").data.copy()

    def _save_metadata(self) -> None:
        sec_to_save = self.sec_df.copy()
        sec_to_save["extra"] = sec_to_save["extra"].apply(json.dumps)
        self.lib.write("sec_df", sec_to_save)
        self.lib.write("ca_df", self.ca_df)

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
            if not self.sec_df[mask].empty:
                return int(self.sec_df[mask].iloc[0]["internal_id"])
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
        self.sec_df = (
            new_row
            if self.sec_df.empty
            else pd.concat([self.sec_df, new_row], ignore_index=True)
        ).copy()
        self._save_metadata()
        return internal_id

    def get_internal_id(
        self, ticker: str, date: Optional[datetime] = None
    ) -> int:
        query_date = date or datetime.now()
        res = self.sec_df[
            (self.sec_df["ticker"] == ticker)
            & (self.sec_df["start"] <= query_date)
            & (self.sec_df["end"] >= query_date)
        ]
        return (
            int(res.iloc[0]["internal_id"])
            if not res.empty
            else self.register_security(ticker, start=query_date)
        )

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
        mask = (self.sec_df["start"] <= date) & (self.sec_df["end"] >= date)
        return cast(
            List[int], self.sec_df[mask]["internal_id"].unique().tolist()
        )

    def _accumulate(self, symbol: str, df: pd.DataFrame) -> None:
        existing = self.lib.read(symbol).data.copy()
        updated = (
            (df if existing.empty else pd.concat([existing, df]))
            .sort_index()
            .copy()
        )
        self.lib.write(symbol, updated)

    def add_events(self, events: List[Event]) -> None:
        if not events:
            return
        df = pd.DataFrame(
            [{**asdict(e), "value": json.dumps(e.value)} for e in events]
        ).set_index("timestamp")
        self._accumulate("events", df)

    def get_events(
        self,
        internal_ids: List[int],
        event_types: Optional[List[str]] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        as_of_dt = as_of or datetime.now()
        query = QueryBuilder()
        query = query[
            query.internal_id.isin(internal_ids)
            & (query.timestamp_knowledge <= as_of_dt)
        ]
        if event_types:
            query = query[query.event_type.isin(event_types)]
        df = (
            self.lib.read("events", query_builder=query)
            .data.copy()
            .reset_index()
        )
        return [
            Event(
                **{
                    **row,
                    "value": json.loads(row["value"])
                    if isinstance(row["value"], str)
                    else row["value"],
                }
            )
            for row in df.to_dict("records")
        ]

    def add_bars(self, bars: List[Bar]) -> None:
        if not bars:
            return
        df = pd.DataFrame([asdict(b) for b in bars])
        float_cols = ["open", "high", "low", "close", "volume"]
        df[float_cols] = df[float_cols].astype("float64")
        self._accumulate("bars", df.set_index("timestamp"))

    def add_ca(self, ca: CorporateAction) -> None:
        new_row = pd.DataFrame([asdict(ca)])
        self.ca_df = (
            new_row
            if self.ca_df.empty
            else pd.concat([self.ca_df, new_row]).drop_duplicates()
        ).copy()
        self._save_metadata()

    def sync_data(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> None:
        if not self.provider:
            return
        raw_bars = self.provider.fetch_bars(tickers, start, end)
        now = datetime.now()
        self.add_bars(
            [
                Bar(
                    self.get_internal_id(row["ticker"], row["timestamp"]),
                    row["timestamp"],
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                    row.get("timeframe", "1D"),
                    now,
                )
                for _, row in raw_bars.iterrows()
            ]
        )
        ca_raw = self.provider.fetch_corporate_actions(tickers, start, end)
        for _, row in ca_raw.iterrows():
            self.add_ca(
                CorporateAction(
                    self.get_internal_id(row["ticker"], row["ex_date"]),
                    row["ex_date"],
                    row["type"],
                    row["ratio"],
                )
            )

    def get_bars(self, iids: List[int], config: QueryConfig) -> pd.DataFrame:
        as_of_dt = config.as_of or datetime.now()
        query = QueryBuilder()
        query = query[
            query.internal_id.isin(iids)
            & (query.timeframe == config.timeframe)
            & (query.timestamp_knowledge <= as_of_dt)
            & (query.timestamp >= config.start)
            & (query.timestamp <= config.end)
        ]
        df = (
            self.lib.read("bars", query_builder=query)
            .data.copy()
            .reset_index()
        )
        if df.empty:
            return df
        df = (
            df.sort_values("timestamp_knowledge")
            .groupby(["internal_id", "timestamp"])
            .last()
            .reset_index()
            .copy()
        )
        if config.adjust and not self.ca_df.empty:
            for iid in iids:
                ca = self.ca_df[
                    (self.ca_df["internal_id"] == iid)
                    & (self.ca_df["ex_date"] <= config.end)
                ]
                for _, row in ca.sort_values(
                    "ex_date", ascending=False
                ).iterrows():
                    mask = (df["internal_id"] == iid) & (
                        df["timestamp"] < row["ex_date"]
                    )
                    cols = ["open", "high", "low", "close"]
                    if row["type"] == "SPLIT":
                        df.loc[mask, cols] /= row["ratio"]
                    elif row["type"] == "DIVIDEND":
                        df.loc[mask, cols] *= row["ratio"]
        rename_map = {
            c: f"{c}_{config.timeframe}"
            for c in ["open", "high", "low", "close", "volume"]
        }
        return df.rename(columns=rename_map).copy()
