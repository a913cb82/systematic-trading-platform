import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, cast

import pandas as pd
from arcticdb import Arctic, QueryBuilder

from src.gateways.base import DataProvider


@dataclass
class Security:
    internal_id: int
    ticker: str
    start: datetime
    end: datetime
    extra: Any


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


class DataPlatform:
    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        db_path: str = "./.arctic_db",
        clear: bool = False,
    ) -> None:
        self.provider, self.arctic = (
            provider,
            _ARCTIC_CACHE.setdefault(db_path, Arctic(f"lmdb://{db_path}")),
        )
        if clear and "platform" in self.arctic.list_libraries():
            self.arctic.delete_library("platform")
        self.lib = self.arctic.get_library("platform", create_if_missing=True)
        self._bootstrap()

    def _bootstrap(self) -> None:
        tm = {int: "int64", float: "float64", datetime: "datetime64[ns]"}
        for s, c in {
            "sec_df": Security,
            "ca_df": CorporateAction,
            "bars": Bar,
            "events": Event,
        }.items():
            if not self.lib.has_symbol(s):
                df = pd.DataFrame(
                    {
                        f.name: pd.Series(
                            dtype=tm.get(cast(Type[Any], f.type), "O")
                        )
                        for f in fields(c)
                    }
                )
                self.lib.write(
                    s,
                    df.set_index("timestamp")
                    if s in ["bars", "events"]
                    else df,
                )
        self.sec_df = (
            self.lib.read("sec_df")
            .data.copy()
            .assign(
                extra=lambda x: x.extra.apply(
                    lambda e: json.loads(e) if isinstance(e, str) else e
                )
            )
        )
        self.ca_df, self._id_counter = (
            self.lib.read("ca_df").data.copy(),
            int(
                self.sec_df.internal_id.max() + 1
                if not self.sec_df.empty
                else 1000
            ),
        )

    def _persist_metadata(self) -> None:
        self.lib.write(
            "sec_df",
            self.sec_df.assign(extra=lambda x: x.extra.apply(json.dumps)),
        )
        self.lib.write("ca_df", self.ca_df)

    def register_security(
        self,
        ticker: str,
        internal_id: Optional[int] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **kwargs: Any,
    ) -> int:
        s, e = start or datetime(1900, 1, 1), end or datetime(2200, 1, 1)
        res = self.sec_df[
            (self.sec_df["ticker"] == ticker)
            & (self.sec_df["start"] <= e)
            & (self.sec_df["end"] >= s)
        ]
        if internal_id is None and not res.empty:
            return int(res.iloc[0].internal_id)
        iid = internal_id or self._id_counter
        if internal_id is None:
            self._id_counter += 1
        new_sec = pd.DataFrame([asdict(Security(iid, ticker, s, e, kwargs))])
        if self.sec_df.empty:
            self.sec_df = new_sec
        else:
            self.sec_df = pd.concat([self.sec_df, new_sec], ignore_index=True)
        self._persist_metadata()
        return iid

    def get_internal_id(
        self, ticker: str, date: Optional[datetime] = None
    ) -> int:
        d = date or datetime.now()
        res = self.sec_df[
            (self.sec_df["ticker"] == ticker)
            & (self.sec_df["start"] <= d)
            & (self.sec_df["end"] >= d)
        ]
        return (
            int(res.iloc[0].internal_id)
            if not res.empty
            else self.register_security(ticker, start=d)
        )

    def get_securities(
        self, tickers: Optional[List[str]] = None
    ) -> List[Security]:
        df = (
            self.sec_df
            if tickers is None
            else self.sec_df[self.sec_df.ticker.isin(tickers)]
        )
        return [Security(**r) for r in df.to_dict("records")]

    @property
    def reverse_ism(self) -> Dict[int, str]:
        return (
            cast(
                Dict[int, str],
                self.sec_df.sort_values("start")
                .groupby("internal_id")
                .ticker.last()
                .to_dict(),
            )
            if not self.sec_df.empty
            else {}
        )

    def get_universe(self, date: datetime) -> List[int]:
        return cast(
            List[int],
            self.sec_df[
                (self.sec_df.start <= date) & (self.sec_df.end >= date)
            ]
            .internal_id.unique()
            .tolist(),
        )

    def _update_timeseries(self, sym: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        existing = self.lib.read(sym).data
        updated = (
            df if existing.empty else pd.concat([existing, df])
        ).sort_index()
        self.lib.write(sym, updated)

    def add_events(self, events: List[Event]) -> None:
        if events:
            self._update_timeseries(
                "events",
                pd.DataFrame(
                    [
                        {**asdict(e), "value": json.dumps(e.value)}
                        for e in events
                    ]
                ).set_index("timestamp"),
            )

    def get_events(
        self,
        iids: List[int],
        types: Optional[List[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        q = QueryBuilder()
        q = q[
            q.internal_id.isin(iids)
            & (q.timestamp_knowledge <= (as_of or datetime.now()))
        ]
        if start:
            q = q[q.timestamp >= start]
        if end:
            q = q[q.timestamp <= end]
        if types:
            q = q[q.event_type.isin(types)]
        res = self.lib.read("events", query_builder=q).data.copy()
        if res.empty:
            return []
        return [
            Event(
                **{
                    **r,
                    "value": json.loads(r["value"])
                    if isinstance(r["value"], str)
                    else r["value"],
                }
            )
            for r in res.reset_index().to_dict("records")
        ]

    def add_bars(self, bars: List[Bar]) -> None:
        if bars:
            self._update_timeseries(
                "bars",
                pd.DataFrame([asdict(b) for b in bars])
                .astype(
                    {
                        c: "float64"
                        for c in ["open", "high", "low", "close", "volume"]
                    }
                )
                .set_index("timestamp"),
            )

    def add_ca(self, ca: CorporateAction) -> None:
        new_row = pd.DataFrame([asdict(ca)])
        if self.ca_df.empty:
            self.ca_df = new_row
        else:
            self.ca_df = pd.concat([self.ca_df, new_row]).drop_duplicates()
        self._persist_metadata()

    def sync_data(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> None:
        if not self.provider:
            return
        now, bars = (
            datetime.now(),
            self.provider.fetch_bars(tickers, start, end),
        )
        if not bars.empty:
            self._update_timeseries(
                "bars",
                bars.assign(
                    internal_id=bars.apply(
                        lambda r: self.get_internal_id(r.ticker, r.timestamp),
                        1,
                    ),
                    timestamp_knowledge=now,
                    timeframe=bars.get("timeframe", "1D"),
                )
                .drop(columns="ticker")
                .set_index("timestamp"),
            )
        cas = self.provider.fetch_corporate_actions(tickers, start, end)
        if not cas.empty:
            new_cas = cas.assign(
                internal_id=cas.apply(
                    lambda r: self.get_internal_id(r.ticker, r.ex_date),
                    1,
                )
            ).drop(columns="ticker")
            if self.ca_df.empty:
                self.ca_df = new_cas
            else:
                self.ca_df = pd.concat([self.ca_df, new_cas]).drop_duplicates()
            self._persist_metadata()
        evs = self.provider.fetch_events(tickers, start, end)
        if not evs.empty:
            self.add_events(
                [
                    Event(
                        self.get_internal_id(r.ticker, r.timestamp),
                        r.timestamp,
                        r.event_type,
                        r.value,
                        now,
                    )
                    for _, r in evs.iterrows()
                ]
            )

    def get_bars(self, iids: List[int], cfg: QueryConfig) -> pd.DataFrame:
        q = QueryBuilder()
        q = q[
            q.internal_id.isin(iids)
            & (q.timeframe == cfg.timeframe)
            & (q.timestamp_knowledge <= (cfg.as_of or datetime.now()))
            & (q.timestamp >= cfg.start)
            & (q.timestamp <= cfg.end)
        ]
        df = self.lib.read("bars", query_builder=q).data.copy().reset_index()
        if df.empty:
            return df
        df = (
            df.sort_values("timestamp_knowledge")
            .groupby(["internal_id", "timestamp"])
            .last()
            .reset_index()
        )
        if cfg.adjust and not self.ca_df.empty:
            for iid in iids:
                for _, r in (
                    self.ca_df[
                        (self.ca_df.internal_id == iid)
                        & (self.ca_df.ex_date <= cfg.end)
                    ]
                    .sort_values("ex_date", ascending=False)
                    .iterrows()
                ):
                    df.loc[
                        (df.internal_id == iid) & (df.timestamp < r.ex_date),
                        ["open", "high", "low", "close"],
                    ] *= 1 / r.ratio if r.type == "SPLIT" else r.ratio
        return df.rename(
            columns={
                c: f"{c}_{cfg.timeframe}"
                for c in ["open", "high", "low", "close", "volume"]
            }
        )
