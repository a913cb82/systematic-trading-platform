from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .base import DataProvider


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
class CorporateAction:
    internal_id: int
    ex_date: datetime
    type: str  # 'SPLIT' or 'DIVIDEND'
    ratio: float


class DataPlatform:
    def __init__(self, provider: Optional[DataProvider] = None) -> None:
        self.provider = provider
        self.ism: Dict[str, int] = {}  # Ticker -> InternalID
        self.reverse_ism: Dict[int, str] = {}
        self.bars: List[Bar] = []
        self.ca_df = pd.DataFrame(
            columns=["internal_id", "ex_date", "type", "ratio"]
        )
        self._id_counter = 1000

    def get_internal_id(self, ticker: str) -> int:
        if ticker not in self.ism:
            self.ism[ticker] = self._id_counter
            self.reverse_ism[self._id_counter] = ticker
            self._id_counter += 1
        return self.ism[ticker]

    def add_bars(self, bars: List[Bar], fill_gaps: bool = False) -> None:
        valid_bars = []
        for b in bars:
            if b.high >= b.low and b.volume >= 0 and b.close > 0:
                valid_bars.append(b)

        if fill_gaps:
            valid_bars = self._fill_gaps(valid_bars)
        self.bars.extend(valid_bars)

    def _fill_gaps(self, bars: List[Bar]) -> List[Bar]:
        if not bars:
            return []
        bars = sorted(bars, key=lambda x: (x.internal_id, x.timestamp))
        filled = []
        last_bars: Dict[int, Bar] = {}

        for b in bars:
            if b.internal_id in last_bars:
                last = last_bars[b.internal_id]
                delta = b.timestamp - last.timestamp
                if delta > pd.Timedelta(minutes=1):
                    # Fill 1-min gaps
                    for j in range(1, int(delta.total_seconds() / 60)):
                        filled.append(
                            Bar(
                                b.internal_id,
                                last.timestamp + pd.Timedelta(minutes=j),
                                last.close,
                                last.close,
                                last.close,
                                last.close,
                                0.0,
                            )
                        )
            filled.append(b)
            last_bars[b.internal_id] = b
        return filled

    def add_ca(self, ca: CorporateAction) -> None:
        new_row = pd.DataFrame(
            [
                {
                    "internal_id": ca.internal_id,
                    "ex_date": ca.ex_date,
                    "type": ca.type,
                    "ratio": ca.ratio,
                }
            ]
        )
        self.ca_df = pd.concat([self.ca_df, new_row]).drop_duplicates()

    def sync_data(
        self, tickers: List[str], start: datetime, end: datetime
    ) -> None:
        if not self.provider:
            return

        # 1. Fetch from Plugin
        raw_bars = self.provider.fetch_bars(tickers, start, end)
        now = datetime.now()

        # 2. Map to Internal IDs and Store (Bitemporal)
        new_bars = []
        for _, row in raw_bars.iterrows():
            iid = self.get_internal_id(row["ticker"])
            new_bars.append(
                Bar(
                    internal_id=iid,
                    timestamp=row["timestamp"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    timestamp_knowledge=now,
                )
            )
        self.bars.extend(new_bars)

        # 3. Fetch Corporate Actions
        raw_ca = self.provider.fetch_corporate_actions(tickers, start, end)
        if not raw_ca.empty:
            raw_ca["internal_id"] = raw_ca["ticker"].apply(
                self.get_internal_id
            )
            self.ca_df = pd.concat(
                [
                    self.ca_df,
                    raw_ca[["internal_id", "ex_date", "type", "ratio"]],
                ]
            ).drop_duplicates()

    def get_bars(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
        adjust: bool = True,
    ) -> pd.DataFrame:
        as_of = as_of or datetime.now()
        valid_bars = [
            b
            for b in self.bars
            if b.internal_id in internal_ids
            and start <= b.timestamp <= end
            and b.timestamp_knowledge <= as_of
        ]

        if not valid_bars:
            return pd.DataFrame()
        df = pd.DataFrame(valid_bars)
        df = (
            df.sort_values("timestamp_knowledge")
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
                for _, action in iid_ca.sort_values(
                    "ex_date", ascending=False
                ).iterrows():
                    mask = (df["internal_id"] == iid) & (
                        df["timestamp"] < action["ex_date"]
                    )
                    if action["type"] == "SPLIT":
                        df.loc[mask, ["open", "high", "low", "close"]] /= (
                            action["ratio"]
                        )
                    elif action["type"] == "DIVIDEND":
                        df.loc[mask, ["open", "high", "low", "close"]] *= (
                            action["ratio"]
                        )
        return df

    def get_returns(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        benchmark_id: Optional[int] = None,
    ) -> pd.DataFrame:
        df = self.get_bars(
            internal_ids + ([benchmark_id] if benchmark_id else []), start, end
        )
        if df.empty:
            return pd.DataFrame()
        px = df.pivot(index="timestamp", columns="internal_id", values="close")
        returns = px.pct_change(fill_method=None).dropna(how="all")

        if benchmark_id and benchmark_id in returns.columns:
            # Drop rows where either benchmark or asset is missing
            returns = returns.dropna()
            bench = returns[benchmark_id]
            returns = returns.drop(columns=[benchmark_id]).sub(bench, axis=0)

        return returns
