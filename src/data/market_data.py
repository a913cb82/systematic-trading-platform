import os
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

from ..common.base import MarketDataProvider
from ..common.config import config
from ..common.types import Bar
from .corporate_actions import CorporateActionMaster


class MarketDataEngine(MarketDataProvider):
    def __init__(
        self,
        base_path: Optional[str] = None,
        ca_master: Optional[CorporateActionMaster] = None,
    ):
        self.base_path = base_path or config.get(
            "data.market_path", "data/market"
        )
        self.ca_master = ca_master
        self.bars_path = os.path.join(self.base_path, "bars")
        os.makedirs(self.bars_path, exist_ok=True)
        self._subscribers: List[tuple[List[int], Callable[[Bar], None]]] = []

    def write_bars(self, data: List[Bar]) -> None:
        if not data:
            return

        df = pd.DataFrame(data)
        # Ensure timestamps are datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "timestamp_knowledge" not in df.columns:
            df["timestamp_knowledge"] = datetime.now()
        df["timestamp_knowledge"] = pd.to_datetime(df["timestamp_knowledge"])

        for internal_id, group in df.groupby("internal_id"):
            id_path = os.path.join(
                self.bars_path, f"internal_id={internal_id}"
            )
            os.makedirs(id_path, exist_ok=True)

            file_path = os.path.join(id_path, "data.parquet")

            if os.path.exists(file_path):
                existing_df = pd.read_parquet(file_path)
                existing_df["timestamp"] = pd.to_datetime(
                    existing_df["timestamp"]
                )
                existing_df["timestamp_knowledge"] = pd.to_datetime(
                    existing_df["timestamp_knowledge"]
                )
                # Append new data. We don't drop duplicates yet because we
                # want to keep historical versions for bitemporal.
                combined_df = pd.concat([existing_df, group])
                combined_df = combined_df.sort_values(
                    ["timestamp", "timestamp_knowledge"]
                )
                combined_df.to_parquet(file_path, index=False)
            else:
                group.sort_values(
                    ["timestamp", "timestamp_knowledge"]
                ).to_parquet(file_path, index=False)

            # Notify subscribers
            for iid_list, callback in self._subscribers:
                if internal_id in iid_list:
                    for bar in group.to_dict("records"):
                        callback(bar)

    def get_bars(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        adjustment: str = "RAW",
        as_of: Optional[datetime] = None,
    ) -> List[Bar]:
        all_bars = []
        if as_of is None:
            as_of = datetime.now()

        for internal_id in internal_ids:
            id_path = os.path.join(
                self.bars_path, f"internal_id={internal_id}"
            )
            file_path = os.path.join(id_path, "data.parquet")

            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["timestamp_knowledge"] = pd.to_datetime(
                    df["timestamp_knowledge"]
                )

                # Filter by knowledge time
                df = df[df["timestamp_knowledge"] <= as_of]

                if df.empty:
                    continue

                # Take the latest knowledge for each event timestamp
                df = (
                    df.sort_values("timestamp_knowledge")
                    .groupby("timestamp")
                    .last()
                    .reset_index()
                )

                # Filter by event time
                mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
                filtered_df = df.loc[mask]

                # Add internal_id back as it might have been lost in
                # groupby if not careful
                if "internal_id" not in filtered_df.columns:
                    filtered_df["internal_id"] = internal_id

                # Dynamic Adjustment
                if (
                    adjustment == "RATIO"
                    and self.ca_master
                    and not filtered_df.empty
                ):
                    # Fetch actions up to the end of the query period
                    actions = self.ca_master.get_actions(
                        [internal_id],
                        start=datetime(1900, 1, 1),
                        end=end,
                        as_of=as_of,
                    )
                    if actions:
                        # Sort actions by ex_date descending
                        actions.sort(key=lambda x: x["ex_date"], reverse=True)
                        # We apply adjustments backwards from the current date

                        # Convert to DataFrame for easier manipulation
                        # but simple loop is fine for performance here.
                        for action in actions:
                            # If ex_date is in the future of a bar, it affects
                            # that bar's price (backward adjustment)
                            # e.g. 2-for-1 split on day T.
                            # All bars < T must have prices divided by 2.
                            if action["type"] == "SPLIT":
                                ratio = action["ratio"]
                                # Find bars before this split
                                mask_pre = (
                                    filtered_df["timestamp"]
                                    < action["ex_date"]
                                )
                                filtered_df.loc[
                                    mask_pre, ["open", "high", "low", "close"]
                                ] /= ratio
                            elif action["type"] == "DIVIDEND":
                                # For dividends, ratio adjustment:
                                # (Price - Div) / Price
                                # This is more complex because it depends on
                                # the price at ex-date.
                                # Simplified for now: multiplier
                                ratio = action["ratio"]
                                mask_pre = (
                                    filtered_df["timestamp"]
                                    < action["ex_date"]
                                )
                                filtered_df.loc[
                                    mask_pre, ["open", "high", "low", "close"]
                                ] *= ratio

                all_bars.extend(filtered_df.to_dict("records"))

        return all_bars

    def subscribe_bars(
        self, internal_ids: List[int], on_bar: Callable[[Bar], None]
    ) -> None:
        self._subscribers.append((internal_ids, on_bar))

    def get_universe(self, date: datetime) -> List[int]:
        internal_ids = []
        if not os.path.exists(self.bars_path):
            return []
        for entry in os.scandir(self.bars_path):
            if entry.is_dir() and entry.name.startswith("internal_id="):
                try:
                    internal_id = int(entry.name.split("=")[1])
                    internal_ids.append(internal_id)
                except (IndexError, ValueError):
                    continue
        return internal_ids

    def get_returns(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        type: str = "RAW",
        as_of: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Calculates returns for given assets.
        If type="RESIDUAL", it requires a risk model to regress out factors.
        """
        # 1. Fetch adjusted bars (Backtesting usually wants adjusted)
        bars = self.get_bars(
            internal_ids, start, end, adjustment="RATIO", as_of=as_of
        )
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df = df.pivot(index="timestamp", columns="internal_id", values="close")
        returns = df.pct_change().dropna(how="all")

        if type == "RAW":
            return returns

        if type == "RESIDUAL":
            # This would ideally take a RiskModel instance.
            # For now, we'll return raw and leave residual logic
            # for the alpha model or a specific 'ResidualReturnsEngine'
            return returns

        return returns
