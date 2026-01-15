import os
import pandas as pd
from datetime import datetime
from typing import List, Optional, Callable, Any
from ..common.types import Bar
from ..common.base import MarketDataProvider
from ..common.config import config


class MarketDataEngine(MarketDataProvider):
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path or config.get(
            "data.market_path", "data/market"
        )
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
                # Append new data. We don't drop duplicates yet because we want to keep historical versions for bitemporal.
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

                # Add internal_id back as it might have been lost in groupby if not careful
                if "internal_id" not in filtered_df.columns:
                    filtered_df["internal_id"] = internal_id

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
