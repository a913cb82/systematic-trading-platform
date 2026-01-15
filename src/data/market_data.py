import logging
import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, cast

import pandas as pd

from ..common.base import MarketDataProvider, RiskModel
from ..common.config import config
from ..common.types import Bar
from .corporate_actions import CorporateActionMaster

logger = logging.getLogger(__name__)


class MarketDataEngine(MarketDataProvider):
    def __init__(
        self,
        base_path: Optional[str] = None,
        ca_master: Optional[CorporateActionMaster] = None,
    ) -> None:
        self.base_path = base_path or config.get(
            "data.market_path", "data/market"
        )
        self.ca_master = ca_master
        self.bars_path = os.path.join(self.base_path, "bars")
        os.makedirs(self.bars_path, exist_ok=True)
        self._subscribers: List[tuple[List[int], Callable[[Bar], None]]] = []
        self._last_bars: Dict[int, Bar] = {}

    def validate_bars(self, data: List[Bar]) -> List[Bar]:
        """
        Validates OHLCV logic and filters out 'bad prints'.
        """
        valid_bars = []
        for bar in data:
            try:
                # 1. Check OHLC Logic
                if not (bar["high"] >= bar["low"]):
                    logger.warning(f"Invalid High/Low: {bar}")
                    continue
                if not (
                    bar["high"] >= bar.get("open", 0)
                    and bar["high"] >= bar["close"]
                ):
                    logger.warning(f"High not highest: {bar}")
                    continue
                if not (
                    bar["low"] <= bar.get("open", float("inf"))
                    and bar["low"] <= bar["close"]
                ):
                    logger.warning(f"Low not lowest: {bar}")
                    continue

                # 2. Volume sanity
                if bar["volume"] < 0:
                    logger.warning(f"Negative volume: {bar}")
                    continue

                # 3. Price sanity (must be positive)
                if bar["close"] <= 0:
                    logger.warning(f"Non-positive price: {bar}")
                    continue

                valid_bars.append(bar)
            except KeyError as e:
                logger.error(f"Missing required field {e} in bar: {bar}")
                continue

        return valid_bars

    def write_bars(self, data: List[Bar], fill_gaps: bool = False) -> None:
        if not data:
            return

        # 1. Validate before anything else
        data = self.validate_bars(data)
        if not data:
            return

        # 2. Ensure every bar has knowledge time BEFORE gap filling
        # (Gap filling needs a valid last_bar with knowledge time)
        now = datetime.now().replace(microsecond=0)
        for bar in data:
            if "timestamp_knowledge" not in bar:
                bar["timestamp_knowledge"] = now

        # 3. Gap filling
        if fill_gaps:
            data = self._fill_gaps(data)

        df = pd.DataFrame(data)
        # Ensure timestamps are normalized datetime64[ns]
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df["timestamp_knowledge"] = pd.to_datetime(
            df["timestamp_knowledge"]
        ).dt.tz_localize(None)

        for internal_id, group in df.groupby("internal_id"):
            # Update cache for gap filling
            last_row = group.sort_values("timestamp").iloc[-1]
            # Convert to dict but preserve types
            self._last_bars[int(internal_id)] = cast(Bar, last_row.to_dict())

            id_path = os.path.join(
                self.bars_path, f"internal_id={internal_id}"
            )
            os.makedirs(id_path, exist_ok=True)

            file_path = os.path.join(id_path, "data.parquet")

            if os.path.exists(file_path):
                existing_df = pd.read_parquet(file_path)
                existing_df["timestamp"] = pd.to_datetime(
                    existing_df["timestamp"]
                ).dt.tz_localize(None)
                existing_df["timestamp_knowledge"] = pd.to_datetime(
                    existing_df["timestamp_knowledge"]
                ).dt.tz_localize(None)
                combined_df = pd.concat([existing_df, group])
                # Bitemporal Sort
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
                        callback(cast(Bar, bar))

    def _fill_gaps(self, data: List[Bar]) -> List[Bar]:
        """
        Simple forward-fill logic for missing 1-minute intervals.
        """
        if not data:
            return data

        filled_data: List[Bar] = []
        # Sort by ID then timestamp
        sorted_data = sorted(
            data, key=lambda x: (x["internal_id"], x["timestamp"])
        )

        for bar in sorted_data:
            iid = bar["internal_id"]
            if iid in self._last_bars:
                last_bar = self._last_bars[iid]
                last_ts = pd.to_datetime(last_bar["timestamp"]).tz_localize(
                    None
                )
                curr_ts = pd.to_datetime(bar["timestamp"]).tz_localize(None)

                delta = curr_ts - last_ts
                if delta > timedelta(minutes=1):
                    num_missing = int(delta.total_seconds() / 60) - 1
                    for j in range(1, num_missing + 1):
                        gap_ts = last_ts + timedelta(minutes=j)
                        # Create synthetic bar from last known state
                        gap_bar = last_bar.copy()
                        gap_bar["timestamp"] = gap_ts
                        gap_bar["timestamp_knowledge"] = (
                            datetime.now().replace(microsecond=0)
                        )
                        gap_bar["volume"] = 0.0
                        filled_data.append(gap_bar)

            filled_data.append(bar)
            # Update cache as we go
            self._last_bars[iid] = bar

        return filled_data

    def get_bars(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        adjustment: str = "RAW",
        as_of: Optional[datetime] = None,
    ) -> List[Bar]:
        all_bars = []
        # Normalize query times
        q_start = pd.to_datetime(start).tz_localize(None)
        q_end = pd.to_datetime(end).tz_localize(None)
        q_as_of = pd.to_datetime(as_of or datetime.now()).tz_localize(None)

        for internal_id in internal_ids:
            id_path = os.path.join(
                self.bars_path, f"internal_id={internal_id}"
            )
            file_path = os.path.join(id_path, "data.parquet")

            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"]
                ).dt.tz_localize(None)
                df["timestamp_knowledge"] = pd.to_datetime(
                    df["timestamp_knowledge"]
                ).dt.tz_localize(None)

                # Filter by knowledge time
                df = df[df["timestamp_knowledge"] <= q_as_of]

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
                mask = (df["timestamp"] >= q_start) & (
                    df["timestamp"] <= q_end
                )
                filtered_df = df.loc[mask]

                if "internal_id" not in filtered_df.columns:
                    filtered_df["internal_id"] = internal_id

                # Dynamic Adjustment
                if (
                    adjustment == "RATIO"
                    and self.ca_master
                    and not filtered_df.empty
                ):
                    actions = self.ca_master.get_actions(
                        [internal_id],
                        start=datetime(1900, 1, 1),
                        end=end,
                        as_of=as_of,
                    )
                    if actions:
                        actions.sort(key=lambda x: x["ex_date"], reverse=True)
                        for action in actions:
                            if action["type"] == "SPLIT":
                                ratio = action["ratio"]
                                mask_pre = filtered_df[
                                    "timestamp"
                                ] < pd.to_datetime(
                                    action["ex_date"]
                                ).tz_localize(None)
                                filtered_df.loc[
                                    mask_pre, ["open", "high", "low", "close"]
                                ] /= ratio
                            elif action["type"] == "DIVIDEND":
                                ratio = action["ratio"]
                                mask_pre = filtered_df[
                                    "timestamp"
                                ] < pd.to_datetime(
                                    action["ex_date"]
                                ).tz_localize(None)
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
        date_range: tuple[datetime, datetime],
        type: str = "RAW",
        as_of: Optional[datetime] = None,
        risk_model: Optional[RiskModel] = None,
    ) -> pd.DataFrame:
        """
        Calculates returns for given assets.
        If type="RESIDUAL", it requires a risk model to regress out factors.
        """
        start, end = date_range
        bars = self.get_bars(
            internal_ids, start, end, adjustment="RATIO", as_of=as_of
        )
        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame(bars)
        df = df.pivot(index="timestamp", columns="internal_id", values="close")
        returns = df.pct_change(fill_method=None).dropna(how="all")

        if type == "RAW":
            return returns

        if type == "RESIDUAL" and risk_model:
            # Reconstruct idiosyncratic returns:
            # resid = raw - (beta * factor_return)
            # In this PoC, we'll implement a simple demeaned residual
            # (treating the cross-sectional mean as the single market factor)
            # as actual factor returns would require a separate feed.
            market_return = returns.mean(axis=1)
            residuals = returns.subtract(market_return, axis=0)
            return residuals

        return returns
