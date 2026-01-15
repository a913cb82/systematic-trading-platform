import os
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd

from ..common.types import Event


class EventStore:
    def __init__(self, base_path: str = "data/events"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        self._subscribers: List[tuple[List[str], Callable[[Event], None]]] = []

    def write_events(self, events: List[Event]) -> None:
        if not events:
            return

        df = pd.DataFrame(events)
        df["timestamp_event"] = pd.to_datetime(df["timestamp_event"])
        df["timestamp_knowledge"] = pd.to_datetime(df["timestamp_knowledge"])

        for (event_type, internal_id), group in df.groupby(
            ["type", "internal_id"]
        ):
            type_path = os.path.join(
                self.base_path,
                f"type={event_type}",
                f"internal_id={internal_id}",
            )
            os.makedirs(type_path, exist_ok=True)

            file_path = os.path.join(type_path, "data.parquet")

            if os.path.exists(file_path):
                existing_df = pd.read_parquet(file_path)
                existing_df["timestamp_event"] = pd.to_datetime(
                    existing_df["timestamp_event"]
                )
                existing_df["timestamp_knowledge"] = pd.to_datetime(
                    existing_df["timestamp_knowledge"]
                )
                combined_df = pd.concat([existing_df, group])
                combined_df = combined_df.sort_values(
                    ["timestamp_event", "timestamp_knowledge"]
                )
                combined_df.to_parquet(file_path, index=False)
            else:
                group.sort_values(
                    ["timestamp_event", "timestamp_knowledge"]
                ).to_parquet(file_path, index=False)

            # Notify subscribers
            for subscribed_types, callback in self._subscribers:
                if event_type in subscribed_types:
                    for event in group.to_dict("records"):
                        callback(event)

    def get_events(
        self,
        event_types: List[str],
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
    ) -> List[Event]:
        all_events = []
        if as_of is None:
            as_of = datetime.now()

        for event_type in event_types:
            for internal_id in internal_ids:
                type_path = os.path.join(
                    self.base_path,
                    f"type={event_type}",
                    f"internal_id={internal_id}",
                )
                file_path = os.path.join(type_path, "data.parquet")

                if os.path.exists(file_path):
                    df = pd.read_parquet(file_path)
                    df["timestamp_event"] = pd.to_datetime(
                        df["timestamp_event"]
                    )
                    df["timestamp_knowledge"] = pd.to_datetime(
                        df["timestamp_knowledge"]
                    )

                    # Bitemporal filter
                    df = df[df["timestamp_knowledge"] <= as_of]
                    if df.empty:
                        continue

                    # Latest knowledge for each event
                    # Since event value can be anything, we might need a
                    # more complex way to identify "same" event if multiple
                    # occur at same timestamp.
                    # For now, assume (timestamp_event, type, internal_id)
                    # is the unique key for an event instance.
                    df = (
                        df.sort_values("timestamp_knowledge")
                        .groupby("timestamp_event")
                        .last()
                        .reset_index()
                    )

                    mask = (df["timestamp_event"] >= start) & (
                        df["timestamp_event"] <= end
                    )
                    filtered_df = df.loc[mask]

                    # Ensure columns are present after groupby/reset_index
                    if "internal_id" not in filtered_df.columns:
                        filtered_df["internal_id"] = internal_id
                    if "type" not in filtered_df.columns:
                        filtered_df["type"] = event_type

                    all_events.extend(filtered_df.to_dict("records"))

        return all_events

    def subscribe_events(
        self, event_types: List[str], on_event: Callable[[Event], None]
    ) -> None:
        self._subscribers.append((event_types, on_event))
