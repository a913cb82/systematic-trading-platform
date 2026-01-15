import os
import shutil
import unittest
from datetime import datetime, timedelta

import numpy as np

from src.alpha.features import FeatureStore
from src.common.types import Bar, Event
from src.data.event_store import EventStore
from src.data.market_data import MarketDataEngine


class TestFeatureStore(unittest.TestCase):
    def setUp(self):
        self.market_dir = "test_market_features"
        self.event_dir = "test_event_features"
        if os.path.exists(self.market_dir):
            shutil.rmtree(self.market_dir)
        if os.path.exists(self.event_dir):
            shutil.rmtree(self.event_dir)

        self.market_data = MarketDataEngine(self.market_dir)
        self.event_store = EventStore(self.event_dir)
        self.feature_store = FeatureStore(self.market_data, self.event_store)

    def tearDown(self):
        if os.path.exists(self.market_dir):
            shutil.rmtree(self.market_dir)
        if os.path.exists(self.event_dir):
            shutil.rmtree(self.event_dir)

    def test_cycle_feature(self):
        # Write some bars
        bars = []
        for i in range(10):
            bars.append(
                Bar(
                    internal_id=1,
                    timestamp=datetime(2023, 1, 1) + timedelta(days=i),
                    timestamp_knowledge=datetime(2023, 1, 1)
                    + timedelta(days=i),
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.5 + i,
                    volume=1000.0,
                )
            )
        self.market_data.write_bars(bars)

        features = self.feature_store.calculate_cycle_feature(
            [1], datetime(2023, 1, 1), datetime(2023, 1, 10), "returns_1d"
        )
        self.assertEqual(len(features), 10)
        # First return should be NaN

        self.assertTrue(np.isnan(features.iloc[0]["feature"]))
        self.assertGreater(features.iloc[1]["feature"], 0)

    def test_event_feature(self):
        events = [
            Event(
                internal_id=1,
                type="EARNINGS",
                value={"surprise": 0.1},
                timestamp_event=datetime(2023, 1, 15),
                timestamp_knowledge=datetime(2023, 1, 15),
            ),
        ]
        self.event_store.write_events(events)

        features = self.feature_store.calculate_event_feature(
            [1],
            datetime(2023, 1, 1),
            datetime(2023, 1, 31),
            "earnings_surprise",
        )
        self.assertEqual(len(features), 1)
        self.assertEqual(features.iloc[0]["feature"], 0.1)


if __name__ == "__main__":
    unittest.main()
