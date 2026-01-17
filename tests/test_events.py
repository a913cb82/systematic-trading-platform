import unittest
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from src.alpha_library.models import EarningsModel
from src.core.alpha_engine import AlphaEngine, AlphaModel, ModelRunConfig
from src.core.data_platform import Bar, DataPlatform, Event


class TestEventSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "./.arctic_test_db"
        self.data = DataPlatform(db_path=self.db_path, clear=True)
        self.iid = self.data.get_internal_id("AAPL")
        self.ts = datetime(2025, 1, 1, 12, 0)

    def test_dataplatform_event_filtering(self) -> None:
        """Test DataPlatform.get_events with start/end and as_of filters."""
        # Event 1: T-2h, known immediately
        e1 = Event(
            self.iid,
            self.ts - timedelta(hours=2),
            "TEST",
            {"v": 1},
            self.ts - timedelta(hours=2),
        )
        # Event 2: T-1h, known at T+1h (future knowledge)
        e2 = Event(
            self.iid,
            self.ts - timedelta(hours=1),
            "TEST",
            {"v": 2},
            self.ts + timedelta(hours=1),
        )

        self.data.add_events([e1, e2])

        # Query as of T: should only see e1
        evs_now = self.data.get_events([self.iid], as_of=self.ts)
        self.assertEqual(len(evs_now), 1)
        self.assertEqual(evs_now[0].value["v"], 1)

        # Query as of T+2h: should see both
        evs_future = self.data.get_events(
            [self.iid], as_of=self.ts + timedelta(hours=2)
        )
        self.assertEqual(len(evs_future), 2)

        # Query with time range (exclude e1)
        evs_range = self.data.get_events(
            [self.iid],
            start=self.ts - timedelta(minutes=90),
            as_of=self.ts + timedelta(hours=2),
        )
        self.assertEqual(len(evs_range), 1)
        self.assertEqual(evs_range[0].value["v"], 2)

    def test_alphamodel_static_get_events(self) -> None:
        """Test the static get_events API and context plumbing."""
        event_ts = self.ts - timedelta(minutes=30)
        self.data.add_events(
            [Event(self.iid, event_ts, "SIGNAL", {"alpha": 0.5}, event_ts)]
        )

        # Add a bar so the model can run
        self.data.add_bars([Bar(self.iid, self.ts, 100, 101, 99, 100, 1000)])

        class MockEventModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame
            ) -> Dict[int, float]:
                # Access static API
                evs = self.get_events(list(latest.index), types=["SIGNAL"])
                return {ev.internal_id: ev.value["alpha"] for ev in evs}

        model = MockEventModel()
        forecasts = AlphaEngine.run_model(
            self.data, model, [self.iid], ModelRunConfig(timestamp=self.ts)
        )

        self.assertEqual(forecasts[self.iid], 0.5)

    def test_earnings_model_decay(self) -> None:
        """Test that EarningsModel correctly applies linear decay."""
        # Event exactly 12 hours ago
        event_ts = self.ts - timedelta(hours=12)
        self.data.add_events(
            [
                Event(
                    self.iid,
                    event_ts,
                    "EARNINGS_RELEASE",
                    {"surprise_pct": 0.1},
                    event_ts,
                )
            ]
        )
        self.data.add_bars([Bar(self.iid, self.ts, 100, 101, 99, 100, 1000)])

        model = EarningsModel()
        forecasts = AlphaEngine.run_model(
            self.data, model, [self.iid], ModelRunConfig(timestamp=self.ts)
        )

        # Base signal is 0.5. At 12h (halfway to 24h), it should be 0.25
        self.assertAlmostEqual(forecasts[self.iid], 0.25)

        # Test at 24h (should be 0)
        ts_24h = self.ts + timedelta(hours=12)
        self.data.add_bars([Bar(self.iid, ts_24h, 100, 101, 99, 100, 1000)])
        forecasts_24h = AlphaEngine.run_model(
            self.data, model, [self.iid], ModelRunConfig(timestamp=ts_24h)
        )
        self.assertEqual(forecasts_24h.get(self.iid, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
