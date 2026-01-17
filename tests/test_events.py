import unittest
from datetime import datetime, timedelta

from src.alpha_library.models import EarningsModel
from src.core.alpha_engine import AlphaEngine, ModelRunConfig
from src.core.data_platform import Bar, DataPlatform, Event
from src.core.types import Timeframe


class TestEvents(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.ts = datetime(2025, 1, 1, 12, 0)
        self.iid = self.data.register_security("AAPL")

        # Basic bar for model to run
        bar = Bar(
            self.iid, self.ts, 100, 101, 99, 100, 1000, timeframe=Timeframe.DAY
        )
        self.data.add_bars([bar])

    def test_earnings_event_alpha(self) -> None:
        """Tests that EarningsModel generates a signal after an event."""
        # 1. No event -> zero signal
        model = EarningsModel()
        config = ModelRunConfig(timestamp=self.ts, timeframe=Timeframe.DAY)
        signals = AlphaEngine.run_model(self.data, model, [self.iid], config)
        self.assertEqual(signals.get(self.iid, 0.0), 0.0)

        # 2. Add surprise event 2h ago.
        # Set knowledge time <= self.ts so PIT retrieval sees it.
        event_ts = self.ts - timedelta(hours=2)
        event = Event(
            self.iid,
            event_ts,
            "EARNINGS_RELEASE",
            {"surprise_pct": 0.05},
            timestamp_knowledge=event_ts,
        )
        self.data.add_events([event])

        # 3. Model should now have positive signal
        signals = AlphaEngine.run_model(self.data, model, [self.iid], config)
        self.assertTrue(signals[self.iid] > 0)

    def test_event_decay(self) -> None:
        """Tests that signal strength decreases over time."""
        model = EarningsModel()
        event_ts = self.ts - timedelta(hours=1)
        event = Event(
            self.iid,
            event_ts,
            "EARNINGS_RELEASE",
            {"surprise_pct": 0.05},
            timestamp_knowledge=event_ts,
        )
        self.data.add_events([event])

        # Signal at T
        config1 = ModelRunConfig(timestamp=self.ts, timeframe=Timeframe.DAY)
        sig1 = AlphaEngine.run_model(self.data, model, [self.iid], config1)[
            self.iid
        ]

        # Signal at T + 10h
        future_ts = self.ts + timedelta(hours=10)
        config2 = ModelRunConfig(timestamp=future_ts, timeframe=Timeframe.DAY)
        # Add another bar so model can run at future time
        self.data.add_bars(
            [
                Bar(
                    self.iid,
                    future_ts,
                    100,
                    100,
                    100,
                    100,
                    1000,
                    timeframe=Timeframe.DAY,
                )
            ]
        )

        sig2 = AlphaEngine.run_model(self.data, model, [self.iid], config2)[
            self.iid
        ]

        self.assertTrue(sig2 < sig1)


if __name__ == "__main__":
    import src.alpha_library.features  # noqa: F401

    unittest.main()
