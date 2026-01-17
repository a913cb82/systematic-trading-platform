import unittest
from datetime import datetime

import pandas as pd

import src.alpha_library.features  # noqa: F401
from src.core.alpha_engine import AlphaEngine
from src.core.data_platform import Bar, DataPlatform
from src.core.types import Timeframe


class TestFeatureDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)
        self.ts = datetime(2025, 1, 1, 12, 0)
        self.iid = self.data.register_security("AAPL")

        # Create history for features
        bars = [
            Bar(
                self.iid,
                self.ts - pd.Timedelta(minutes=30 * i),
                100,
                101,
                99,
                100,
                1000,
                timeframe=Timeframe.MIN_30,
            )
            for i in range(25)
        ]
        self.data.add_bars(bars)

    def test_recursive_hydration(self) -> None:
        """
        Verify that requesting 'residual_vol_20_30min' automatically
        hydrates 'returns_residual_30min' and 'returns_raw_30min'.
        """
        from src.core.types import QueryConfig

        df = self.data.get_bars(
            [self.iid],
            QueryConfig(
                start=self.ts - pd.Timedelta(days=1),
                end=self.ts,
                timeframe=Timeframe.MIN_30,
            ),
        )

        # Initially, features don't exist
        self.assertNotIn("residual_vol_20_30min", df.columns)

        # Hydrate
        AlphaEngine._hydrate_features(df, ["residual_vol_20_30min"])

        # Check dependencies exist
        self.assertIn("returns_raw_30min", df.columns)
        self.assertIn("returns_residual_30min", df.columns)
        self.assertIn("residual_vol_20_30min", df.columns)


if __name__ == "__main__":
    unittest.main()
