import unittest
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from src.core.alpha_engine import AlphaEngine, AlphaModel, feature
from src.core.data_platform import Bar, DataPlatform


class TestFeatureDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform()
        self.iid = self.data.get_internal_id("AAPL")

    def test_recursive_hydration(self) -> None:
        # Define features with dependencies and explicit naming
        @feature("base_val_30min")
        def base_val(df: pd.DataFrame) -> pd.Series:
            return df["close_30min"]

        @feature("plus_one_30min", dependencies=["base_val_30min"])
        def plus_one(df: pd.DataFrame) -> pd.Series:
            return df["base_val_30min"] + 1

        @feature("plus_two_30min", dependencies=["plus_one_30min"])
        def plus_two(df: pd.DataFrame) -> pd.Series:
            return df["plus_one_30min"] + 1

        class DepModel(AlphaModel):
            def __init__(self) -> None:
                super().__init__()
                self.feature_names = ["plus_two_30min"]

            def compute_signals(
                self, latest: pd.DataFrame
            ) -> Dict[int, float]:
                return {
                    int(idx): float(row["plus_two_30min"])
                    for idx, row in latest.iterrows()
                }

        model = DepModel()
        ts = datetime(2025, 1, 1, 10, 0)

        # Add data with 30min timeframe
        for i in range(5):
            t = ts - timedelta(minutes=30 * i)
            self.data.add_bars(
                [
                    Bar(
                        self.iid,
                        t,
                        100 + i,
                        101 + i,
                        99 + i,
                        100 + i,
                        1000,
                        timeframe="30min",
                    )
                ]
            )

        from src.core.alpha_engine import ModelRunConfig

        forecasts = AlphaEngine.run_model(
            self.data,
            model,
            [self.iid],
            ModelRunConfig(timestamp=ts, timeframe="30min"),
        )

        # AAPL close at t is 100. plus_one = 101, plus_two = 102.
        self.assertEqual(forecasts[self.iid], 102.0)


if __name__ == "__main__":
    unittest.main()
