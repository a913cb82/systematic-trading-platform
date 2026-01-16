import unittest
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from src.core.alpha_engine import AlphaModel, feature
from src.core.data_platform import Bar, DataPlatform


class TestFeatureDependencies(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform()
        self.iid = self.data.get_internal_id("AAPL")
        for i in range(5):
            ts = datetime(2025, 1, 1) + timedelta(days=i)
            self.data.add_bars(
                [Bar(self.iid, ts, 100, 101, 99, 100 + i, 1000)]
            )

    def test_recursive_hydration(self) -> None:
        # Define features with dependencies
        @feature("base_val")
        def base_val(df: pd.DataFrame) -> pd.Series:
            return df["close"]

        @feature("plus_one", dependencies=["base_val"])
        def plus_one(df: pd.DataFrame) -> pd.Series:
            return df["base_val"] + 1

        @feature("plus_two", dependencies=["plus_one"])
        def plus_two(df: pd.DataFrame) -> pd.Series:
            return df["plus_one"] + 1

        class DepModel(AlphaModel):
            def compute_signals(
                self, latest: pd.DataFrame, history: pd.DataFrame
            ) -> Dict[int, float]:
                return {
                    int(idx): float(row["plus_two"])
                    for idx, row in latest.iterrows()
                }

        model = DepModel(self.data, features=["plus_two"])
        ts = datetime(2025, 1, 1) + timedelta(days=4)
        forecasts = model.generate_forecasts([self.iid], ts)

        # AAPL close at day 4 is 104. plus_one = 105, plus_two = 106.
        self.assertEqual(forecasts[self.iid], 106.0)


if __name__ == "__main__":
    unittest.main()
