import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from typing import List, cast

import numpy as np

from src.alpha.backtest_event_driven import EventDrivenBacktester
from src.alpha.features import FeatureStore
from src.alpha.model import MeanReversionModel
from src.common.types import Bar
from src.data.event_store import EventStore
from src.data.market_data import MarketDataEngine
from src.portfolio.optimizer import CvxpyOptimizer
from src.portfolio.risk import RollingWindowRiskModel


class TestFullSystem(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.market_path = os.path.join(self.tmp_dir, "market")
        self.event_path = os.path.join(self.tmp_dir, "event")

        self.mde = MarketDataEngine(base_path=self.market_path)
        self.event_store = EventStore(self.event_path)
        self.feature_store = FeatureStore(self.mde, self.event_store)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_full_backtest_run(self):
        # 1. Generate dummy market data for 2 assets
        # Asset 1: Mean reverting around 100
        # Asset 2: Mean reverting around 200
        start_date = datetime(2023, 1, 1)
        num_days = 30

        for iid, base_price in [(1, 100.0), (2, 200.0)]:
            bars = cast(
                List[Bar],
                [
                    {
                        "internal_id": iid,
                        "timestamp": start_date + timedelta(days=i),
                        "open": base_price
                        + np.sin(i * 0.5) * 5
                        + np.random.normal(0, 1),
                        "high": base_price
                        + np.sin(i * 0.5) * 5
                        + np.random.normal(0, 1)
                        + 1,
                        "low": base_price
                        + np.sin(i * 0.5) * 5
                        + np.random.normal(0, 1)
                        - 1,
                        "close": base_price
                        + np.sin(i * 0.5) * 5
                        + np.random.normal(0, 1),
                        "volume": 10000.0,
                    }
                    for i in range(num_days)
                ],
            )
            self.mde.write_bars(bars)

        # 2. Setup components
        alpha_model = MeanReversionModel(
            self.feature_store, internal_ids=[1, 2]
        )
        risk_model = RollingWindowRiskModel(
            market_data_engine=self.mde, window_days=10
        )
        optimizer = CvxpyOptimizer(risk_model=risk_model, lambda_risk=0.5)

        # 3. Initialize Backtester
        backtester = EventDrivenBacktester(
            market_data=self.mde,
            alpha_model=alpha_model,
            optimizer=optimizer,
            start_date=start_date
            + timedelta(days=10),  # Start after we have some data for models
            end_date=start_date + timedelta(days=num_days - 1),
        )

        # 4. Run
        backtester.run()

        # 5. Verify Results
        metrics = backtester.calculate_metrics()
        self.assertIn("sharpe", metrics)
        self.assertGreater(metrics["final_equity"], 0)

        results = backtester.get_results()
        self.assertFalse(results.empty)

        print("\nFull System Backtest Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    unittest.main()
