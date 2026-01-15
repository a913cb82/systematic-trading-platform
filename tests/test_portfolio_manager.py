import os
import shutil
import tempfile
import unittest
from datetime import datetime
from typing import List, cast

from src.alpha.publisher import ForecastPublisher
from src.common.types import Bar
from src.data.market_data import MarketDataEngine
from src.portfolio.manager import PortfolioManager
from src.portfolio.optimizer import CvxpyOptimizer
from src.portfolio.publisher import TargetWeightPublisher
from src.portfolio.risk import RollingWindowRiskModel


class TestPortfolioManager(unittest.TestCase):
    def test_portfolio_manager_flow(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            # Setup
            forecast_db = os.path.join(tmp_dir, "forecasts.db")
            weights_db = os.path.join(tmp_dir, "weights.db")
            market_data_path = os.path.join(tmp_dir, "market")

            publisher = ForecastPublisher(db_path=forecast_db)
            mde = MarketDataEngine(base_path=market_data_path)
            risk_model = RollingWindowRiskModel(market_data_engine=mde)
            optimizer = CvxpyOptimizer(risk_model=risk_model)
            weight_publisher = TargetWeightPublisher(db_path=weights_db)

            PortfolioManager(publisher, optimizer, weight_publisher)

            # Mock some market data for risk model
            timestamp = datetime(2023, 1, 1, 12, 0)
            bars = cast(
                List[Bar],
                [
                    {
                        "internal_id": 1,
                        "timestamp": timestamp,
                        "close": 100.0,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "volume": 1000.0,
                    },
                    {
                        "internal_id": 2,
                        "timestamp": timestamp,
                        "close": 200.0,
                        "open": 200.0,
                        "high": 202.0,
                        "low": 198.0,
                        "volume": 500.0,
                    },
                ],
            )
            mde.write_bars(bars)

            # Submit forecasts
            forecasts = {1: 0.05, 2: -0.02}
            publisher.submit_forecasts(timestamp, forecasts)

            # Check if target weights were published
            published_weights = weight_publisher.get_target_weights(timestamp)
            self.assertGreater(len(published_weights), 0)
            self.assertIn(1, published_weights)
            self.assertGreater(published_weights[1], 0)
            self.assertAlmostEqual(
                sum(published_weights.values()), 1.0, places=5
            )
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
