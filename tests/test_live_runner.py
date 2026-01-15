import unittest
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.data.market_data import MarketDataEngine
from src.alpha.model import MeanReversionModel
from src.alpha.publisher import ForecastPublisher
from src.portfolio.manager import PortfolioManager
from src.portfolio.optimizer import SimpleOptimizer
from src.portfolio.publisher import TargetWeightPublisher
from src.data.mock_live_provider import MockLiveProvider
from src.live_runner import LiveRunner
from src.alpha.features import FeatureStore
from src.data.event_store import EventStore

class TestLiveRunner(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.market_path = os.path.join(self.tmp_dir, "market")
        self.event_path = os.path.join(self.tmp_dir, "event")
        self.forecast_db = os.path.join(self.tmp_dir, "forecasts.db")
        self.target_weights_db = os.path.join(self.tmp_dir, "target_weights.db")

        self.mde = MarketDataEngine(self.market_path)
        self.event_store = EventStore(self.event_path)
        self.feature_store = FeatureStore(self.mde, self.event_store)
        self.forecast_publisher = ForecastPublisher(self.forecast_db)
        
        self.internal_ids = [1, 2]
        self.alpha_model = MeanReversionModel(self.feature_store, self.internal_ids, publisher=self.forecast_publisher)
        
        # Simple setup for portfolio
        self.optimizer = SimpleOptimizer()
        self.weight_publisher = TargetWeightPublisher(self.target_weights_db)
        self.portfolio_manager = PortfolioManager(self.forecast_publisher, self.optimizer, self.weight_publisher)
        
        self.live_provider = MockLiveProvider()
        self.runner = LiveRunner(
            self.live_provider,
            self.mde,
            self.alpha_model,
            self.portfolio_manager,
            self.internal_ids
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_on_live_bar_flow(self):
        """
        Tests that receiving a live bar triggers the expected pipeline.
        """
        timestamp = datetime.now()
        bar = {
            "internal_id": 1,
            "timestamp": timestamp,
            "timestamp_knowledge": timestamp,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 1000, "buy_volume": 500, "sell_volume": 500
        }

        # Mock alpha_model to verify it gets called
        self.alpha_model.generate_forecasts = MagicMock(return_value={1: -0.01})
        
        # Trigger callback
        self.runner._on_live_bar(bar)
        
        # 1. Verify persistence
        stored_bars = self.mde.get_bars([1], timestamp, timestamp)
        self.assertEqual(len(stored_bars), 1)
        
        # 2. Verify alpha model call
        self.alpha_model.generate_forecasts.assert_called_with(timestamp)

if __name__ == "__main__":
    unittest.main()
