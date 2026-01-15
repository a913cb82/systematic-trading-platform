import unittest
import os
import shutil
from datetime import datetime, timedelta
from src.alpha.model import MeanReversionModel
from src.alpha.features import FeatureStore
from src.data.market_data import MarketDataEngine
from src.data.event_store import EventStore
from src.common.types import Bar

class TestAlphaModel(unittest.TestCase):
    def setUp(self):
        self.market_dir = "test_market_alpha"
        self.event_dir = "test_event_alpha"
        if os.path.exists(self.market_dir): shutil.rmtree(self.market_dir)
        if os.path.exists(self.event_dir): shutil.rmtree(self.event_dir)
        
        self.market_data = MarketDataEngine(self.market_dir)
        self.event_store = EventStore(self.event_dir)
        self.feature_store = FeatureStore(self.market_data, self.event_store)
        self.model = MeanReversionModel(self.feature_store, [1])

    def tearDown(self):
        if os.path.exists(self.market_dir): shutil.rmtree(self.market_dir)
        if os.path.exists(self.event_dir): shutil.rmtree(self.event_dir)

    def test_mean_reversion_forecast(self):
        # Write bars: price going up
        bars = [
            Bar(internal_id=1, timestamp=datetime(2023, 1, 1), timestamp_knowledge=datetime(2023, 1, 1), open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0),
            Bar(internal_id=1, timestamp=datetime(2023, 1, 2), timestamp_knowledge=datetime(2023, 1, 2), open=100.0, high=102.0, low=100.0, close=105.0, volume=1000.0),
        ]
        self.market_data.write_bars(bars)
        
        forecasts = self.model.generate_forecasts(datetime(2023, 1, 2))
        self.assertIn(1, forecasts)
        # Return was (105-100)/100 = 0.05. Forecast should be -0.05
        self.assertAlmostEqual(forecasts[1], -0.05)

if __name__ == "__main__":
    unittest.main()
