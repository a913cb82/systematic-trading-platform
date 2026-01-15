import os
import shutil
import unittest
from datetime import datetime, timedelta

from src.alpha.backtest_event_driven import EventDrivenBacktester
from src.alpha.features import FeatureStore
from src.alpha.model import MeanReversionModel
from src.common.types import Bar
from src.data.event_store import EventStore
from src.data.market_data import MarketDataEngine
from src.portfolio.optimizer import SimpleOptimizer


class TestEventDrivenBacktester(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_backtest_data"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

        self.market_data = MarketDataEngine(
            os.path.join(self.test_dir, "market")
        )
        self.event_store = EventStore(os.path.join(self.test_dir, "events"))
        self.feature_store = FeatureStore(self.market_data, self.event_store)
        self.optimizer = SimpleOptimizer()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_backtest_run(self):
        # 1. Create fake data: 10 days of prices
        start_date = datetime(2023, 1, 1)
        bars = []
        for i in range(10):
            ts = start_date + timedelta(days=i)
            # ID 1 goes up: 100, 101, 102...
            bars.append(
                Bar(
                    internal_id=1,
                    timestamp=ts,
                    timestamp_knowledge=ts,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.5 + i,
                    volume=1000.0,
                )
            )
            # ID 2 goes down: 100, 99, 98...
            bars.append(
                Bar(
                    internal_id=2,
                    timestamp=ts,
                    timestamp_knowledge=ts,
                    open=100.0 - i,
                    high=101.0 - i,
                    low=99.0 - i,
                    close=99.5 - i,
                    volume=1000.0,
                )
            )

        self.market_data.write_bars(bars)

        # 2. Setup Model
        model = MeanReversionModel(self.feature_store, [1, 2])

        # 3. Setup Backtester
        bt = EventDrivenBacktester(
            self.market_data,
            model,
            self.optimizer,
            start_date=start_date,
            end_date=start_date + timedelta(days=9),
            initial_capital=100000.0,
            transaction_cost_bps=0.0,  # No TC for simple test
        )

        # 4. Run
        bt.run()

        results = bt.get_results()
        metrics = bt.calculate_metrics()

        self.assertGreater(len(results), 0)
        self.assertIn("equity", results.columns)
        self.assertGreater(metrics.get("total_trades", 0), 0)

        print(f"\nFinal Equity: {metrics.get('final_equity')}")
        print(f"Total Trades: {metrics.get('total_trades')}")


if __name__ == "__main__":
    unittest.main()
