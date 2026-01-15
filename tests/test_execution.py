import os
import shutil
import tempfile
import unittest
from datetime import datetime
from typing import List, cast

from src.common.types import Bar
from src.data.market_data import MarketDataEngine
from src.execution.algos import ExecutionAlgorithm
from src.execution.engine import SimulatedExecutionEngine
from src.execution.oms import OrderManagementSystem
from src.execution.safety import SafetyLayer
from src.portfolio.publisher import TargetWeightPublisher


class TestExecution(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "weights.db")
        self.market_path = os.path.join(self.tmp_dir, "market")

        self.publisher = TargetWeightPublisher(db_path=self.db_path)
        self.mde = MarketDataEngine(base_path=self.market_path)
        self.algo = ExecutionAlgorithm(market_data_engine=self.mde)
        self.engine = SimulatedExecutionEngine(algo=self.algo)
        self.safety = SafetyLayer(max_weight=0.5)
        self.oms = OrderManagementSystem(
            self.publisher,
            self.engine,
            market_data=self.mde,
            safety_layer=self.safety,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_oms_execution_flow(self):
        timestamp = datetime(2023, 1, 1, 10, 0)

        # Mock market data for the algo to find a price
        bars = cast(
            List[Bar],
            [
                {
                    "internal_id": 1,
                    "timestamp": timestamp,
                    "close": 150.0,
                    "open": 150.0,
                    "high": 151.0,
                    "low": 149.0,
                    "volume": 10000.0,
                }
            ],
        )
        self.mde.write_bars(bars)

        # Publisher submits target weights -> OMS should pick it up ->
        # Engine should execute
        weights = {1: 0.1}
        self.publisher.submit_target_weights(timestamp, weights)

        # Verify engine received and processed trades
        fills = self.engine.get_fills(timestamp, timestamp)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["internal_id"], 1)
        self.assertEqual(fills[0]["price"], 150.0)

    def test_safety_check_rejection(self):
        timestamp = datetime(2023, 1, 1, 10, 0)

        # Submit weight exceeding safety limit (max_weight=0.5)
        weights = {1: 0.6}
        self.publisher.submit_target_weights(timestamp, weights)

        # Verify no trades were executed
        fills = self.engine.get_fills(timestamp, timestamp)
        self.assertEqual(len(fills), 0)


if __name__ == "__main__":
    unittest.main()
