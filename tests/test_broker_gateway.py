import unittest
from datetime import datetime
from src.execution.broker_gateway import MockBrokerGateway
from src.execution.engine import SimulatedExecutionEngine
from src.execution.algos import ExecutionAlgorithm
from src.data.market_data import MarketDataEngine

class TestBrokerGateway(unittest.TestCase):
    def test_mock_broker_flow(self):
        # We don't need real market data for the mock gateway logic test
        mde = MarketDataEngine() 
        algo = ExecutionAlgorithm(mde)
        engine = SimulatedExecutionEngine(algo)
        gateway = MockBrokerGateway(engine)
        
        # Test connection
        self.assertTrue(gateway.connect())
        
        # Test fill subscription/reporting
        fills = []
        gateway.subscribe_fills(lambda f: fills.append(f))
        
        # Test order submission
        order_id = gateway.submit_order(1, 'BUY', 100.0)
        self.assertTrue(order_id.startswith("ORDER_"))
        
        # Since MockBrokerGateway.submit_order immediately reports a fill:
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]['internal_id'], 1)
        self.assertEqual(fills[0]['side'], 'BUY')

if __name__ == "__main__":
    unittest.main()