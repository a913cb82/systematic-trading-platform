import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from src.alpha.features import FeatureStore
from src.alpha.model import MeanReversionModel
from src.alpha.publisher import ForecastPublisher
from src.data.event_store import EventStore
from src.data.ism import InternalSecurityMaster
from src.data.market_data import MarketDataEngine
from src.data.mock_live_provider import MockLiveProvider
from src.execution.algos import ExecutionAlgorithm
from src.execution.engine import SimulatedExecutionEngine
from src.execution.oms import OrderManagementSystem
from src.execution.safety import SafetyLayer
from src.live_runner import LiveRunner
from src.portfolio.manager import PortfolioManager
from src.portfolio.optimizer import CvxpyOptimizer
from src.portfolio.publisher import TargetWeightPublisher
from src.portfolio.risk import RollingWindowRiskModel


class TestInitialization(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "trading_system.db")
        self.market_path = os.path.join(self.tmp_dir, "market")
        self.event_path = os.path.join(self.tmp_dir, "event")
        self.forecast_db = os.path.join(self.tmp_dir, "forecasts.db")
        self.target_weights_db = os.path.join(
            self.tmp_dir, "target_weights.db"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_full_system_instantiation(self):
        """
        Tests that all components from main.py can be instantiated and linked.
        This would catch NameErrors (like missing Optional import).
        """
        # 1. Setup Infrastructure
        ism = InternalSecurityMaster(self.db_path)
        mde = MarketDataEngine(self.market_path)
        event_store = EventStore(self.event_path)

        # Register some securities
        start_date = datetime.now() - timedelta(days=30)
        aapl_id = ism.register_security(
            "AAPL", "NASDAQ", start_date, sector="Technology"
        )
        msft_id = ism.register_security(
            "MSFT", "NASDAQ", start_date, sector="Technology"
        )
        internal_ids = [aapl_id, msft_id]

        # 2. Setup Alpha & Portfolio
        feature_store = FeatureStore(mde, event_store)
        forecast_publisher = ForecastPublisher(self.forecast_db)
        alpha_model = MeanReversionModel(
            feature_store, internal_ids, publisher=forecast_publisher
        )

        risk_model = RollingWindowRiskModel(mde)
        optimizer = CvxpyOptimizer(risk_model)
        weight_publisher = TargetWeightPublisher(self.target_weights_db)
        portfolio_manager = PortfolioManager(
            forecast_publisher, optimizer, weight_publisher
        )

        # 3. Setup Execution
        algo = ExecutionAlgorithm(mde)
        execution_engine = SimulatedExecutionEngine(algo)
        safety = SafetyLayer()
        OrderManagementSystem(weight_publisher, execution_engine, safety)

        # 4. Setup Live Ingestion
        live_provider = MockLiveProvider()

        # 5. Start Runner
        runner = LiveRunner(
            live_provider, mde, alpha_model, portfolio_manager, internal_ids
        )

        self.assertIsNotNone(runner)
        self.assertEqual(runner.internal_ids, internal_ids)


if __name__ == "__main__":
    unittest.main()
