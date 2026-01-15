import logging
import time
from datetime import datetime, timedelta
from src.data.ism import InternalSecurityMaster
from src.data.market_data import MarketDataEngine
from src.data.event_store import EventStore
from src.alpha.features import FeatureStore
from src.alpha.model import MeanReversionModel
from src.alpha.publisher import ForecastPublisher
from src.portfolio.risk import RollingWindowRiskModel
from src.portfolio.optimizer import CvxpyOptimizer
from src.portfolio.publisher import TargetWeightPublisher
from src.portfolio.manager import PortfolioManager
from src.execution.engine import SimulatedExecutionEngine
from src.execution.algos import ExecutionAlgorithm
from src.execution.oms import OrderManagementSystem
from src.execution.safety import SafetyLayer
from src.data.mock_live_provider import MockLiveProvider
from src.live_runner import LiveRunner
from src.common.utils import setup_logging

def main():
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("main")
    
    # 1. Setup Infrastructure
    ism = InternalSecurityMaster("data/trading_system.db")
    mde = MarketDataEngine("data/market")
    event_store = EventStore("data/event")
    
    # Register some securities
    start_date = datetime.now() - timedelta(days=30)
    aapl_id = ism.register_security("AAPL", "NASDAQ", start_date, sector="Technology")
    msft_id = ism.register_security("MSFT", "NASDAQ", start_date, sector="Technology")
    internal_ids = [aapl_id, msft_id]

    # Pre-populate some historical data for the models to work
    logger.info("Pre-populating historical data...")
    for iid, price in [(aapl_id, 150.0), (msft_id, 250.0)]:
        historical_bars = []
        for i in range(20):
            ts = start_date + timedelta(days=i)
            p = price + (i * 0.5)
            historical_bars.append({
                "internal_id": iid,
                "timestamp": ts,
                "timestamp_knowledge": ts,
                "open": p, "high": p+1, "low": p-1, "close": p, 
                "volume": 1000, "buy_volume": 500, "sell_volume": 500
            })
        mde.write_bars(historical_bars)

    # 2. Setup Alpha & Portfolio
    feature_store = FeatureStore(mde, event_store)
    forecast_publisher = ForecastPublisher("data/forecasts.db")
    alpha_model = MeanReversionModel(feature_store, internal_ids, publisher=forecast_publisher)
    
    risk_model = RollingWindowRiskModel(mde)
    optimizer = CvxpyOptimizer(risk_model)
    weight_publisher = TargetWeightPublisher("data/target_weights.db")
    portfolio_manager = PortfolioManager(forecast_publisher, optimizer, weight_publisher)
    
    # 3. Setup Execution
    algo = ExecutionAlgorithm(mde)
    execution_engine = SimulatedExecutionEngine(algo)
    safety = SafetyLayer()
    oms = OrderManagementSystem(weight_publisher, execution_engine, safety)
    
    # 4. Setup Live Ingestion
    live_provider = MockLiveProvider()
    
    # 5. Start Runner
    runner = LiveRunner(
        live_provider,
        mde,
        alpha_model,
        portfolio_manager,
        internal_ids
    )
    
    logger.info("System initialized and ready.")
    runner.start()

if __name__ == "__main__":
    main()
