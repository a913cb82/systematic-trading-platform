import logging
from datetime import datetime, timedelta

from src.alpha.features import FeatureStore
from src.alpha.model import MeanReversionModel
from src.alpha.publisher import ForecastPublisher
from src.common.base import ExecutionEngine
from src.common.config import config
from src.common.utils import setup_logging
from src.data.alpaca_provider import AlpacaLiveProvider
from src.data.event_store import EventStore
from src.data.ism import InternalSecurityMaster
from src.data.live_provider import LiveDataProvider
from src.data.market_data import MarketDataEngine
from src.data.mock_live_provider import MockLiveProvider
from src.execution.algos import ExecutionAlgorithm
from src.execution.alpaca_engine import AlpacaExecutionEngine
from src.execution.alpaca_gateway import AlpacaBrokerGateway
from src.execution.engine import SimulatedExecutionEngine
from src.execution.oms import OrderManagementSystem
from src.execution.safety import SafetyLayer
from src.live_runner import LiveRunner
from src.portfolio.manager import PortfolioManager
from src.portfolio.optimizer import CvxpyOptimizer
from src.portfolio.publisher import TargetWeightPublisher
from src.portfolio.risk import RollingWindowRiskModel


def main() -> None:
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("main")

    # 1. Setup Infrastructure
    ism = InternalSecurityMaster(config.data.ism_db)
    mde = MarketDataEngine(config.data.market_path)
    event_store = EventStore(config.data.event_path)

    # Register some securities (Example)
    start_date = datetime.now() - timedelta(days=30)
    aapl_id = ism.register_security(
        "AAPL", "NASDAQ", start_date, sector="Technology"
    )
    msft_id = ism.register_security(
        "MSFT", "NASDAQ", start_date, sector="Technology"
    )
    internal_ids = [aapl_id, msft_id]

    # 2. Setup Alpha & Portfolio
    FeatureStore(mde, event_store)
    forecast_publisher = ForecastPublisher(config.data.forecast_db)
    alpha_model = MeanReversionModel(
        mde, internal_ids, publisher=forecast_publisher
    )

    risk_model = RollingWindowRiskModel(mde)
    optimizer = CvxpyOptimizer(risk_model)
    weight_publisher = TargetWeightPublisher(config.data.target_weights_db)
    portfolio_manager = PortfolioManager(
        forecast_publisher, optimizer, weight_publisher
    )

    # 3. Setup Execution & Live Provider based on config
    execution_engine: ExecutionEngine
    live_provider: LiveDataProvider

    if config.execution.provider == "alpaca":
        logger.info("Using Alpaca provider")
        gateway = AlpacaBrokerGateway(ism)
        execution_engine = AlpacaExecutionEngine(gateway, mde)
        live_provider = AlpacaLiveProvider(ism)
    else:
        logger.info("Using Mock provider")
        algo = ExecutionAlgorithm(mde)
        execution_engine = SimulatedExecutionEngine(algo)
        live_provider = MockLiveProvider()

    safety = SafetyLayer()
    OrderManagementSystem(
        weight_publisher,
        execution_engine,
        market_data=mde,
        safety_layer=safety,
    )

    # 4. Start Runner
    runner = LiveRunner(
        live_provider, mde, alpha_model, portfolio_manager, internal_ids
    )

    logger.info("System initialized and ready.")
    runner.start()


if __name__ == "__main__":
    main()
