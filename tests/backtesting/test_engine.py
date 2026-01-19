from datetime import datetime
from typing import Any

from src.alpha_library.models import MomentumModel
from src.backtesting.demo import MarketDataMock
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.portfolio_manager import PortfolioManager
from src.core.types import Timeframe


def test_backtest_engine_completes_full_cycle_and_reports_stats(
    data_platform: Any,
) -> None:
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 3)
    tickers = ["AAPL", "MSFT"]

    # Use real MarketDataMock for high-fidelity test
    provider = MarketDataMock()
    data_platform.providers = [provider]
    data_platform.sync_data(tickers, start, end, timeframe=Timeframe.MIN_30)

    pm = PortfolioManager()
    engine = BacktestEngine(data_platform, pm)

    config = BacktestConfig(
        start_date=start,
        end_date=end,
        alpha_models=[MomentumModel()],
        weights=[1.0],
        tickers=tickers,
        timeframe=Timeframe.MIN_30,
    )

    report = engine.run(config)

    assert report["status"] == "ACTIVE"
    assert "total_return" in report
    assert "sharpe" in report
    assert len(engine.interval_results) > 0
