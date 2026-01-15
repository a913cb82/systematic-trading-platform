import time
import logging
from datetime import datetime
from typing import List, Dict
from .data.live_provider import LiveDataProvider
from .data.market_data import MarketDataEngine
from .alpha.model import AlphaModel
from .portfolio.manager import PortfolioManager
from .common.monitoring import monitor
from .common.types import Bar

logger = logging.getLogger(__name__)


class LiveRunner:
    def __init__(
        self,
        live_provider: LiveDataProvider,
        market_data_engine: MarketDataEngine,
        alpha_model: AlphaModel,
        portfolio_manager: PortfolioManager,
        internal_ids: List[int],
    ):
        self.live_provider = live_provider
        self.market_data_engine = market_data_engine
        self.alpha_model = alpha_model
        self.portfolio_manager = portfolio_manager
        self.internal_ids = internal_ids
        self.running = False

    def start(self):
        """
        Starts the live trading loop.
        """
        logger.info("Starting Live Runner...")
        self.running = True

        # 1. Connect to live data
        if not self.live_provider.connect():
            monitor.alert(
                "CRITICAL", "Failed to connect to Live Data Provider"
            )
            return

        # 2. Subscribe to bars
        # When a live bar arrives, we:
        # a) Write it to the MarketDataEngine (persistence)
        # b) The MarketDataEngine notifies its subscribers (which could be features)
        # c) We trigger the Alpha/Portfolio pipeline
        self.live_provider.subscribe_bars(self.internal_ids, self._on_live_bar)

        monitor.heartbeat("LiveRunner")
        logger.info(f"Live Runner active for IDs: {self.internal_ids}")

        try:
            while self.running:
                time.sleep(10)
                monitor.heartbeat("LiveRunner")
                if not monitor.check_health():
                    logger.warning("System health check failed!")
        except KeyboardInterrupt:
            self.stop()

    def _on_live_bar(self, bar: Bar):
        """
        Core callback for live data ingestion.
        """
        logger.debug(
            f"Received live bar for {bar['internal_id']} at {bar['timestamp']}"
        )

        # 1. Persist to Market Data Engine
        # This allows bitemporal queries to work for the Alpha model immediately
        self.market_data_engine.write_bars([bar])

        # 2. Trigger Alpha Generation
        # The alpha model will generate forecasts and publish them (if configured)
        timestamp = bar["timestamp"]

        try:
            self.alpha_model.generate_forecasts(timestamp)
        except Exception as e:
            monitor.alert(
                "ERROR", f"Failed to process live bar: {e}", {"bar": bar}
            )

    def stop(self):
        logger.info("Stopping Live Runner...")
        self.running = False
        self.live_provider.disconnect()
