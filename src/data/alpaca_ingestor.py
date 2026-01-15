import logging
from datetime import datetime
from typing import List, cast

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from ..common.config import config
from ..common.types import Bar
from ..data.ism import InternalSecurityMaster
from ..data.market_data import MarketDataEngine

logger = logging.getLogger(__name__)


class AlpacaBulkIngestor:
    """
    Utility to bulk-download historical data from Alpaca and store it locally.
    """

    def __init__(
        self,
        ism: InternalSecurityMaster,
        market_data_engine: MarketDataEngine,
    ) -> None:
        self.ism = ism
        self.market_data_engine = market_data_engine
        alpaca_cfg = config.execution.alpaca
        self.client = StockHistoricalDataClient(
            api_key=alpaca_cfg.key, secret_key=alpaca_cfg.secret
        )

    def ingest(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> None:
        """
        Downloads and writes historical bars for a set of internal_ids.
        """
        now = datetime.now()
        id_to_ticker = {}
        for iid in internal_ids:
            info = self.ism.get_symbol_info(iid, now)
            if info:
                id_to_ticker[iid] = info["ticker"]
            else:
                logger.warning(f"Could not resolve ticker for iid {iid}")

        if not id_to_ticker:
            logger.error("No valid tickers found for ingestion.")
            return

        tickers = list(id_to_ticker.values())
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        try:
            logger.info(f"Fetching bars for {tickers} from {start} to {end}")
            bars_data = cast(
                BarSet, self.client.get_stock_bars(request_params)
            )

            # Convert to our Bar format and write
            all_bars: List[Bar] = []
            for ticker, bars in bars_data.data.items():
                iid = next(k for k, v in id_to_ticker.items() if v == ticker)
                for b in bars:
                    all_bars.append(
                        cast(
                            Bar,
                            {
                                "internal_id": iid,
                                "timestamp": b.timestamp,
                                "timestamp_knowledge": now,
                                "open": float(b.open),
                                "high": float(b.high),
                                "low": float(b.low),
                                "close": float(b.close),
                                "volume": float(b.volume),
                            },
                        )
                    )

            if all_bars:
                logger.info(
                    f"Writing {len(all_bars)} bars to MarketDataEngine"
                )
                self.market_data_engine.write_bars(all_bars)
            else:
                logger.warning("No bars received from Alpaca.")

        except Exception as e:
            logger.error(f"Alpaca ingestion failed: {e}")
            raise
