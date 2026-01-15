from datetime import datetime
from typing import Dict, List
import pandas as pd
from ..common.types import Trade, Bar
from ..data.market_data import MarketDataEngine


class ExecutionAlgorithm:
    def __init__(self, market_data_engine: MarketDataEngine):
        self.market_data_engine = market_data_engine

    def simulate_fills(
        self,
        timestamp: datetime,
        target_weights: Dict[int, float],
        current_portfolio_value: float = 1_000_000.0,
    ) -> List[Trade]:
        """
        Simple execution: fills at the 'close' price of the current timestamp.
        """
        internal_ids = list(target_weights.keys())
        # We look for bars at or slightly before this timestamp
        bars = self.market_data_engine.get_bars(
            internal_ids, timestamp, timestamp
        )

        # If no bars exactly at timestamp, try to find the latest ones
        if not bars:
            # In a real backtest, this would be more sophisticated
            return []

        fills = []
        bar_map = {b["internal_id"]: b for b in bars}

        for iid, weight in target_weights.items():
            if iid in bar_map:
                bar = bar_map[iid]
                # Simple fill at close
                fill = Trade(
                    internal_id=iid,
                    side="BUY" if weight > 0 else "SELL",  # Simplified
                    quantity=abs(
                        weight * current_portfolio_value / bar["close"]
                    ),
                    price=bar["close"],
                    fees=0.0,
                    venue="SIM",
                    timestamp=timestamp,
                )
                fills.append(fill)
        return fills
