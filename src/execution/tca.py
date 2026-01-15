from typing import List, Dict
from datetime import datetime
import pandas as pd
from ..common.types import Trade

class TCAEngine:
    def calculate_slippage(self, fills: List[Trade], arrival_prices: Dict[int, float]) -> Dict[int, float]:
        """
        Calculates slippage for each trade relative to an arrival price.
        Slippage = (Fill Price - Arrival Price) / Arrival Price * 10000 (in bps)
        """
        slippage_results = {}
        for fill in fills:
            iid = fill['internal_id']
            if iid in arrival_prices:
                arrival = arrival_prices[iid]
                if arrival == 0: continue
                
                # For BUY: higher fill price is positive slippage (bad)
                # For SELL: lower fill price is positive slippage (bad)
                # Wait, usually slippage is defined as (Executed - Bench) * Side
                side_mult = 1 if fill['side'] == 'BUY' else -1
                bps = (fill['price'] - arrival) / arrival * 10000 * side_mult
                slippage_results[iid] = bps
        return slippage_results

    def attribution(self, fills: List[Trade], portfolio_value: float) -> Dict[str, float]:
        """
        Basic PnL attribution from trades.
        """
        total_fees = sum(f['fees'] for f in fills)
        return {
            "total_fees": total_fees,
            "trade_count": len(fills)
        }
