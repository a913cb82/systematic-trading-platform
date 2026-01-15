from typing import Dict, Tuple
from datetime import datetime

class SafetyLayer:
    def __init__(self, max_weight: float = 0.2, price_collar_pct: float = 0.05):
        self.max_weight = max_weight
        self.price_collar_pct = price_collar_pct

    def validate_weights(self, weights: Dict[int, float]) -> Tuple[bool, str]:
        """
        Validates target weights against safety limits.
        """
        for iid, weight in weights.items():
            if abs(weight) > self.max_weight:
                return False, f"Weight for {iid} exceeds max_weight {self.max_weight}: {weight}"
        
        # Check sum of absolute weights (leverage)
        total_leverage = sum(abs(w) for w in weights.values())
        if total_leverage > 2.0: # Example leverage limit
             return False, f"Total leverage {total_leverage} exceeds limit 2.0"
             
        return True, "Success"

    def validate_fill(self, fill_price: float, market_price: float) -> bool:
        """
        Checks if fill price is within a collar of the market price.
        """
        if market_price == 0: return False
        deviation = abs(fill_price - market_price) / market_price
        return deviation <= self.price_collar_pct
