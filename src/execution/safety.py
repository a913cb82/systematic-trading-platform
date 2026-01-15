from typing import Dict, Tuple, List
from datetime import datetime, timedelta


class SafetyLayer:
    def __init__(
        self,
        max_weight: float = 0.2,
        price_collar_pct: float = 0.05,
        max_messages_per_second: int = 10,
        max_drawdown_limit: float = -0.02,
    ):
        self.max_weight = max_weight
        self.price_collar_pct = price_collar_pct
        self.max_messages_per_second = max_messages_per_second
        self.max_drawdown_limit = max_drawdown_limit

        # State
        self.message_timestamps: List[datetime] = []
        self.initial_equity: float = 0.0
        self.current_drawdown: float = 0.0
        self.is_killed: bool = False

    def check_rate_limit(self) -> bool:
        """
        Implements a simple sliding window rate limiter.
        """
        now = datetime.now()
        # Remove timestamps older than 1 second
        self.message_timestamps = [
            t
            for t in self.message_timestamps
            if now - t < timedelta(seconds=1)
        ]

        if len(self.message_timestamps) >= self.max_messages_per_second:
            return False

        self.message_timestamps.append(now)
        return True

    def update_pnl(self, current_equity: float):
        """
        Updates the kill-switch state based on current equity.
        """
        if self.initial_equity == 0:
            self.initial_equity = current_equity
            return

        pnl_pct = (current_equity - self.initial_equity) / self.initial_equity
        if pnl_pct < self.max_drawdown_limit:
            self.is_killed = True
            print(
                f"CRITICAL: Kill-switch triggered. Drawdown {pnl_pct:.2%} exceeds limit {self.max_drawdown_limit:.2%}"
            )

    def validate_weights(self, weights: Dict[int, float]) -> Tuple[bool, str]:
        """
        Validates target weights against safety limits.
        """
        if self.is_killed:
            return False, "Kill-switch is active. No further trading allowed."

        if not self.check_rate_limit():
            return (
                False,
                f"Message rate limit of {self.max_messages_per_second}/s exceeded.",
            )

        for iid, weight in weights.items():
            if abs(weight) > self.max_weight:
                return (
                    False,
                    f"Weight for {iid} exceeds max_weight {self.max_weight}: {weight}",
                )

        # Check sum of absolute weights (leverage)
        total_leverage = sum(abs(w) for w in weights.values())
        if total_leverage > 2.0:  # Example leverage limit
            return False, f"Total leverage {total_leverage} exceeds limit 2.0"

        return True, "Success"

    def validate_fill(self, fill_price: float, market_price: float) -> bool:
        """
        Checks if fill price is within a collar of the market price.
        """
        if market_price == 0:
            return False
        deviation = abs(fill_price - market_price) / market_price
        return deviation <= self.price_collar_pct
