from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common.base import MarketDataProvider


class SafetyLayer:
    MAX_LEVERAGE = 2.0

    def __init__(
        self,
        max_weight: float = 0.2,
        price_collar_pct: float = 0.05,
        max_messages_per_second: int = 10,
        max_drawdown_limit: float = -0.02,
        max_adv_participation: float = 0.01,  # 1% of ADV
    ):
        self.max_weight = max_weight
        self.price_collar_pct = price_collar_pct
        self.max_messages_per_second = max_messages_per_second
        self.max_drawdown_limit = max_drawdown_limit
        self.max_adv_participation = max_adv_participation

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

    def update_pnl(self, current_equity: float) -> None:
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
                f"CRITICAL: Kill-switch triggered. Drawdown {pnl_pct:.2%} "
                f"exceeds limit {self.max_drawdown_limit:.2%}"
            )

    def validate_weights(
        self,
        weights: Dict[int, float],
        market_data: Optional[MarketDataProvider] = None,
        capital: float = 1_000_000.0,
    ) -> Tuple[bool, str]:
        """
        Validates target weights against safety limits, including ADV.
        """
        if self.is_killed:
            return False, "Kill-switch is active. No further trading allowed."

        if not self.check_rate_limit():
            return (
                False,
                f"Message rate limit of "
                f"{self.max_messages_per_second}/s exceeded.",
            )

        for iid, weight in weights.items():
            if abs(weight) > self.max_weight:
                return (
                    False,
                    f"Weight for {iid} exceeds max_weight "
                    f"{self.max_weight}: {weight}",
                )

            # ADV Check
            if market_data:
                # Use a 30-day window for ADV
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=30)
                bars = market_data.get_bars(
                    [iid], start_dt, end_dt, adjustment="RAW"
                )
                if bars:
                    adv = np.mean([b["volume"] for b in bars])
                    latest_price = bars[-1]["close"]

                    # Estimate required shares
                    target_val = weight * capital
                    target_shares = (
                        target_val / latest_price if latest_price > 0 else 0
                    )

                    if abs(target_shares) > adv * self.max_adv_participation:
                        return (
                            False,
                            f"Order for {iid} exceeds ADV limit. "
                            f"Target shares: {abs(target_shares):.0f}, "
                            f"ADV: {adv:.0f}, Limit: "
                            f"{self.max_adv_participation:.1%}",
                        )

        # Check sum of absolute weights (leverage)
        total_leverage = sum(abs(w) for w in weights.values())
        if total_leverage > self.MAX_LEVERAGE:  # Example leverage limit
            return (
                False,
                f"Total leverage {total_leverage} exceeds limit "
                f"{self.MAX_LEVERAGE}",
            )

        return True, "Success"

    def validate_fill(self, fill_price: float, market_price: float) -> bool:
        """
        Checks if fill price is within a collar of the market price.
        """
        if market_price == 0:
            return False
        deviation = abs(fill_price - market_price) / market_price
        return deviation <= self.price_collar_pct
