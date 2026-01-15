import logging
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)


class OrderInstruction(TypedDict):
    internal_id: int
    side: str
    quantity: float


class OrderGenerator:
    """
    Converts target weights into specific order instructions.
    """

    def __init__(self, lot_size: int = 1) -> None:
        self.lot_size = lot_size

    def generate_orders(
        self,
        target_weights: Dict[int, float],
        current_positions: Dict[int, float],
        total_capital: float,
        current_prices: Dict[int, float],
    ) -> List[OrderInstruction]:
        orders: List[OrderInstruction] = []

        # 1. Calculate target shares for each asset
        target_shares: Dict[int, float] = {}
        for iid, weight in target_weights.items():
            price = current_prices.get(iid)
            if price is None or price <= 0:
                logger.warning(f"Skipping iid {iid}: invalid price {price}")
                continue

            # Target Value = Capital * Weight
            # Target Shares = Target Value / Price
            raw_shares = (total_capital * weight) / price

            # Round to lot size
            target_shares[iid] = (
                round(raw_shares / self.lot_size) * self.lot_size
            )

        # 2. Compare with current positions to find delta
        all_ids = set(target_shares.keys()) | set(current_positions.keys())

        for iid in all_ids:
            target = target_shares.get(iid, 0.0)
            current = current_positions.get(iid, 0.0)
            diff = target - current

            if abs(diff) < self.lot_size:
                continue

            side = "BUY" if diff > 0 else "SELL"
            orders.append(
                {"internal_id": iid, "side": side, "quantity": abs(diff)}
            )

        return orders
