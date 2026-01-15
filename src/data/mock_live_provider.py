import random
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..common.types import Bar
from .live_provider import LiveDataProvider


class MockLiveProvider(LiveDataProvider):
    """
    Simulates a live data provider by generating random price movements.
    """

    def __init__(self):
        self.internal_ids: List[int] = []
        self.callbacks: List[Callable[[Bar], None]] = []
        self.last_bars: Dict[int, Bar] = {}
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        self.running = True
        self.thread = threading.Thread(target=self._generate_data, daemon=True)
        self.thread.start()
        return True

    def subscribe_bars(
        self, internal_ids: List[int], callback: Callable[[Bar], None]
    ) -> None:
        self.internal_ids.extend(internal_ids)
        self.callbacks.append(callback)

        # Initialize last_bars with some starting price
        for iid in internal_ids:
            if iid not in self.last_bars:
                self.last_bars[iid] = self._create_bar(iid, 100.0)

    def get_latest_bar(self, internal_id: int) -> Optional[Bar]:
        return self.last_bars.get(internal_id)

    def disconnect(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _create_bar(self, internal_id: int, price: float) -> Bar:
        return {
            "internal_id": internal_id,
            "timestamp": datetime.now(),
            "timestamp_knowledge": datetime.now(),
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": random.uniform(1000, 5000),
            "buy_volume": random.uniform(500, 2500),
            "sell_volume": random.uniform(500, 2500),
        }

    def _generate_data(self):
        while self.running:
            time.sleep(1.0)  # Generate data every second
            for iid in self.internal_ids:
                last_bar = self.last_bars[iid]
                # Random walk
                new_price = last_bar["close"] * (
                    1 + random.uniform(-0.001, 0.001)
                )
                new_bar = self._create_bar(iid, new_price)
                self.last_bars[iid] = new_bar

                for callback in self.callbacks:
                    callback(new_bar)
