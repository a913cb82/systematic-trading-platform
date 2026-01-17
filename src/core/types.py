import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Timeframe(Enum):
    MINUTE = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR = "1hour"
    DAY = "1D"

    @property
    def minutes(self) -> int:
        """Returns the number of minutes in the timeframe."""
        mapping = {
            Timeframe.MINUTE: 1,
            Timeframe.MIN_5: 5,
            Timeframe.MIN_15: 15,
            Timeframe.MIN_30: 30,
            Timeframe.HOUR: 60,
            Timeframe.DAY: 1440,
        }
        return mapping[self]

    @property
    def pandas_freq(self) -> str:
        """Returns a string compatible with pandas frequency."""
        mapping = {
            Timeframe.MINUTE: "1min",
            Timeframe.MIN_5: "5min",
            Timeframe.MIN_15: "15min",
            Timeframe.MIN_30: "30min",
            Timeframe.HOUR: "1h",
            Timeframe.DAY: "D",
        }
        return mapping[self]

    @property
    def is_intraday(self) -> bool:
        return self != Timeframe.DAY


@dataclass
class Security:
    internal_id: int
    ticker: str
    start: datetime
    end: datetime
    extra: Any


@dataclass
class Bar:
    internal_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Timeframe = Timeframe.DAY
    timestamp_knowledge: datetime = field(default_factory=datetime.now)


@dataclass
class Event:
    internal_id: int
    timestamp: datetime
    event_type: str
    value: Any
    timestamp_knowledge: datetime = field(default_factory=datetime.now)


@dataclass
class CorporateAction:
    internal_id: int
    ex_date: datetime
    type: str  # SPLIT, DIVIDEND
    value: float


@dataclass
class QueryConfig:
    start: datetime
    end: datetime
    timeframe: Timeframe = Timeframe.DAY
    as_of: Optional[datetime] = None
    adjust: bool = True


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_active(self) -> bool:
        return self in (
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.PARTIAL,
        )


class Order:
    _id_counter = 1
    _id_lock = threading.Lock()

    def __init__(self, ticker: str, quantity: float, side: str):
        with Order._id_lock:
            self.order_id = Order._id_counter
            Order._id_counter += 1
        self.ticker = ticker
        self.quantity = quantity
        self.filled_qty = 0.0
        self.side = side
        self.state = OrderState.PENDING
        self.timestamp = datetime.now()

    def update(self, fill_qty: float) -> None:
        self.filled_qty += fill_qty
        self.state = (
            OrderState.FILLED
            if self.filled_qty >= self.quantity
            else OrderState.PARTIAL
        )


@dataclass
class ChildOrder:
    parent: Order
    quantity: float
    scheduled_at: float
