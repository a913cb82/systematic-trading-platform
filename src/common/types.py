from typing import TypedDict, Any
from datetime import datetime

class Bar(TypedDict):
    internal_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class Trade(TypedDict):
    internal_id: int
    side: str  # 'BUY' | 'SELL'
    quantity: float
    price: float
    fees: float
    venue: str
    timestamp: datetime

class Event(TypedDict):
    internal_id: int
    type: str
    value: Any
    timestamp_event: datetime
    timestamp_knowledge: datetime
