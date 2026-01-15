from typing import TypedDict, Any
from datetime import datetime


class Bar(TypedDict):
    internal_id: int
    timestamp: datetime
    timestamp_knowledge: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float


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


class CorporateAction(TypedDict):
    internal_id: int
    type: str  # 'SPLIT' | 'DIVIDEND'
    ex_date: datetime
    record_date: datetime
    pay_date: datetime
    ratio: float  # For splits: 2.0 for 2-for-1. For dividends: amount.
    timestamp_knowledge: datetime
