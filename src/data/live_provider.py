from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from datetime import datetime
from ..common.types import Bar

class LiveDataProvider(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the live data source."""
        pass

    @abstractmethod
    def subscribe_bars(self, internal_ids: List[int], callback: Callable[[Bar], None]) -> None:
        """Subscribe to real-time bars for a list of internal IDs."""
        pass

    @abstractmethod
    def get_latest_bar(self, internal_id: int) -> Optional[Bar]:
        """Fetch the most recent bar for a specific asset."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection."""
        pass
