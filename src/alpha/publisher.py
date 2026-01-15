import sqlite3
from datetime import datetime
from typing import Dict, Optional, List, Callable
from ..common.config import config


class ForecastPublisher:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.get(
            "data.forecast_db", "data/forecasts.db"
        )
        self._init_db()
        self._subscribers: List[
            Callable[[datetime, Dict[int, float]], None]
        ] = []

    def _init_db(self):
        import os

        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    timestamp TEXT,
                    internal_id INTEGER,
                    signal REAL,
                    PRIMARY KEY (timestamp, internal_id)
                )
            """
            )

    def submit_forecasts(
        self, timestamp: datetime, forecasts: Dict[int, float]
    ) -> None:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            for iid, signal in forecasts.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO forecasts (timestamp, internal_id, signal)
                    VALUES (?, ?, ?)
                """,
                    (ts_str, iid, signal),
                )

        # Notify subscribers
        for callback in self._subscribers:
            callback(timestamp, forecasts)

    def get_forecasts(self, timestamp: datetime) -> Dict[int, float]:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        forecasts = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT internal_id, signal FROM forecasts WHERE timestamp = ?",
                (ts_str,),
            )
            for row in cursor.fetchall():
                forecasts[row[0]] = row[1]
        return forecasts

    def subscribe_forecasts(
        self, on_forecast: Callable[[datetime, Dict[int, float]], None]
    ) -> None:
        self._subscribers.append(on_forecast)
