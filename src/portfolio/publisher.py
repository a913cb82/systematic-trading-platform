import os
import sqlite3
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..common.config import config


class TargetWeightPublisher:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.get(
            "data.target_weights_db", "data/target_weights.db"
        )
        self._init_db()
        self._subscribers: List[
            Callable[[datetime, Dict[int, float]], None]
        ] = []

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS target_weights (
                    timestamp TEXT,
                    internal_id INTEGER,
                    weight REAL,
                    PRIMARY KEY (timestamp, internal_id)
                )
            """
            )

    def submit_target_weights(
        self, timestamp: datetime, weights: Dict[int, float]
    ) -> None:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            for iid, weight in weights.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO target_weights
                    (timestamp, internal_id, weight)
                    VALUES (?, ?, ?)
                """,
                    (ts_str, iid, weight),
                )

        # Notify subscribers
        for callback in self._subscribers:
            callback(timestamp, weights)

    def get_target_weights(self, timestamp: datetime) -> Dict[int, float]:
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        weights = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT internal_id, weight FROM target_weights
                WHERE timestamp = ?
                """,
                (ts_str,),
            )
            for row in cursor.fetchall():
                weights[row[0]] = row[1]
        return weights

    def subscribe_target_weights(
        self, on_weights: Callable[[datetime, Dict[int, float]], None]
    ) -> None:
        self._subscribers.append(on_weights)
