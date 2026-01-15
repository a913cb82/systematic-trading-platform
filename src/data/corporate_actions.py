import os
import sqlite3
from datetime import datetime
from typing import List, Optional

from ..common.types import CorporateAction


class CorporateActionMaster:
    def __init__(self, db_path: str = "data/corporate_actions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corporate_actions (
                    internal_id INTEGER,
                    type TEXT,
                    ex_date TEXT,
                    record_date TEXT,
                    pay_date TEXT,
                    ratio REAL,
                    timestamp_knowledge TEXT,
                    PRIMARY KEY (
                        internal_id, type, ex_date, timestamp_knowledge
                    )
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ca_id "
                "ON corporate_actions (internal_id, ex_date)"
            )

    def write_actions(self, actions: List[CorporateAction]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            for action in actions:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO corporate_actions
                    (
                        internal_id, type, ex_date, record_date,
                        pay_date, ratio, timestamp_knowledge
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        action["internal_id"],
                        action["type"],
                        action["ex_date"].strftime("%Y-%m-%d"),
                        (
                            action.get("record_date").strftime("%Y-%m-%d")  # type: ignore
                            if action.get("record_date")
                            else None
                        ),
                        (
                            action.get("pay_date").strftime("%Y-%m-%d")  # type: ignore
                            if action.get("pay_date")
                            else None
                        ),
                        action["ratio"],
                        action["timestamp_knowledge"].strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    ),
                )

    def get_actions(
        self,
        internal_ids: List[int],
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
    ) -> List[CorporateAction]:
        if as_of is None:
            as_of = datetime.now()

        as_of_str = as_of.strftime("%Y-%m-%d %H:%M:%S")
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        actions = []
        with sqlite3.connect(self.db_path) as conn:
            for internal_id in internal_ids:
                # Bitemporal query: latest knowledge for each event
                cursor = conn.execute(
                    """
                    SELECT
                        type, ex_date, record_date, pay_date,
                        ratio, timestamp_knowledge
                    FROM corporate_actions
                    WHERE internal_id = ?
                    AND ex_date >= ? AND ex_date <= ?
                    AND timestamp_knowledge <= ?
                    ORDER BY ex_date, timestamp_knowledge DESC
                """,
                    (internal_id, start_str, end_str, as_of_str),
                )

                # Deduplicate to get latest knowledge for each ex_date/type
                seen = set()
                for row in cursor.fetchall():
                    key = (row[0], row[1])  # (type, ex_date)
                    if key not in seen:
                        seen.add(key)
                        actions.append(
                            CorporateAction(
                                internal_id=internal_id,
                                type=row[0],
                                ex_date=datetime.strptime(row[1], "%Y-%m-%d"),
                                record_date=(
                                    datetime.strptime(row[2], "%Y-%m-%d")
                                    if row[2]
                                    else None
                                ),
                                pay_date=(
                                    datetime.strptime(row[3], "%Y-%m-%d")
                                    if row[3]
                                    else None
                                ),
                                ratio=row[4],
                                timestamp_knowledge=datetime.strptime(
                                    row[5], "%Y-%m-%d %H:%M:%S"
                                ),
                            )
                        )
        return actions
