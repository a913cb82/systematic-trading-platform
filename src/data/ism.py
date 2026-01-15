import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional


class InternalSecurityMaster:
    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ism_mapping (
                    internal_id INTEGER,
                    ticker TEXT,
                    exchange TEXT,
                    sector TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    timestamp_knowledge TEXT,
                    PRIMARY KEY (internal_id, start_date, timestamp_knowledge)
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ism_ticker "
                "ON ism_mapping (ticker, exchange)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ism_id "
                "ON ism_mapping (internal_id)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS external_mappings (
                    internal_id INTEGER,
                    id_type TEXT,
                    id_value TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    timestamp_knowledge TEXT,
                    PRIMARY KEY (
                        internal_id, id_type, start_date, timestamp_knowledge
                    )
                )
            """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ext_val "
                "ON external_mappings (id_type, id_value)"
            )

    def register_security(
        self,
        ticker: str,
        exchange: str,
        start_date: datetime,
        sector: Optional[str] = None,
        internal_id: Optional[int] = None,
    ) -> int:
        """
        Registers a ticker/exchange mapping. If internal_id is provided,
        it links to that ID. Otherwise, it creates a new one.
        """
        start_date_str = start_date.strftime("%Y-%m-%d")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            if internal_id is None:
                # Find max internal_id from both tables to be safe
                cursor = conn.execute(
                    "SELECT MAX(internal_id) FROM ism_mapping"
                )
                max_id_ism = cursor.fetchone()[0] or 1000
                cursor = conn.execute(
                    "SELECT MAX(internal_id) FROM external_mappings"
                )
                max_id_ext = cursor.fetchone()[0] or 1000
                internal_id = max(max_id_ism, max_id_ext) + 1

            # Close any existing record for this internal_id in ism_mapping
            conn.execute(
                """
                UPDATE ism_mapping
                SET end_date = ?
                WHERE internal_id = ? AND end_date IS NULL
            """,
                (start_date_str, internal_id),
            )

            # Insert new record
            conn.execute(
                """
                INSERT INTO ism_mapping (
                    internal_id, ticker, exchange, sector,
                    start_date, end_date, timestamp_knowledge
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    internal_id,
                    ticker,
                    exchange,
                    sector,
                    start_date_str,
                    None,
                    now_str,
                ),
            )

            return internal_id

    def delist_security(self, internal_id: int, end_date: datetime):
        end_date_str = end_date.strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE ism_mapping
                SET end_date = ?
                WHERE internal_id = ? AND end_date IS NULL
            """,
                (end_date_str, internal_id),
            )

    def add_external_mapping(
        self,
        internal_id: int,
        id_type: str,
        id_value: str,
        start_date: datetime,
    ):
        start_date_str = start_date.strftime("%Y-%m-%d")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            # Close existing
            conn.execute(
                """
                UPDATE external_mappings
                SET end_date = ?
                WHERE internal_id = ? AND id_type = ? AND end_date IS NULL
            """,
                (start_date_str, internal_id, id_type),
            )

            # Insert new
            conn.execute(
                """
                INSERT INTO external_mappings (
                    internal_id, id_type, id_value,
                    start_date, end_date, timestamp_knowledge
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    internal_id,
                    id_type,
                    id_value,
                    start_date_str,
                    None,
                    now_str,
                ),
            )

    def get_internal_id(
        self,
        ticker: str,
        exchange: str,
        date: datetime,
        as_of: Optional[datetime] = None,
    ) -> Optional[int]:
        date_str = date.strftime("%Y-%m-%d")
        as_of_str = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT internal_id FROM ism_mapping
                WHERE ticker = ? AND exchange = ?
                AND start_date <= ? AND (end_date > ? OR end_date IS NULL)
                AND timestamp_knowledge <= ?
                ORDER BY timestamp_knowledge DESC LIMIT 1
            """,
                (ticker, exchange, date_str, date_str, as_of_str),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_internal_id_by_external(
        self,
        id_type: str,
        id_value: str,
        date: datetime,
        as_of: Optional[datetime] = None,
    ) -> Optional[int]:
        date_str = date.strftime("%Y-%m-%d")
        as_of_str = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT internal_id FROM external_mappings
                WHERE id_type = ? AND id_value = ?
                AND start_date <= ? AND (end_date > ? OR end_date IS NULL)
                AND timestamp_knowledge <= ?
                ORDER BY timestamp_knowledge DESC LIMIT 1
            """,
                (id_type, id_value, date_str, date_str, as_of_str),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_symbol_info(
        self,
        internal_id: int,
        date: datetime,
        as_of: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        date_str = date.strftime("%Y-%m-%d")
        as_of_str = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT ticker, exchange, sector, start_date, end_date
                FROM ism_mapping
                WHERE internal_id = ?
                AND start_date <= ? AND (end_date > ? OR end_date IS NULL)
                AND timestamp_knowledge <= ?
                ORDER BY timestamp_knowledge DESC LIMIT 1
            """,
                (internal_id, date_str, date_str, as_of_str),
            )
            result = cursor.fetchone()
            if result:
                return {
                    "ticker": result[0],
                    "exchange": result[1],
                    "sector": result[2],
                    "start_date": result[3],
                    "end_date": result[4],
                }
            return None

    def get_external_mappings(
        self,
        internal_id: int,
        date: datetime,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, str]:
        date_str = date.strftime("%Y-%m-%d")
        as_of_str = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        mappings = {}
        with sqlite3.connect(self.db_path) as conn:
            # This one is tricky because we want the latest knowledge
            # for each id_type
            cursor = conn.execute(
                """
                SELECT id_type, id_value FROM (
                    SELECT id_type, id_value, timestamp_knowledge,
                    ROW_NUMBER() OVER (
                        PARTITION BY id_type ORDER BY timestamp_knowledge DESC
                    ) as rn
                    FROM external_mappings
                    WHERE internal_id = ?
                    AND start_date <= ? AND (end_date > ? OR end_date IS NULL)
                    AND timestamp_knowledge <= ?
                ) WHERE rn = 1
            """,
                (internal_id, date_str, date_str, as_of_str),
            )
            for row in cursor.fetchall():
                mappings[row[0]] = row[1]
        return mappings

    def get_universe(
        self, date: datetime, as_of: Optional[datetime] = None
    ) -> list[int]:
        date_str = date.strftime("%Y-%m-%d")
        as_of_str = (as_of or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT internal_id FROM ism_mapping
                WHERE start_date <= ? AND (end_date > ? OR end_date IS NULL)
                AND timestamp_knowledge <= ?
            """,
                (date_str, date_str, as_of_str),
            )
            return [row[0] for row in cursor.fetchall()]
