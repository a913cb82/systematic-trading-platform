import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from src.common.types import Bar, CorporateAction
from src.data.corporate_actions import CorporateActionMaster
from src.data.market_data import MarketDataEngine


class TestPriceAdjustment(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.market_path = os.path.join(self.tmp_dir, "market")
        self.ca_db = os.path.join(self.tmp_dir, "ca.db")

        self.ca_master = CorporateActionMaster(self.ca_db)
        self.mde = MarketDataEngine(self.market_path, ca_master=self.ca_master)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_split_adjustment(self) -> None:
        # 1. Write raw bars (Price 100 before, 50 after a 2-for-1 split)
        # We simulate that the data was ingested RAW.
        # Day 1: 100
        # Day 2: 100
        # --- 2-for-1 Split ---
        # Day 3: 50
        # Day 4: 50

        internal_id = 1
        start_date = datetime(2023, 1, 1)
        bars = []
        split_day_index = 2
        for i in range(4):
            ts = start_date + timedelta(days=i)
            price = 100.0 if i < split_day_index else 50.0
            bars.append(
                Bar(
                    internal_id=internal_id,
                    timestamp=ts,
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price,
                    volume=1000.0,
                )
            )
        self.mde.write_bars(bars)

        # 2. Register Split (2-for-1 on Day 3)
        self.ca_master.write_actions(
            [
                CorporateAction(
                    internal_id=internal_id,
                    type="SPLIT",
                    ex_date=datetime(2023, 1, 3),
                    ratio=2.0,
                    timestamp_knowledge=datetime.now(),
                )
            ]
        )

        # 3. Query RAW
        raw_bars = self.mde.get_bars(
            [internal_id],
            start_date,
            start_date + timedelta(days=3),
            adjustment="RAW",
        )
        self.assertEqual(raw_bars[0]["close"], 100.0)
        self.assertEqual(raw_bars[2]["close"], 50.0)

        # 4. Query RATIO (Backward adjusted)
        # Day 1 and 2 should be divided by 2 -> 50.0
        # Day 3 and 4 remain 50.0
        adj_bars = self.mde.get_bars(
            [internal_id],
            start_date,
            start_date + timedelta(days=3),
            adjustment="RATIO",
        )
        self.assertEqual(adj_bars[0]["close"], 50.0)
        self.assertEqual(adj_bars[1]["close"], 50.0)
        self.assertEqual(adj_bars[2]["close"], 50.0)


if __name__ == "__main__":
    unittest.main()
