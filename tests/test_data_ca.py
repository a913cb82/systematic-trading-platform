import os
import unittest
from datetime import datetime

from src.common.types import CorporateAction
from src.data.corporate_actions import CorporateActionMaster


class TestCorporateActionMaster(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "test_ca.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.master = CorporateActionMaster(self.db_path)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_write_and_read_actions(self) -> None:
        actions = [
            CorporateAction(
                internal_id=1,
                type="SPLIT",
                ex_date=datetime(2023, 6, 1),
                record_date=datetime(2023, 5, 31),
                pay_date=datetime(2023, 6, 1),
                ratio=2.0,
                timestamp_knowledge=datetime(2023, 5, 20),
            ),
            CorporateAction(
                internal_id=2,
                type="DIVIDEND",
                ex_date=datetime(2023, 6, 15),
                record_date=datetime(2023, 6, 14),
                pay_date=datetime(2023, 6, 30),
                ratio=0.5,
                timestamp_knowledge=datetime(2023, 6, 1),
            ),
        ]

        self.master.write_actions(actions)

        read_actions = self.master.get_actions(
            [1, 2], datetime(2023, 1, 1), datetime(2023, 12, 31)
        )
        self.assertEqual(len(read_actions), 2)

        # Test bitemporal correction
        correction = CorporateAction(
            internal_id=1,
            type="SPLIT",
            ex_date=datetime(2023, 6, 1),
            record_date=datetime(2023, 5, 31),
            pay_date=datetime(2023, 6, 1),
            ratio=3.0,  # Corrected ratio
            timestamp_knowledge=datetime(2023, 5, 25),
        )
        self.master.write_actions([correction])

        # As of 5/22, should see ratio 2.0
        actions_522 = self.master.get_actions(
            [1],
            datetime(2023, 6, 1),
            datetime(2023, 6, 1),
            as_of=datetime(2023, 5, 22),
        )
        self.assertEqual(actions_522[0]["ratio"], 2.0)

        # As of 5/26, should see ratio 3.0
        actions_526 = self.master.get_actions(
            [1],
            datetime(2023, 6, 1),
            datetime(2023, 6, 1),
            as_of=datetime(2023, 5, 26),
        )
        self.assertEqual(actions_526[0]["ratio"], 3.0)


if __name__ == "__main__":
    unittest.main()
