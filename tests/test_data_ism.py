import unittest
import os
from datetime import datetime
from src.data.ism import InternalSecurityMaster


class TestInternalSecurityMaster(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_ism.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.ism = InternalSecurityMaster(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_registration_and_lookup(self):
        # Register a new security
        start_date = datetime(2020, 1, 1)
        internal_id = self.ism.register_security("AAPL", "NASDAQ", start_date)
        self.assertGreaterEqual(internal_id, 1001)

        # Lookup by ticker
        found_id = self.ism.get_internal_id(
            "AAPL", "NASDAQ", datetime(2020, 6, 1)
        )
        self.assertEqual(internal_id, found_id)

        # Lookup by ID
        info = self.ism.get_symbol_info(internal_id, datetime(2020, 6, 1))
        self.assertEqual(info["ticker"], "AAPL")
        self.assertEqual(info["exchange"], "NASDAQ")

    def test_ticker_change(self):
        # Initial registration
        internal_id = self.ism.register_security(
            "FB", "NASDAQ", datetime(2012, 5, 18)
        )

        # Ticker change
        self.ism.register_security(
            "META", "NASDAQ", datetime(2022, 6, 9), internal_id=internal_id
        )

        # Test old ticker lookup
        id_old = self.ism.get_internal_id("FB", "NASDAQ", datetime(2020, 1, 1))
        self.assertEqual(id_old, internal_id)

        # Test new ticker lookup
        id_new = self.ism.get_internal_id(
            "META", "NASDAQ", datetime(2023, 1, 1)
        )
        self.assertEqual(id_new, internal_id)

        # Test old ticker after change (should be None or handled)
        id_old_after = self.ism.get_internal_id(
            "FB", "NASDAQ", datetime(2023, 1, 1)
        )
        self.assertIsNone(id_old_after)

    def test_invalid_lookup(self):
        found_id = self.ism.get_internal_id(
            "INVALID", "NONE", datetime(2020, 1, 1)
        )
        self.assertIsNone(found_id)

    def test_get_universe(self):
        id1 = self.ism.register_security(
            "AAPL", "NASDAQ", datetime(2020, 1, 1)
        )
        id2 = self.ism.register_security(
            "TSLA", "NASDAQ", datetime(2020, 1, 1)
        )

        # At 2020-06-01, both should be in universe
        univ_2020 = self.ism.get_universe(datetime(2020, 6, 1))
        self.assertIn(id1, univ_2020)
        self.assertIn(id2, univ_2020)
        self.assertEqual(len(univ_2020), 2)

        # Delist id2 at 2021-01-01
        self.ism.delist_security(id2, datetime(2021, 1, 1))

        # At 2021-06-01, only id1 should be in universe
        univ_2021 = self.ism.get_universe(datetime(2021, 6, 1))
        self.assertIn(id1, univ_2021)
        self.assertNotIn(id2, univ_2021)
        self.assertEqual(len(univ_2021), 1)


if __name__ == "__main__":
    unittest.main()
