import os
import unittest
from datetime import datetime

from src.data.ism import InternalSecurityMaster
from src.data.symbology import SymbologyService


class TestSymbologyService(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_symbology.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.ism = InternalSecurityMaster(self.db_path)
        self.service = SymbologyService(self.ism)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_multi_id_mapping(self):
        # Register security
        start_date = datetime(2020, 1, 1)
        internal_id = self.ism.register_security("AAPL", "NASDAQ", start_date)

        # Add external identifiers
        self.ism.add_external_mapping(
            internal_id, "FIGI", "BBG000B9XRY4", start_date
        )
        self.ism.add_external_mapping(
            internal_id, "ISIN", "US0378331005", start_date
        )

        # Test mapping from FIGI
        id_figi = self.service.map_to_internal_id(
            "FIGI", "BBG000B9XRY4", datetime(2020, 6, 1)
        )
        self.assertEqual(id_figi, internal_id)

        # Test mapping from ISIN
        id_isin = self.service.map_to_internal_id(
            "ISIN", "US0378331005", datetime(2020, 6, 1)
        )
        self.assertEqual(id_isin, internal_id)

        # Test get all identifiers
        all_ids = self.service.get_all_identifiers(
            internal_id, datetime(2020, 6, 1)
        )
        self.assertEqual(all_ids["TICKER"], "AAPL")
        self.assertEqual(all_ids["FIGI"], "BBG000B9XRY4")
        self.assertEqual(all_ids["ISIN"], "US0378331005")

    def test_ticker_lookup(self):
        start_date = datetime(2020, 1, 1)
        internal_id = self.ism.register_security("TSLA", "NASDAQ", start_date)

        id_ticker = self.service.map_to_internal_id(
            "TICKER", "TSLA", datetime(2020, 6, 1), exchange="NASDAQ"
        )
        self.assertEqual(id_ticker, internal_id)

        with self.assertRaises(ValueError):
            self.service.map_to_internal_id(
                "TICKER", "TSLA", datetime(2020, 6, 1)
            )


if __name__ == "__main__":
    unittest.main()
