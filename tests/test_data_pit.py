import unittest
from datetime import datetime, timedelta

from src.core.data_platform import DataPlatform


class TestDataPIT(unittest.TestCase):
    def setUp(self) -> None:
        self.data = DataPlatform(clear=True)

    def test_ticker_history_ism(self) -> None:
        """Tests that same internal ID can have different tickers over time."""
        t1 = datetime(2020, 1, 1)
        t2 = datetime(2022, 6, 9)

        # Register FB
        iid = self.data.register_security(
            "FB", start=t1, end=t2 - timedelta(days=1)
        )
        # Register META with same ID
        self.data.register_security("META", internal_id=iid, start=t2)

        # Test lookups
        self.assertEqual(
            self.data.get_internal_id("FB", date=datetime(2021, 1, 1)), iid
        )
        self.assertEqual(
            self.data.get_internal_id("META", date=datetime(2023, 1, 1)), iid
        )

        # FB should not be found in 2023
        # Actually, get_internal_id registers if not found.
        # But for FB in 2023, it should return a NEW ID if we don't handle it.
        # Our current implementation: if empty, it registers.
        new_iid = self.data.get_internal_id("FB", date=datetime(2023, 1, 1))
        self.assertNotEqual(new_iid, iid)

    def test_pit_universe(self) -> None:
        """Tests that universe reconstruction works for historical dates."""
        t1 = datetime(2020, 1, 1)
        t2 = datetime(2021, 1, 1)

        id1 = self.data.register_security("AAPL", start=t1)
        id2 = self.data.register_security("TSLA", start=t2)

        # At t1 + 1 month, only AAPL
        univ_t1 = self.data.get_universe(t1 + timedelta(days=30))
        self.assertIn(id1, univ_t1)
        self.assertNotIn(id2, univ_t1)

        # At t2 + 1 month, both
        univ_t2 = self.data.get_universe(t2 + timedelta(days=30))
        self.assertIn(id1, univ_t2)
        self.assertIn(id2, univ_t2)

    def test_extra_identifiers(self) -> None:
        """Tests storing institutional identifiers."""
        iid = self.data.register_security("AAPL", cusip="12345", isin="US123")
        # Extra is stored in the dataframe
        info = self.data.sec_df[self.data.sec_df["internal_id"] == iid].iloc[
            0
        ]["extra"]
        self.assertEqual(info["cusip"], "12345")
        self.assertEqual(info["isin"], "US123")


if __name__ == "__main__":
    unittest.main()
