import unittest
from datetime import datetime

from src.core.data_platform import Bar, DataPlatform, Event, QueryConfig


class TestArcticPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = "./.arctic_test_db"
        # Ensure clean start for tests
        DataPlatform(db_path=self.db_path, clear=True)

    def test_data_persistence_across_instances(self) -> None:
        """
        Verify that data written by one instance is readable by another.
        """
        ts = datetime(2025, 1, 1, 12, 0)

        # Instance 1: Write data
        dp1 = DataPlatform(db_path=self.db_path)
        iid = dp1.get_internal_id("AAPL")
        dp1.add_bars([Bar(iid, ts, 100.0, 101.0, 99.0, 100.0, 1000.0)])

        # Instance 2: Read data
        dp2 = DataPlatform(db_path=self.db_path)
        df = dp2.get_bars([iid], QueryConfig(start=ts, end=ts))

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["close_1D"], 100.0)
        self.assertEqual(dp2.reverse_ism[iid], "AAPL")

    def test_json_serialization_robustness(self) -> None:
        """
        Test that various types in 'extra' and 'value' are
        correctly serialized.
        """
        dp = DataPlatform(db_path=self.db_path, clear=True)

        # Nested dict in register_security
        extra_data = {"key": "val", "nested": {"a": 1}}
        iid = dp.register_security("MSFT", None, None, None, **extra_data)

        # Complex types in Events
        ts = datetime(2025, 1, 1, 12, 0)
        complex_val = [1, 2, {"three": 3}]
        dp.add_events([Event(iid, ts, "COMPLEX", complex_val)])

        # Reload and verify
        dp2 = DataPlatform(db_path=self.db_path)

        # Verify Security Master extra
        sec_info = dp2.sec_df[dp2.sec_df["internal_id"] == iid].iloc[0][
            "extra"
        ]
        self.assertEqual(sec_info["nested"]["a"], 1)

        # Verify Event value
        events = dp2.get_events([iid])
        self.assertEqual(events[0].value[2]["three"], 3)


if __name__ == "__main__":
    unittest.main()
