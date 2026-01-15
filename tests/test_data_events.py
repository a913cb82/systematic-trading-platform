import unittest
import os
import shutil
from datetime import datetime
from src.data.event_store import EventStore
from src.common.types import Event


class TestEventStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_events"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.store = EventStore(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_write_and_read_events(self):
        events = [
            Event(
                internal_id=1,
                type="EARNINGS",
                value={"eps": 1.25},
                timestamp_event=datetime(2023, 1, 15),
                timestamp_knowledge=datetime(2023, 1, 15),
            ),
            Event(
                internal_id=1,
                type="NEWS",
                value="Positive sentiment",
                timestamp_event=datetime(2023, 1, 16),
                timestamp_knowledge=datetime(2023, 1, 16),
            ),
        ]

        self.store.write_events(events)

        read_events = self.store.get_events(
            ["EARNINGS"], [1], datetime(2023, 1, 1), datetime(2023, 1, 31)
        )
        self.assertEqual(len(read_events), 1)
        self.assertEqual(read_events[0]["type"], "EARNINGS")
        self.assertEqual(read_events[0]["value"]["eps"], 1.25)

    def test_bitemporal_events(self):
        # Initial news event
        ev1 = Event(
            internal_id=1,
            type="NEWS",
            value="Initial report",
            timestamp_event=datetime(2023, 1, 1),
            timestamp_knowledge=datetime(2023, 1, 1, 10),
        )
        self.store.write_events([ev1])

        # Correction known later
        ev2 = Event(
            internal_id=1,
            type="NEWS",
            value="Corrected report",
            timestamp_event=datetime(2023, 1, 1),
            timestamp_knowledge=datetime(2023, 1, 1, 12),
        )
        self.store.write_events([ev2])

        # Query as of 11:00
        events_11 = self.store.get_events(
            ["NEWS"],
            [1],
            datetime(2023, 1, 1),
            datetime(2023, 1, 1),
            as_of=datetime(2023, 1, 1, 11),
        )
        self.assertEqual(events_11[0]["value"], "Initial report")

        # Query as of 13:00
        events_13 = self.store.get_events(
            ["NEWS"],
            [1],
            datetime(2023, 1, 1),
            datetime(2023, 1, 1),
            as_of=datetime(2023, 1, 1, 13),
        )
        self.assertEqual(events_13[0]["value"], "Corrected report")


if __name__ == "__main__":
    unittest.main()
