import threading
import time
import unittest
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from src.core.alpha_engine import AlphaModel, alpha_context
from src.core.data_platform import DataPlatform


class MockModel(AlphaModel):
    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, Any]:
        # Return the context as of time so we can check it

        return {0: self.context_as_of}


class TestAlphaConcurrency(unittest.TestCase):
    def test_context_isolation(self) -> None:
        data = DataPlatform()

        ts1 = datetime(2025, 1, 1)

        ts2 = datetime(2025, 2, 2)

        model = MockModel()

        results: Dict[str, Any] = {}

        def run_model(ts: datetime, key: str) -> None:
            with alpha_context(data, ts):
                # Simulate some work

                time.sleep(0.1)

                results[key] = model.compute_signals(pd.DataFrame())[0]

        t1 = threading.Thread(target=run_model, args=(ts1, "one"))
        t2 = threading.Thread(target=run_model, args=(ts2, "two"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(results["one"], ts1)
        self.assertEqual(results["two"], ts2)


if __name__ == "__main__":
    unittest.main()
