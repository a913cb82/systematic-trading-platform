import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.alpha.backtest_vectorized import VectorizedBacktester

class TestVectorizedBacktester(unittest.TestCase):
    def test_backtest_run(self):
        timestamps = [datetime(2023, 1, i) for i in range(1, 6)]
        
        # Returns: 1% every day for ID 1
        rets_data = []
        for ts in timestamps:
            rets_data.append({'timestamp': ts, 'internal_id': 1, 'returns': 0.01})
        returns_df = pd.DataFrame(rets_data)
        
        # Signals: 1.0 every day for ID 1
        sigs_data = []
        for ts in timestamps:
            sigs_data.append({'timestamp': ts, 'internal_id': 1, 'signal': 1.0})
        signals_df = pd.DataFrame(sigs_data)
        
        bt = VectorizedBacktester(returns_df, signals_df)
        pnl = bt.run()
        print(f"\nPNL:\n{pnl}")
        
        # Signal from day 1 applies to return of day 2
        # Day 1: NaN
        # Day 2: 0.01 * 1.0 = 0.01
        # Day 3: 0.01
        # Day 4: 0.01
        # Day 5: 0.01
        self.assertEqual(len(pnl.dropna()), 4)
        self.assertAlmostEqual(pnl.dropna().iloc[0], 0.01)

        metrics = bt.calculate_metrics(pnl.dropna())
        self.assertGreater(metrics['cumulative_return'], 0.04) # (1.01^4)-1

if __name__ == "__main__":
    unittest.main()
