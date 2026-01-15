import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

class VectorizedBacktester:
    def __init__(self, returns_df: pd.DataFrame, signals_df: pd.DataFrame):
        """
        returns_df and signals_df should have internal_id, timestamp, and value/signal.
        They should be indexed by (timestamp, internal_id).
        """
        self.returns = returns_df
        self.signals = signals_df

    def run(self) -> pd.Series:
        # Align signals and returns
        # Shift signals by 1 period because signal at T determines return at T+1
        df = pd.merge(self.signals, self.returns, on=['timestamp', 'internal_id'], suffixes=('_sig', '_ret'))
        df = df.sort_values(['internal_id', 'timestamp'])
        
        # Shift signal
        df['signal_delayed'] = df.groupby('internal_id')['signal'].shift(1)
        
        # Calculate daily pnl
        df['pnl'] = df['signal_delayed'] * df['returns']
        
        # Aggregate across all IDs
        daily_pnl = df.groupby('timestamp')['pnl'].sum(min_count=1)
        return daily_pnl

    @staticmethod
    def calculate_metrics(daily_pnl: pd.Series) -> Dict[str, float]:
        if daily_pnl.empty:
            return {}
        
        sharpe = np.sqrt(252) * daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() != 0 else 0
        cumulative = (1 + daily_pnl).prod() - 1
        
        # Max Drawdown
        cum_ret = (1 + daily_pnl).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            "sharpe": sharpe,
            "cumulative_return": cumulative,
            "max_drawdown": max_dd
        }
