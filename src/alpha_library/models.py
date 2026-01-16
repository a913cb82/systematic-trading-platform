from typing import Dict

import pandas as pd

from src.core.alpha_engine import AlphaModel


class MomentumModel(AlphaModel):
    def compute_signals(
        self, latest: pd.DataFrame, history: pd.DataFrame
    ) -> Dict[int, float]:
        return {
            iid: (float(row["rsi_14"]) - 50.0) / 100.0
            for iid, row in latest.iterrows()
        }


class ValueModel(AlphaModel):
    def compute_signals(
        self, latest: pd.DataFrame, history: pd.DataFrame
    ) -> Dict[int, float]:
        signals = {}
        for iid, row in latest.iterrows():
            sma = float(row["sma_10"])
            close = float(row["close"])
            signals[iid] = -(close - sma) / sma if sma != 0 else 0.0
        return signals
