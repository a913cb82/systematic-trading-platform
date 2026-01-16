from typing import Dict

import pandas as pd

from src.core.alpha_engine import AlphaModel


class ResidualMomentumModel(AlphaModel):
    """
    Forecasts returns based on cumulative idiosyncratic outperformance.
    Logic: Assets that have consistently outperformed their factor
    expectations tend to persist in the short term.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["residual_mom_10"]

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        return {
            int(idx): float(row["residual_mom_10"])
            for idx, row in latest.iterrows()
            if not pd.isna(row["residual_mom_10"])
        }


class ResidualReversionModel(AlphaModel):
    """
    Forecasts mean reversion in the idiosyncratic space.
    Logic: Large spikes in residual returns (normalized by volatility)
    often represent liquidity shocks that mean-revert.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["returns_residual", "residual_vol_20"]

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        signals = {}
        for idx, row in latest.iterrows():
            res = float(row["returns_residual"])
            vol = float(row["residual_vol_20"])

            if vol > 0 and not pd.isna(res):
                signals[int(idx)] = -(res / vol)
        return signals
