from typing import Dict

import pandas as pd

from src.core.alpha_engine import AlphaModel


class MomentumModel(AlphaModel):
    """
    Forecasts returns based on cumulative idiosyncratic outperformance.
    Logic: Assets that have consistently outperformed their factor
    expectations tend to persist in the short term.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = ["residual_mom_10_30min"]

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        return {
            int(idx): float(row["residual_mom_10_30min"])
            for idx, row in latest.iterrows()
            if not pd.isna(row["residual_mom_10_30min"])
        }


class ReversionModel(AlphaModel):
    """
    Forecasts mean reversion in the idiosyncratic space.
    Logic: Large spikes in residual returns (normalized by volatility)
    often represent liquidity shocks that mean-revert.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = [
            "returns_residual_30min",
            "residual_vol_20_30min",
        ]

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        signals = {}
        for idx, row in latest.iterrows():
            res = float(row["returns_residual_30min"])
            vol = float(row["residual_vol_20_30min"])

            if vol > 0 and not pd.isna(res):
                signals[int(idx)] = -(res / vol)
        return signals


class EarningsModel(AlphaModel):
    """
    Goes long if a positive earnings surprise happened in the last 24h.
    Signal decays linearly to zero over the 24h window.
    """

    def compute_signals(self, latest: pd.DataFrame) -> Dict[int, float]:
        signals = {int(iid): 0.0 for iid in latest.index}

        now = self.context_as_of

        if now is None:
            return signals

        # Query events from the last 24h
        events = self.get_events(
            list(latest.index),
            types=["EARNINGS_RELEASE"],
            start=now - pd.Timedelta(hours=24),
        )

        for ev in events:
            surprise = ev.value.get("surprise_pct", 0)
            if surprise > 0:
                # Linear decay over 24 hours
                hours_since = (now - ev.timestamp).total_seconds() / 3600
                decay = max(0.0, 1.0 - (hours_since / 24.0))
                signals[ev.internal_id] = 0.5 * decay

        return signals
