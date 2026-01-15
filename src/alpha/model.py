from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.base import BaseAlphaModel, MarketDataProvider, RiskModel
from ..common.types import ModelState
from .publisher import ForecastPublisher


class MeanReversionModel(BaseAlphaModel):
    def __init__(
        self,
        market_data: MarketDataProvider,
        internal_ids: List[int],
        risk_model: Optional[RiskModel] = None,
        publisher: Optional[ForecastPublisher] = None,
    ) -> None:
        super().__init__(market_data, internal_ids, risk_model)
        self.publisher = publisher

    def on_cycle(
        self, timestamp: datetime, state: ModelState
    ) -> Dict[int, float]:
        """
        Simplified API: implementation of mean reversion.
        """
        # 1. Use residual returns if available, else raw
        returns_df = state["residuals"]
        if returns_df.empty:
            return {}

        latest_returns = returns_df.iloc[-1]

        forecasts = {}
        for iid in self.internal_ids:
            if iid in latest_returns.index:
                ret = latest_returns[iid]
                if pd.isna(ret) or np.isinf(ret):
                    forecasts[iid] = 0.0
                else:
                    forecasts[iid] = -float(ret)  # Mean Reversion
            else:
                forecasts[iid] = 0.0

        if self.publisher:
            self.publisher.submit_forecasts(timestamp, forecasts)

        return forecasts
