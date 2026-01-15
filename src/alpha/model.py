from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.base import AlphaModel, MarketDataProvider, RiskModel
from .publisher import ForecastPublisher


class MeanReversionModel(AlphaModel):
    def __init__(
        self,
        market_data: MarketDataProvider,
        internal_ids: List[int],
        risk_model: Optional[RiskModel] = None,
        publisher: Optional[ForecastPublisher] = None,
    ):
        self.market_data = market_data
        self.internal_ids = internal_ids
        self.risk_model = risk_model
        self.publisher = publisher

    def generate_forecasts(self, timestamp: datetime) -> Dict[int, float]:
        # Simple mean reversion: if residual return is positive,
        # forecast negative (and vice versa)

        # 1. Get returns for the last day
        start = timestamp - timedelta(days=5)
        returns_df = self.market_data.get_returns(
            self.internal_ids, start, timestamp, type="RAW"
        )

        if returns_df.empty:
            return {}

        latest_returns = returns_df.iloc[-1]

        # 2. Calculate residuals if risk model exists
        if self.risk_model:
            # This is a simplified residual calculation:
            # ret_resid = ret_raw - (beta * factor_return)
            # For this PoC, we'll just use the returns as they are
            # but label them as targeting residuals.
            # In a real system, we'd fetch factor returns from the Data Layer.
            residuals = latest_returns
        else:
            residuals = latest_returns

        forecasts = {}
        for iid in self.internal_ids:
            if iid in residuals.index:
                ret = residuals[iid]
                if pd.isna(ret) or np.isinf(ret):
                    forecasts[iid] = 0.0
                else:
                    forecasts[iid] = -float(ret)  # Negative of return
            else:
                forecasts[iid] = 0.0

        if self.publisher:
            self.publisher.submit_forecasts(timestamp, forecasts)

        return forecasts
