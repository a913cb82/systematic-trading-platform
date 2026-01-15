from datetime import datetime, timedelta
from typing import Dict, List
from ..common.base import AlphaModel
from .features import FeatureStore

class MeanReversionModel(AlphaModel):
    def __init__(self, feature_store: FeatureStore, internal_ids: List[int]):
        self.feature_store = feature_store
        self.internal_ids = internal_ids

    def generate_forecasts(self, timestamp: datetime) -> Dict[int, float]:
        # Simple mean reversion: if 1d return is positive, forecast negative (and vice versa)
        # We need data up to timestamp
        start = timestamp - timedelta(days=5)
        df = self.feature_store.calculate_cycle_feature(self.internal_ids, start, timestamp, 'returns_1d')
        
        if df.empty:
            return {}
            
        # Get latest return for each id
        latest_returns = df.sort_values('timestamp').groupby('internal_id').last()
        
        forecasts = {}
        for internal_id, row in latest_returns.iterrows():
            ret = row['feature']
            if not isinstance(ret, (int, float)) or float('-inf') == ret or float('inf') == ret:
                forecasts[internal_id] = 0.0
            else:
                forecasts[internal_id] = -ret # Negative of return
                
        return forecasts
