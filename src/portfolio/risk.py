import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from ..common.base import RiskModel

class RiskProvider(RiskModel):
    def __init__(self, storage_path: str = "data/risk"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def write_risk_model(self, date: datetime, matrix: List[List[float]], exposures: Dict[int, Dict[str, float]]) -> None:
        date_str = date.strftime("%Y%m%d")
        model_path = os.path.join(self.storage_path, date_str)
        os.makedirs(model_path, exist_ok=True)
        
        with open(os.path.join(model_path, "covariance.json"), "w") as f:
            json.dump(matrix, f)
            
        with open(os.path.join(model_path, "exposures.json"), "w") as f:
            json.dump({str(k): v for k, v in exposures.items()}, f)

    def get_covariance_matrix(self, date: datetime, internal_ids: List[int]) -> List[List[float]]:
        # This implementation assumes the matrix in storage corresponds to a specific set of IDs
        # In a real system, we'd need to map these correctly.
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(self.storage_path, date_str, "covariance.json")
        
        if not os.path.exists(file_path):
            # Fallback or error
            n = len(internal_ids)
            return np.eye(n).tolist()
            
        with open(file_path, "r") as f:
            matrix = json.load(f)
        return matrix

    def get_factor_exposures(self, date: datetime, internal_ids: List[int]) -> Dict[int, Dict[str, float]]:
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(self.storage_path, date_str, "exposures.json")
        
        if not os.path.exists(file_path):
            return {iid: {} for iid in internal_ids}
            
        with open(file_path, "r") as f:
            exposures = json.load(f)
        return {int(k): v for k, v in exposures.items()}

class RollingWindowRiskModel(RiskModel):
    """
    Calculates covariance from historical market data.
    """
    def __init__(self, market_data_engine, window_days: int = 60):
        self.market_data_engine = market_data_engine
        self.window_days = window_days

    def get_covariance_matrix(self, date: datetime, internal_ids: List[int]) -> List[List[float]]:
        if not internal_ids:
            return []
            
        start_date = date - pd.Timedelta(days=self.window_days * 1.5) # Extra buffer for non-trading days
        bars = self.market_data_engine.get_bars(internal_ids, start_date, date)
        
        if not bars:
            return np.eye(len(internal_ids)).tolist()
            
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Pivot to get returns
        pivot_df = df.pivot(index='timestamp', columns='internal_id', values='close')
        returns_df = pivot_df.pct_change().dropna()
        
        if returns_df.empty:
            return np.eye(len(internal_ids)).tolist()
            
        # Ensure all internal_ids are present
        for iid in internal_ids:
            if iid not in returns_df.columns:
                returns_df[iid] = 0.0
        
        returns_df = returns_df[internal_ids] # Reorder
        cov_matrix = returns_df.cov().values
        
        # Fill NaNs with 0
        cov_matrix = np.nan_to_num(cov_matrix)
        
        return cov_matrix.tolist()

    def get_factor_exposures(self, date: datetime, internal_ids: List[int]) -> Dict[int, Dict[str, float]]:
        # Simple risk model might not have factors
        return {iid: {} for iid in internal_ids}
