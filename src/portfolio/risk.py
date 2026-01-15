import json
import os
from datetime import datetime
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..common.base import RiskModel


class FundamentalRiskModel(RiskModel):
    """
    Fundamental risk model using sectors as factors.
    """

    def __init__(self, market_data_engine: Any, ism: Any) -> None:
        self.market_data_engine = market_data_engine
        self.ism = ism

    def get_covariance_matrix(
        self, date: datetime, internal_ids: List[int]
    ) -> List[List[float]]:
        if not internal_ids:
            return []

        exposures = self.get_factor_exposures(date, internal_ids)
        # Convert exposures to a matrix B (n_assets, n_factors)
        # Flatten the list of keys
        unique_sectors = sorted(
            list(set(s for exp in exposures.values() for s in exp.keys()))
        )

        n_assets = len(internal_ids)
        n_factors = len(unique_sectors)

        b_matrix = np.zeros((n_assets, n_factors))
        for i, iid in enumerate(internal_ids):
            asset_exp = exposures[iid]
            for sector, val in asset_exp.items():
                if sector in unique_sectors:
                    j = unique_sectors.index(sector)
                    b_matrix[i, j] = val

        # Simplified: assume factor covariance is identity
        # (uncorrelated sectors) and specific risk is some constant.
        # In reality, these should be estimated.
        factor_cov = np.eye(n_factors) * 0.01
        specific_risk = np.eye(n_assets) * 0.05

        total_cov = b_matrix @ factor_cov @ b_matrix.T + specific_risk

        return cast(List[List[float]], total_cov.tolist())

    def get_factor_exposures(
        self, date: datetime, internal_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        exposures: Dict[int, Dict[str, float]] = {}
        for iid in internal_ids:
            info = self.ism.get_symbol_info(iid, date)
            sector = info.get("sector", "Unknown") if info else "Unknown"
            exposures[iid] = {sector: 1.0}
        return exposures


class PCARiskModel(RiskModel):
    """
    Statistical risk model using PCA.
    """

    def __init__(
        self,
        market_data_engine: Any,
        window_days: int = 252,
        n_factors: int = 5,
    ) -> None:
        self.market_data_engine = market_data_engine
        self.window_days = window_days
        self.n_factors = n_factors

    def get_covariance_matrix(
        self, date: datetime, internal_ids: List[int]
    ) -> List[List[float]]:
        if not internal_ids:
            return []

        start_date = date - pd.Timedelta(days=self.window_days * 1.5)
        bars = self.market_data_engine.get_bars(internal_ids, start_date, date)

        if not bars:
            return cast(List[List[float]], np.eye(len(internal_ids)).tolist())

        df = pd.DataFrame(bars)
        pivot_df = df.pivot(
            index="timestamp", columns="internal_id", values="close"
        )
        returns_df = pivot_df.pct_change().dropna()

        if returns_df.empty or len(returns_df) < self.n_factors:
            return cast(List[List[float]], np.eye(len(internal_ids)).tolist())

        # Reorder and handle missing values
        returns_df = returns_df.reindex(columns=internal_ids).fillna(0)

        # PCA
        n_comp = min(self.n_factors, returns_df.shape[1], returns_df.shape[0])
        pca = PCA(n_components=n_comp)
        pca.fit(returns_df)

        # Factor loadings (components)
        loadings = pca.components_  # (n_comp, n_assets)
        # Factor covariance (variance of principal components)
        factor_cov = np.diag(pca.explained_variance_)

        # Systematic covariance
        systematic_cov = loadings.T @ factor_cov @ loadings

        # Idiosyncratic risk (specific risk)
        residuals = returns_df - (returns_df @ loadings.T @ loadings)
        specific_variances = np.diag(np.var(residuals, axis=0))

        # Total covariance
        total_cov = systematic_cov + specific_variances

        return cast(List[List[float]], total_cov.tolist())

    def get_factor_exposures(
        self, date: datetime, internal_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        # For statistical PCA, factors are abstract
        return {iid: {} for iid in internal_ids}


class RiskProvider(RiskModel):
    def __init__(self, storage_path: str = "data/risk") -> None:
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def write_risk_model(
        self,
        date: datetime,
        matrix: List[List[float]],
        exposures: Dict[int, Dict[str, float]],
    ) -> None:
        date_str = date.strftime("%Y%m%d")
        model_path = os.path.join(self.storage_path, date_str)
        os.makedirs(model_path, exist_ok=True)

        with open(os.path.join(model_path, "covariance.json"), "w") as f:
            json.dump(matrix, f)

        with open(os.path.join(model_path, "exposures.json"), "w") as f:
            json.dump({str(k): v for k, v in exposures.items()}, f)

    def get_covariance_matrix(
        self, date: datetime, internal_ids: List[int]
    ) -> List[List[float]]:
        # This implementation assumes the matrix in storage corresponds to
        # a specific set of IDs. In a real system, we'd need to map these
        # correctly.
        date_str = date.strftime("%Y%m%d")
        file_path = os.path.join(
            self.storage_path, date_str, "covariance.json"
        )

        if not os.path.exists(file_path):
            # Fallback or error
            n = len(internal_ids)
            return cast(List[List[float]], np.eye(n).tolist())

        with open(file_path, "r") as f:
            matrix = json.load(f)
        return cast(List[List[float]], matrix)

    def get_factor_exposures(
        self, date: datetime, internal_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
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

    def __init__(self, market_data_engine: Any, window_days: int = 60) -> None:
        self.market_data_engine = market_data_engine
        self.window_days = window_days

    def get_covariance_matrix(
        self, date: datetime, internal_ids: List[int]
    ) -> List[List[float]]:
        if not internal_ids:
            return []

        start_date = date - pd.Timedelta(
            days=self.window_days * 1.5
        )  # Extra buffer for non-trading days
        bars = self.market_data_engine.get_bars(internal_ids, start_date, date)

        if not bars:
            return cast(List[List[float]], np.eye(len(internal_ids)).tolist())

        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Pivot to get returns
        pivot_df = df.pivot(
            index="timestamp", columns="internal_id", values="close"
        )
        returns_df = pivot_df.pct_change().dropna()

        if returns_df.empty:
            return cast(List[List[float]], np.eye(len(internal_ids)).tolist())

        # Ensure all internal_ids are present
        for iid in internal_ids:
            if iid not in returns_df.columns:
                returns_df[iid] = 0.0

        returns_df = returns_df[internal_ids]  # Reorder
        cov_matrix = returns_df.cov().values

        # Fill NaNs with 0
        cov_matrix = np.nan_to_num(cov_matrix)

        return cast(List[List[float]], cov_matrix.tolist())

    def get_factor_exposures(
        self, date: datetime, internal_ids: List[int]
    ) -> Dict[int, Dict[str, float]]:
        # Simple risk model might not have factors
        return {iid: {} for iid in internal_ids}
