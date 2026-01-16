from typing import Any, Tuple, cast

import numpy as np


class RiskModel:
    @staticmethod
    def _fit_pca(
        returns: np.ndarray, n_factors: int
    ) -> Tuple[Any, Any, np.ndarray, np.ndarray]:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        z = scaler.fit_transform(returns)

        # n_components must be <= min(n_samples, n_features)
        k = min(n_factors, z.shape[0], z.shape[1])
        pca = PCA(n_components=k)
        z_pca = pca.fit_transform(z)
        return pca, scaler, z, z_pca

    @staticmethod
    def estimate_pca_covariance(
        returns: np.ndarray, n_factors: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposes returns into factor and specific risk using PCA.
        Returns: (Sigma, Loadings)
        """
        pca, scaler, z, _ = RiskModel._fit_pca(returns, n_factors)

        loadings = pca.components_.T
        f_cov = loadings @ np.diag(pca.explained_variance_) @ loadings.T

        spec_var = np.maximum(1.0 - np.diag(f_cov), 0)
        sigma_z = f_cov + np.diag(spec_var)

        std = scaler.scale_
        sigma = sigma_z * np.outer(std, std)

        return cast(np.ndarray, sigma), loadings

    @staticmethod
    def get_residual_returns(
        returns: np.ndarray, n_factors: int = 3
    ) -> np.ndarray:
        """
        Removes components explained by top K PCA factors.
        """
        pca, scaler, z, z_pca = RiskModel._fit_pca(returns, n_factors)

        z_explained = pca.inverse_transform(z_pca)
        z_residual = z - z_explained

        return cast(np.ndarray, z_residual * scaler.scale_)
