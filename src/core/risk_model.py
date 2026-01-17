from typing import TYPE_CHECKING, Tuple, cast

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    pass


class RiskModel:
    @staticmethod
    def _fit_pca(
        returns: npt.NDArray[np.float64], n_factors: int
    ) -> Tuple[
        PCA,
        StandardScaler,
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        # Handle zero variance columns to avoid division by zero warnings
        std = np.std(returns, axis=0)
        clean_returns = returns.copy()
        if np.any(std == 0):
            # Add tiny noise to constant columns to allow scaling
            clean_returns[:, std == 0] += np.random.normal(
                0, 1e-10, clean_returns[:, std == 0].shape
            )

        scaler = StandardScaler()
        z = scaler.fit_transform(clean_returns)

        # n_components must be <= min(n_samples, n_features)
        k = min(n_factors, z.shape[0], z.shape[1])
        pca = PCA(n_components=k)
        z_pca = pca.fit_transform(z)
        return pca, scaler, z, z_pca

    @staticmethod
    def estimate_pca_covariance(
        returns: npt.NDArray[np.float64], n_factors: int = 3
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

        return cast(npt.NDArray[np.float64], sigma), loadings

    @staticmethod
    def get_residual_returns(
        returns: npt.NDArray[np.float64], n_factors: int = 3
    ) -> npt.NDArray[np.float64]:
        """
        Removes components explained by top K PCA factors.
        """
        pca, scaler, z, z_pca = RiskModel._fit_pca(returns, n_factors)

        z_explained = pca.inverse_transform(z_pca)
        z_residual = z - z_explained

        return cast(npt.NDArray[np.float64], z_residual * scaler.scale_)

    @staticmethod
    def get_factor_returns(
        returns: npt.NDArray[np.float64], n_factors: int = 3
    ) -> npt.NDArray[np.float64]:
        """
        Calculates the realized returns for each PCA factor.
        Returns: Factor Returns matrix (T_samples, K_factors)
        """
        pca, _, _, z_pca = RiskModel._fit_pca(returns, n_factors)
        return cast(npt.NDArray[np.float64], z_pca)
