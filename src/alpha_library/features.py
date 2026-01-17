import numpy as np
import pandas as pd

from src.core.alpha_engine import feature, multi_tf_feature
from src.core.risk_model import RiskModel
from src.core.types import Timeframe


@multi_tf_feature(
    name="returns_raw", timeframes=[Timeframe.DAY, Timeframe.MIN_30]
)
def returns_raw(df: pd.DataFrame, tf: Timeframe) -> pd.Series:
    col = f"close_{tf.value}"
    return df.groupby("internal_id")[col].pct_change()


@multi_tf_feature(
    name="returns_residual",
    timeframes=[Timeframe.MIN_30],
    dependencies=["returns_raw"],
)
def returns_residual(df: pd.DataFrame, tf: Timeframe) -> pd.Series:
    raw_col = f"returns_raw_{tf.value}"
    if raw_col not in df.columns or df[raw_col].dropna().empty:
        return pd.Series(np.nan, index=df.index)

    # Pivot to (T_samples, N_assets) matrix
    pivot_df = df.pivot(
        index="timestamp", columns="internal_id", values=raw_col
    )

    # Handle NaNs: first row of returns is always NaN. Fill with 0 for PCA.
    clean_pivot = pivot_df.fillna(0)

    min_assets, min_obs = 2, 2
    if clean_pivot.shape[1] < min_assets or clean_pivot.shape[0] < min_obs:
        return pd.Series(np.nan, index=df.index)

    # Calculate residuals for the whole matrix
    res_matrix = RiskModel.get_residual_returns(
        clean_pivot.values, n_factors=min(3, clean_pivot.shape[1] - 1)
    )
    res_df = pd.DataFrame(
        res_matrix, index=clean_pivot.index, columns=clean_pivot.columns
    )

    # Map back to original dataframe index
    lookup = res_df.stack().reset_index(name="res")
    merged = df.merge(lookup, on=["timestamp", "internal_id"], how="left")
    return merged["res"]


@multi_tf_feature(
    name="residual_vol_20",
    timeframes=[Timeframe.MIN_30],
    dependencies=["returns_residual"],
)
def residual_vol_20(df: pd.DataFrame, tf: Timeframe) -> pd.Series:
    res_col = f"returns_residual_{tf.value}"
    return (
        df.groupby("internal_id")[res_col]
        .rolling(window=20)
        .std()
        .reset_index(level=0, drop=True)
    )


@multi_tf_feature(
    name="residual_mom_10",
    timeframes=[Timeframe.MIN_30],
    dependencies=["returns_residual"],
)
def residual_mom_10(df: pd.DataFrame, tf: Timeframe) -> pd.Series:
    res_col = f"returns_residual_{tf.value}"
    return (
        df.groupby("internal_id")[res_col]
        .rolling(window=10)
        .sum()
        .reset_index(level=0, drop=True)
    )


@feature(name="sma_20_30min")
def sma_20_30min(df: pd.DataFrame) -> pd.Series:
    return (
        df.groupby("internal_id")["close_30min"]
        .rolling(window=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
