from typing import cast

import numpy as np
import pandas as pd

from src.core.alpha_engine import feature
from src.core.risk_model import RiskModel

MIN_SAMPLES = 2


@feature("returns_raw")
def returns_raw(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        df.groupby("internal_id")["close"].pct_change(fill_method=None),
    )


@feature("returns_residual", dependencies=["returns_raw"])
def returns_residual(df: pd.DataFrame) -> pd.Series:
    """
    Computes idiosyncratic returns via PCA factor neutralization.
    """
    if df.empty or "returns_raw" not in df.columns:
        return pd.Series(dtype="float64")

    # Pivot to Matrix (T x N)
    # Using fillna(0.0) to handle the first row of returns which is always NaN
    rets_matrix = df.pivot(
        index="timestamp", columns="internal_id", values="returns_raw"
    ).fillna(0.0)

    if (
        rets_matrix.shape[0] < MIN_SAMPLES
        or rets_matrix.shape[1] < MIN_SAMPLES
    ):
        # Simple demeaned returns fallback
        demeaned = df["returns_raw"].fillna(0.0) - df.groupby("timestamp")[
            "returns_raw"
        ].transform("mean").fillna(0.0)
        return cast(pd.Series, demeaned)

    # Use RiskModel for factor neutralization
    rets_residual_matrix = RiskModel.get_residual_returns(rets_matrix.values)

    # Map back to long-form Series
    res_df = pd.DataFrame(
        rets_residual_matrix,
        index=rets_matrix.index,
        columns=rets_matrix.columns,
    )
    res_long = res_df.stack().reset_index()
    res_long.columns = ["timestamp", "internal_id", "residual"]

    # Ensure internal_id is same type for merge
    res_long["internal_id"] = res_long["internal_id"].astype(
        df["internal_id"].dtype
    )

    # Merge back to original df to ensure index alignment
    merged = df.merge(res_long, on=["timestamp", "internal_id"], how="left")
    result = merged["residual"].fillna(0.0)
    result.index = df.index
    return cast(pd.Series, result)


@feature("sma_10")
def sma_10(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        df.groupby("internal_id")["close"].transform(
            lambda x: x.rolling(10).mean()
        ),
    )


@feature("rsi_14")
def rsi_14(df: pd.DataFrame) -> pd.Series:
    def _rsi(x: pd.Series) -> pd.Series:
        delta = x.diff()
        u, d = delta.clip(lower=0), -delta.clip(upper=0)
        ma_u, ma_d = u.rolling(14).mean(), d.rolling(14).mean()
        return cast(pd.Series, 100 - (100 / (1 + ma_u / ma_d)))

    return cast(pd.Series, df.groupby("internal_id")["close"].transform(_rsi))


@feature("macd")
def macd(df: pd.DataFrame) -> pd.Series:
    def _macd(x: pd.Series) -> pd.Series:
        return cast(pd.Series, x.ewm(span=12).mean() - x.ewm(span=26).mean())

    return cast(pd.Series, df.groupby("internal_id")["close"].transform(_macd))


@feature("ofi")
def ofi(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series, df["volume"] * np.where(df["close"] >= df["open"], 1, -1)
    )
