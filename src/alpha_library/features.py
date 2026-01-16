from typing import cast

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

    rets_matrix = df.pivot(
        index="timestamp", columns="internal_id", values="returns_raw"
    ).fillna(0.0)

    if (
        rets_matrix.shape[0] < MIN_SAMPLES
        or rets_matrix.shape[1] < MIN_SAMPLES
    ):
        demeaned = df["returns_raw"].fillna(0.0) - df.groupby("timestamp")[
            "returns_raw"
        ].transform("mean").fillna(0.0)
        return cast(pd.Series, demeaned)

    rets_residual_matrix = RiskModel.get_residual_returns(rets_matrix.values)

    res_df = pd.DataFrame(
        rets_residual_matrix,
        index=rets_matrix.index,
        columns=rets_matrix.columns,
    )
    res_long = res_df.stack().reset_index()
    res_long.columns = ["timestamp", "internal_id", "residual"]
    res_long["internal_id"] = res_long["internal_id"].astype(
        df["internal_id"].dtype
    )

    merged = df.merge(res_long, on=["timestamp", "internal_id"], how="left")
    result = merged["residual"].fillna(0.0)
    result.index = df.index
    return cast(pd.Series, result)


@feature("residual_mom_10", dependencies=["returns_residual"])
def residual_mom_10(df: pd.DataFrame) -> pd.Series:
    """
    10-period cumulative idiosyncratic return.
    """
    return cast(
        pd.Series,
        df.groupby("internal_id")["returns_residual"].transform(
            lambda x: x.rolling(10).sum()
        ),
    )


@feature("residual_vol_20", dependencies=["returns_residual"])
def residual_vol_20(df: pd.DataFrame) -> pd.Series:
    """
    Rolling volatility of idiosyncratic returns.
    """
    return cast(
        pd.Series,
        df.groupby("internal_id")["returns_residual"].transform(
            lambda x: x.rolling(20).std()
        ),
    )
