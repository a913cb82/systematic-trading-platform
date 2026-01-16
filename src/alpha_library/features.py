from typing import cast

import pandas as pd

from src.core.alpha_engine import multi_tf_feature
from src.core.risk_model import RiskModel

MIN_SAMPLES = 2

TIMEFRAMES = ["30min", "1D"]


@multi_tf_feature("returns_raw", TIMEFRAMES)
def returns_raw(df: pd.DataFrame, tf: str) -> pd.Series:
    return cast(
        pd.Series,
        df.groupby("internal_id")[f"close_{tf}"].pct_change(fill_method=None),
    )


@multi_tf_feature("returns_residual", TIMEFRAMES, dependencies=["returns_raw"])
def returns_residual(df: pd.DataFrame, tf: str) -> pd.Series:
    """
    Computes idiosyncratic returns via PCA factor neutralization.
    """
    raw_col = f"returns_raw_{tf}"
    if df.empty or raw_col not in df.columns:
        return pd.Series(dtype="float64")

    rets_matrix = df.pivot(
        index="timestamp", columns="internal_id", values=raw_col
    ).fillna(0.0)

    if (
        rets_matrix.shape[0] < MIN_SAMPLES
        or rets_matrix.shape[1] < MIN_SAMPLES
    ):
        demeaned = df[raw_col].fillna(0.0) - df.groupby("timestamp")[
            raw_col
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


@multi_tf_feature(
    "residual_mom_10", TIMEFRAMES, dependencies=["returns_residual"]
)
def residual_mom_10(df: pd.DataFrame, tf: str) -> pd.Series:
    """
    10-period cumulative idiosyncratic return.
    """
    res_col = f"returns_residual_{tf}"
    return cast(
        pd.Series,
        df.groupby("internal_id")[res_col].transform(
            lambda x: x.rolling(10).sum()
        ),
    )


@multi_tf_feature(
    "residual_vol_20", TIMEFRAMES, dependencies=["returns_residual"]
)
def residual_vol_20(df: pd.DataFrame, tf: str) -> pd.Series:
    """
    Rolling volatility of idiosyncratic returns.
    """
    res_col = f"returns_residual_{tf}"
    return cast(
        pd.Series,
        df.groupby("internal_id")[res_col].transform(
            lambda x: x.rolling(20).std()
        ),
    )
