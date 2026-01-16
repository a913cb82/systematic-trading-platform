from typing import cast

import numpy as np
import pandas as pd

from src.core.alpha_engine import feature


@feature("returns_raw")
def returns_raw(df: pd.DataFrame) -> pd.Series:
    return cast(
        pd.Series,
        df.groupby("internal_id")["close"].pct_change(fill_method=None),
    )


@feature("returns_residual")
def returns_residual(df: pd.DataFrame) -> pd.Series:
    # Demeaned returns (idiosyncratic proxy)
    rets = df.groupby("internal_id")["close"].pct_change(fill_method=None)
    return cast(
        pd.Series, rets - rets.groupby(df["timestamp"]).transform("mean")
    )


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
