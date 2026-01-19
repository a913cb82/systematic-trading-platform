import numpy as np
import pandas as pd
import pytest

from src.backtesting.analytics import PerformanceAnalyzer

# Constants to avoid magic values
AAPL_RET_TOTAL = 0.15
DAILY_TABLE_LEN = 6
ME_TABLE_LEN = 2


def test_performanceanalyzer_calculates_sharpe_ratio_correctly() -> None:
    assert PerformanceAnalyzer.calculate_sharpe(pd.Series([0, 0])) == 0.0
    rets = pd.Series([0.01, -0.01, 0.02])
    mean_ret = rets.mean()
    std_ret = rets.std()
    expected = (mean_ret / std_ret) * np.sqrt(252)
    res = PerformanceAnalyzer.calculate_sharpe(rets)
    assert pytest.approx(res) == expected


def test_performanceanalyzer_calculates_drawdowns_accurately() -> None:
    empty_series = pd.Series([], dtype=float)
    res_empty = PerformanceAnalyzer.calculate_drawdown(empty_series)
    assert res_empty == {"max_dd": 0.0}

    equity = pd.Series([100, 110, 90, 95, 80, 120])
    res = PerformanceAnalyzer.calculate_drawdown(equity)
    assert pytest.approx(res["max_dd"]) == (80 - 110) / 110.0
    assert pytest.approx(res["current_dd"]) == (120 - 120) / 120.0


def test_performanceanalyzer_decomposes_returns_via_factor_attribution() -> (
    None
):
    weights = {1: 0.5, 2: 0.5}
    returns = {1: 0.1, 2: 0.2}
    loadings = np.array([[1.0], [1.0]])
    res = PerformanceAnalyzer.factor_attribution(weights, returns, loadings)
    assert pytest.approx(res["total"]) == AAPL_RET_TOTAL


def test_performanceanalyzer_generates_performance_table_with_resampling() -> (
    None
):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D"),
            "gross_ret": [0.01] * 5,
            "net_ret": [0.008] * 5,
        }
    )
    table_d = PerformanceAnalyzer.generate_performance_table(df, freq="D")
    assert "Summary" in table_d.index
    assert len(table_d) == DAILY_TABLE_LEN

    table_me = PerformanceAnalyzer.generate_performance_table(df, freq="ME")
    assert len(table_me) == ME_TABLE_LEN


def test_performanceanalyzer_handles_empty_datasets_gracefully() -> None:
    df = pd.DataFrame(columns=["timestamp", "gross_ret", "net_ret"])
    table = PerformanceAnalyzer.generate_performance_table(df)
    assert len(table) == 1
    assert table.index[0] == "Summary"
