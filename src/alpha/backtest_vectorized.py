import pandas as pd


class VectorizedBacktester:
    def __init__(self, returns_df: pd.DataFrame, signals_df: pd.DataFrame):
        """
        returns_df and signals_df should have internal_id, timestamp,
        and value/signal.
        They should be indexed by (timestamp, internal_id).
        """
        self.returns = returns_df
        self.signals = signals_df

    def run(self) -> pd.Series:
        # Align signals and returns
        # Shift signals by 1 period because signal at T determines
        # return at T+1
        signals = self.signals.copy()
        signals["signal"] = signals.groupby("internal_id")["signal"].shift(1)

        df = pd.merge(
            signals,
            self.returns,
            on=["timestamp", "internal_id"],
            suffixes=("_sig", "_ret"),
        )
        df["strategy_ret"] = df["signal"] * df["returns"]
        return df.groupby("timestamp")["strategy_ret"].mean()

    def calculate_metrics(self, pnl: pd.Series) -> dict:
        """
        Calculates basic performance metrics from a PnL series.
        """
        cumulative_return = (1 + pnl).prod() - 1
        sharpe_ratio = (
            pnl.mean() / pnl.std() * (252**0.5) if pnl.std() != 0 else 0
        )
        return {
            "cumulative_return": cumulative_return,
            "sharpe_ratio": sharpe_ratio,
        }
