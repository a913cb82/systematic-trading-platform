import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv

from src.alpha_library.models import MomentumModel
from src.core.alpha_engine import (
    AlphaEngine,
    ModelRunConfig,
    SignalCombiner,
    SignalProcessor,
)
from src.core.data_platform import DataPlatform
from src.core.execution_handler import ExecutionHandler
from src.core.portfolio_manager import PortfolioManager
from src.core.types import QueryConfig, Timeframe
from src.gateways.alpaca import (
    AlpacaDataProvider,
    AlpacaExecutionBackend,
    AlpacaRealtimeClient,
)
from src.gateways.base import (
    ExecutionBackend,
)


def get_alpaca_credentials() -> Tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID") or ""
    api_secret = os.getenv("APCA_API_SECRET_KEY") or ""
    return api_key, api_secret


def setup_platform(
    api_key: str, api_secret: str
) -> Tuple[DataPlatform, ExecutionBackend]:
    print("[INFO] Using Alpaca Paper Trading")
    provider = AlpacaDataProvider(api_key, api_secret)
    backend: ExecutionBackend = AlpacaExecutionBackend(
        api_key, api_secret, paper=True
    )
    stream_provider = AlpacaRealtimeClient(api_key, api_secret)

    data = DataPlatform(
        provider,
        stream_provider,
        db_path="./.arctic_live_demo_db",
        clear=True,
    )
    return data, backend


def run_strategy_loop(
    data: DataPlatform,
    backend: ExecutionBackend,
    tickers: List[str],
    iids: List[int],
    history_range: Tuple[datetime, datetime],
) -> None:
    print("[STRATEGY] Loop started...")
    start_hist, end_hist = history_range
    pm = PortfolioManager()
    models = [MomentumModel()]

    # Calculate initial risk model
    hist_bars = data.get_bars(
        iids,
        QueryConfig(
            start=start_hist,
            end=end_hist,
            timeframe=Timeframe.MIN_30,
        ),
    )
    expected_factor_returns = None
    if not hist_bars.empty:
        pivot_rets = (
            hist_bars.pivot(
                index="timestamp",
                columns="internal_id",
                values="close_30min",
            )
            .pct_change()
            .dropna()
        )
        if not pivot_rets.empty:
            pm.update_risk_model(pivot_rets.values)
            hist_f_rets = pm.get_factor_returns(pivot_rets.values)
            expected_factor_returns = np.mean(hist_f_rets, axis=0)

    executor = ExecutionHandler(backend)
    while True:
        current_time = datetime.now()
        signals = [
            AlphaEngine.run_model(
                data,
                m,
                iids,
                ModelRunConfig(current_time, Timeframe.MIN_30),
            )
            for m in models
        ]
        combined = SignalCombiner.combine(
            [SignalProcessor.zscore(s) for s in signals]
        )

        pm.optimize(combined, factor_returns=expected_factor_returns)
        prices = backend.get_prices(tickers)
        reverse_ism = data.reverse_ism

        goal_positions: Dict[str, float] = {}
        total_equity = 100000.0

        for iid, weight in pm.current_weights.items():
            ticker = reverse_ism.get(iid)
            if ticker and ticker in prices and prices[ticker] > 0:
                safe_weight = max(min(weight, 0.1), -0.1)
                goal_positions[ticker] = float(
                    round((safe_weight * total_equity) / prices[ticker])
                )

        orders = executor.rebalance(goal_positions, interval=0)
        if orders:
            print(f"[EXECUTION] Generated {len(orders)} parent orders.")

        time.sleep(60)


def run_live_demo() -> None:
    api_key, api_secret = get_alpaca_credentials()
    tickers = ["AAPL", "MSFT"]

    if not api_key or not api_secret:
        print("[ERROR] APCA_API_KEY_ID and APCA_API_SECRET_KEY not set")
        return

    print("\n" + "=" * 50)
    print("RUNNING LIVE TRADING SYSTEM")
    print("=" * 50)

    data, backend = setup_platform(api_key, api_secret)

    end_hist = datetime.now() - timedelta(minutes=16)
    start_hist = end_hist - timedelta(days=5)

    print(f"[INFO] Syncing history from {start_hist} to {end_hist}...")
    data.sync_data(tickers, start_hist, end_hist, timeframe=Timeframe.MINUTE)

    iids = [data.get_internal_id(t) for t in tickers]

    t = threading.Thread(
        target=run_strategy_loop,
        args=(data, backend, tickers, iids, (start_hist, end_hist)),
        daemon=True,
    )
    t.start()

    print("[INFO] Starting Realtime Stream (Blocking)...")
    try:
        data.start_streaming(tickers)
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    import src.alpha_library.features  # noqa: F401

    run_live_demo()
