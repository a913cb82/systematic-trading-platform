import time
from typing import Any

from src.core.execution_handler import (
    ExecutionHandler,
    FIXEngine,
    Order,
    OrderState,
    TCAEngine,
)

# Constants to avoid magic values
AAPL_TICKER = "AAPL"
START_QTY = 10.0
START_PRICE = 100.0
GOAL_QTY = 50.0
TOTAL_SLICES = 5
SLICE_QTY = 8.0  # (50-10)/5
VWAP_QTY = 100.0
VWAP_SLICES = 10
FILL_QTY = 50.0
SLIPPAGE_BASE = 100.0
MAX_SLICES_ON_CANCEL = 2
DUAL_SLICE_COUNT = 2


def test_execution_handler_rebalances_portfolio_using_sliced_orders(
    mock_backend: Any,
) -> None:
    mock_backend.get_positions.return_value = {AAPL_TICKER: START_QTY}
    mock_backend.get_prices.return_value = {AAPL_TICKER: START_PRICE}
    handler = ExecutionHandler(mock_backend)
    handler.rebalance({AAPL_TICKER: GOAL_QTY}, interval=0)

    max_wait = 1.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if handler.orders and handler.orders[0].state == OrderState.FILLED:
            break
        time.sleep(0.01)

    assert mock_backend.submit_order.call_count == TOTAL_SLICES
    mock_backend.submit_order.assert_any_call(AAPL_TICKER, SLICE_QTY, "BUY")


def test_execution_handler_stops_slicing_on_order_cancellation(
    mock_backend: Any,
) -> None:
    handler = ExecutionHandler(mock_backend)
    order = handler.vwap_execute(
        AAPL_TICKER, VWAP_QTY, "BUY", slices=VWAP_SLICES, interval=0.1
    )
    time.sleep(0.05)
    handler.cancel_order(order.order_id)
    assert order.state == OrderState.CANCELLED
    time.sleep(0.3)
    assert mock_backend.submit_order.call_count <= MAX_SLICES_ON_CANCEL


def test_execution_handler_sets_rejected_state_on_backend_failure(
    mock_backend: Any,
) -> None:
    mock_backend.submit_order.side_effect = [True, False]
    handler = ExecutionHandler(mock_backend)
    order = handler.vwap_execute(
        AAPL_TICKER, VWAP_QTY, "BUY", slices=DUAL_SLICE_COUNT, interval=0.01
    )

    max_wait = 1.0
    start_time = time.time()
    while time.time() - start_time < max_wait:
        if order.state == OrderState.REJECTED:
            break
        time.sleep(0.01)

    assert order.state == OrderState.REJECTED
    assert mock_backend.submit_order.call_count == DUAL_SLICE_COUNT


def test_order_object_updates_state_correctly_on_partial_fills() -> None:
    o = Order(AAPL_TICKER, VWAP_QTY, "BUY")
    o.update(FILL_QTY)
    assert o.state == OrderState.PARTIAL
    o.update(FILL_QTY)
    assert o.state == OrderState.FILLED


def test_tca_engine_calculates_slippage_from_arrival_prices() -> None:
    assert TCAEngine.calculate_slippage(0, 100, "BUY") == 0.0
    assert TCAEngine.calculate_slippage(100, 101, "BUY") == SLIPPAGE_BASE
    assert TCAEngine.calculate_slippage(100, 99, "SELL") == SLIPPAGE_BASE


def test_fix_engine_logon_stub_returns_success() -> None:
    fix = FIXEngine("TEST")
    assert fix.logon()
    assert AAPL_TICKER in fix.send_order(AAPL_TICKER, 10, "BUY")
