# Project TODOs

This document outlines the roadmap for building the systematic trading system. Workstreams are designed to be executed in parallel, with key inter-workstream interfaces acting as the contracts between them.

## Pre-requisite: System Architecture & Scaffolding
**Goal:** Establish the core Python framework and directory structure for the PoC.

### Tasks
- [x] **Directory Scaffolding:** Create the project structure (`src/data/`, `src/alpha/`, `src/portfolio/`, `src/execution/`, `src/common/`).
- [x] **Core Base Classes:** Define abstract base classes (ABCs) for key components (e.g., `BaseProvider`, `BaseModel`, `BaseOptimizer`) to ensure interface adherence.
- [x] **Common Library:** Implement shared utilities for logging, timestamp handling, and the `TypedDict` contracts.

---

## Workstream 1: Data Platform & Infrastructure
**Goal:** Build the "Source of Truth" for all reference, market, and fundamental data.

### Tasks
- [ ] **Implement ISM:** Create the service to assign and manage immutable `Internal_ID`s.
- [ ] **Symbology Service:** Build logic to map external tickers to `Internal_ID`.
- [ ] **Market Data Engine:** Set up Parquet storage for cycle-based (bars/ticks) data.
- [ ] **Bitemporal Query Layer:** Implement the abstract access layer handling 'Event' vs 'Knowledge' timestamps to prevent look-ahead bias.
- [ ] **Corporate Action Master:** Design schema and ingestion for Splits, Dividends, and Mergers to enable dynamic price adjustment.
- [ ] **Event Store:** Design schema and ingestion for aperiodic events (Earnings, News, Macro Data).
- [ ] **Point-in-Time Universe:** Implement logic to reconstruct tradable sets for any historical date.

---

## Workstream 2: Alpha Research & Backtesting
**Goal:** Create the environment for generating signals and validating them.

### Tasks
- [ ] **Feature Store:** Build the registry and calculation engine for `CycleFeatures` and `EventFeatures`.
- [ ] **Alpha Generation:** Implement predictive models targeting residual returns.
- [ ] **Signal Combiner:** Implement logic to aggregate multiple signals (e.g., Equal Weighting, Inverse Variance) into a single forecast.
- [ ] **Forecast Publisher:** Implement the mechanism to deliver consolidated forecasts (`dict[int, float]`) to the Optimizer.
- [ ] **Vectorized Backtester:** Build a fast, array-based simulation engine for preliminary research and optimization.
- [ ] **Backtesting Engine:** Build an event-driven simulation loop for OOS vetting.
- [ ] **Backtest Metrics Calculator:** Implement a tool to compute key statistics (Sharpe, Sortino, Max Drawdown) over tunable timeframes from simulation outputs.
- [ ] **Signal Processing:** Implement Z-scoring, Winsorization, and signal decay logic.

---

## Workstream 3: Portfolio Optimization & Risk
**Goal:** Translate alpha signals into an optimal portfolio.

### Tasks
- [ ] **Forecast Subscriber:** Implement the mechanism to receive forecasts from WS2.
- [ ] **Risk Model:** Implement structured factor models (e.g., PCA or Fundamental) to produce covariance matrices.
- [ ] **The Optimizer:** Build the QP solver using `CVXPY` to maximize $U = \text{Forecast} - \text{Risk Penalty} - \text{Transaction Costs}$.
- [ ] **Constraints:** Implement soft constraints for leverage, position limits, and turnover.
- [ ] **Target Weight Publisher:** Implement the mechanism to deliver target weights (`dict[int, float]`) to WS4.

---

## Workstream 4: Execution & Operations
**Goal:** Safely execute trades and manage the interface with the market.

### Tasks
- [ ] **Target Weight Subscriber:** Implement the mechanism to receive target weights from WS3.
- [ ] **OMS:** Implement the state machine for order lifecycles and broker reconciliation. Responsible for deciding specific trades based on target weights and current positions.
- [ ] **Execution Algos:** Build VWAP, TWAP, and Implementation Shortfall algorithms.
- [ ] **Safety Layer:** Implement pre-trade fat-finger and message-rate limits.
- [ ] **Broker Gateway:** Build the adapter for connectivity (FIX/REST).
- [ ] **TCA Engine:** Build the post-trade analysis module to calculate slippage, implementation shortfall, and PnL attribution.

---

## Key Inter-Workstream Interfaces (The Contracts)

These function signatures define the formal contracts between workstreams using simple Python types.

### Type Definitions
```python
class Bar(TypedDict):
    internal_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class Trade(TypedDict):
    internal_id: int
    side: str  # 'BUY' | 'SELL'
    quantity: float
    price: float
    fees: float
    venue: str
    timestamp: datetime

class Event(TypedDict):
    internal_id: int
    type: str
    value: Any
    timestamp_event: datetime
    timestamp_knowledge: datetime
```

### 1. Market Data Interface (WS1)
*Control Flow:* Ingestion runs as batch jobs writing to disk. Research pulls historical data. Production subscribes to live feeds.
- `def write_bars(data: list[Bar]) -> None`
- `def get_bars(internal_ids: list[int], start: datetime, end: datetime, adjustment: str, as_of: datetime = None) -> list[Bar]`
- `def subscribe_bars(internal_ids: list[int], on_bar: Callable[[Bar], None]) -> None`
- `def get_universe(date: datetime) -> list[int]`

### 2. Event Data Interface (WS1)
*Control Flow:* Similar to market data, but for aperiodic events.
- `def write_events(events: list[Event]) -> None`
- `def get_events(event_types: list[str], internal_ids: list[int], start: datetime, end: datetime, as_of: datetime = None) -> list[Event]`
- `def subscribe_events(event_types: list[str], on_event: Callable[[Event], None]) -> None`

### 3. Risk Data Interface (WS1/WS3)
*Control Flow:* Risk models are calculated periodically (Batch) or updated live (Stream).
- `def write_risk_model(date: datetime, matrix: list[list[float]], exposures: list[dict[str, float]]) -> None`
- `def get_covariance_matrix(date: datetime, internal_ids: list[int]) -> list[list[float]]`
- `def get_factor_exposures(date: datetime, internal_ids: list[int]) -> dict[int, dict[str, float]]`

### 4. Alpha Signal Interface (WS2 <-> WS3)
Defines the exchange of alpha forecasts.
*Data Structure:* `forecasts` is a `dict` mapping `Internal_ID` (int) to `Signal Strength` (float).
- `def submit_forecasts(timestamp: datetime, forecasts: dict[int, float]) -> None`
- `def get_forecasts(timestamp: datetime) -> dict[int, float]`
- `def subscribe_forecasts(on_forecast: Callable[[dict[int, float]], None]) -> None`

### 5. Order Instruction Interface (WS3 <-> WS4)
Defines the exchange of portfolio targets.
*Data Structure:* `weights` is a `dict` mapping `Internal_ID` (int) to `Target Portfolio %` (float).
- `def submit_target_weights(timestamp: datetime, weights: dict[int, float]) -> None`
- `def get_target_weights(timestamp: datetime) -> dict[int, float]`
- `def subscribe_target_weights(on_weights: Callable[[dict[int, float]], None]) -> None`

### 6. Execution Feedback Interface (WS4 <-> WS2)
Defines the stream of realized trades for analysis.
- `def report_fill(fill: Trade) -> None`
- `def get_fills(start_time: datetime, end_time: datetime) -> list[Trade]`
- `def subscribe_fills(on_fill: Callable[[Trade], None]) -> None`
