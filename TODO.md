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
- [x] **Implement ISM:** Create the service to assign and manage immutable `Internal_ID`s.
- [x] **Symbology Service:** Build logic to map external tickers to `Internal_ID`.
- [ ] **Market Data Engine:** Set up Parquet storage for cycle-based (bars/ticks) data.
  - [x] Initial Parquet implementation.
  - [ ] Implement dynamic price adjustment (Ratio/Raw) in `get_bars`.
- [x] **Bitemporal Query Layer:** Implement the abstract access layer handling 'Event' vs 'Knowledge' timestamps to prevent look-ahead bias.
- [x] **Corporate Action Master:** Design schema and ingestion for Splits, Dividends, and Mergers to enable dynamic price adjustment.
- [x] **Event Store:** Design schema and ingestion for aperiodic events (Earnings, News, Macro Data).
- [x] **Point-in-Time Universe:** Implement logic to reconstruct tradable sets for any historical date.

---

## Workstream 2: Alpha Research & Backtesting
**Goal:** Create the environment for generating signals and validating them.

### Tasks
- [x] **Feature Store:** Build the registry and calculation engine for `CycleFeatures` and `EventFeatures`.
- [ ] **Alpha Generation:** Implement predictive models targeting residual returns.
  - [x] Mean Reversion PoC (Raw returns).
  - [ ] Refactor to target Residual Returns using Risk Model.
- [x] **Signal Combiner:** Implement logic to aggregate multiple signals (e.g., Equal Weighting, Inverse Variance) into a single forecast.
- [x] **Forecast Publisher:** Implement the mechanism to deliver consolidated forecasts (`dict[int, float]`) to the Optimizer.
- [x] **Vectorized Backtester:** Build a fast, array-based simulation engine for preliminary research and optimization.
- [x] **Backtesting Engine:** Build an event-driven simulation loop for OOS vetting.
- [x] **Backtest Metrics Calculator:** Implement a tool to compute key statistics (Sharpe, Sortino, Max Drawdown) over tunable timeframes from simulation outputs.
- [ ] **Signal Processing:** Implement Z-scoring, Winsorization, and signal decay logic.
  - [x] Z-scoring and Ranking.
  - [ ] Implement Winsorization and exponential decay functions.

---

## Workstream 3: Portfolio Optimization & Risk
**Goal:** Translate alpha signals into an optimal portfolio.

### Tasks
- [x] **Forecast Subscriber:** Implement the mechanism to receive forecasts from WS2.
- [x] **Risk Model:** Implement structured factor models (e.g., PCA or Fundamental) to produce covariance matrices.
- [ ] **The Optimizer:** Build the QP solver using `CVXPY` to maximize $U = \text{Forecast} - \text{Risk Penalty} - \text{Transaction Costs}$.
  - [x] Initial Cvxpy implementation.
  - [ ] Reconstruct Total Expected Return ($\mu$) from residuals and factor returns.
- [ ] **Constraints:** Implement soft constraints for leverage, position limits, and turnover.
  - [x] Hard constraints implementation.
  - [ ] Refactor Hard Constraints into Soft Penalties for solver robustness.
- [x] **Target Weight Publisher:** Implement the mechanism to deliver target weights (`dict[int, float]`) to WS4.

---

## Workstream 4: Execution & Operations
**Goal:** Safely execute trades and manage the interface with the market.

### Tasks
- [x] **Target Weight Subscriber:** Implement the mechanism to receive target weights from WS3.
- [ ] **OMS:** Implement the state machine for order lifecycles and broker reconciliation.
  - [x] Initial Position/Weight relay.
  - [ ] Implement formal State Machine (PENDING, SUBMITTED, FILLED).
- [x] **Execution Algos:** Build basic simulation algorithm (filling at close).
- [ ] **Safety Layer:** Implement pre-trade weight and leverage limits.
  - [x] Weight and Leverage limits.
  - [ ] Implement ADV-based "Fat Finger" rejection logic.
- [x] **Broker Gateway:** Build the adapter for connectivity (FIX/REST). (Simulated/Stubbed for PoC)
- [x] **TCA Engine:** Build basic slippage calculation module.

---

## Phase 1.5: Architectural Alignment & Compliance
**Goal:** Close gaps between current PoC and the Systematic Hedge Fund Guide.

### Data Platform Compliance
- [ ] **Dynamic Price Adjustment:** Use `CorporateActionMaster` to apply Ratio/Raw adjustments on-the-fly in `MarketDataEngine`.
- [ ] **Returns Engine:** Implement `get_returns` supporting both Raw and Residual (factor-neutral) calculations.

### Alpha Research Compliance
- [ ] **Residual Forecasting:** Refactor `MeanReversionModel` to explicitly target idiosyncratic movement ($\epsilon$).
- [ ] **Processing Enhancements:** Add `winsorize` and `apply_decay` to `SignalProcessor`.

### Portfolio & Risk Compliance
- [ ] **Utility Refinement:** Update objective function to treat forecasts as residuals and add soft constraint penalties.

### Execution Compliance
- [ ] **Order Lifecycle:** Transition OMS from weight-relay to an event-driven state machine.
- [ ] **ADV Safety:** Integrate 30-day Average Daily Volume into pre-trade risk checks.

---

## Phase 2: Production Readiness (Upcoming)

### Workstream 1: Data Platform (Expansion)
- [x] **A1: Live Ingestion Adapters:** Build connectivity to external providers (e.g., Alpaca, IEX, Polygon).
- [x] **A2: Persistent Security Master:** Move ISM to a relational database (PostgreSQL) with full temporal history.

### Workstream 2: Alpha Research (Sophistication)
- [x] **B1: Advanced Feature Library:** Implement technical indicators (RSI, MACD) and microstructure features (Order Flow Imbalance).

### Workstream 3: Portfolio & Risk (Institutional Grade)
- [x] **C1: Structured Risk Models:** Implement PCA-based and Fundamental Factor risk models.
- [x] **C2: Non-Linear Cost Modeling:** Integrate the Square Root Law for market impact in the optimizer.
- [x] **C3: Neutrality Constraints:** Implement soft constraints for Sector and Factor (Market) neutrality.


### Workstream 4: Execution & Operations (Live)
- [x] **D1: Live Broker Gateway:** Implement FIX/REST connectivity for a production broker (e.g., Interactive Brokers).
- [x] **D2: OMS Reconciliation:** Build the T+0 reconciliation engine to sync internal positions with broker state.
- [x] **D3: Live Safety Layer:** Implement message-rate limiting and intraday kill-switches.

### Infrastructure & Monitoring
- [x] **E1: Configuration System:** Replace hardcoded values with a robust YAML/Env-based configuration manager.
- [x] **E2: Monitoring & Alerting:** Build health-check heartbeats and PnL/Exposure alerts.

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
