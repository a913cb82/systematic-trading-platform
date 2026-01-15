# Project Roadmap: Institutional Systematic Trading System

This roadmap outlines the transition from a PoC to an institutional-grade systematic hedge fund architecture, as defined in `hedge_fund_guide.md`.

## Workstream 1: Institutional Data Infrastructure
**Goal:** Implement high-integrity, bitemporal data systems for symbology, universe, and multi-type market data.

- [ ] **Temporal ISM (Security Master):**
  - [ ] Implement relational schema for `Internal_ID` history (FB -> META mapping).
  - [ ] Add support for institutional identifiers: CUSIP, ISIN, SEDOL, and BBG FIGI.
- [ ] **Point-in-Time (PIT) Universe:**
  - [ ] Build a PIT universe management system to eliminate survivorship bias.
  - [ ] Implement `get_universe(date)` to reconstruct tradable sets historically.
- [ ] **Market Data Depth:**
  - [ ] Extend data platform to support Tick-level (Time & Sales) data.
  - [ ] Implement Level 2 (Depth) and Level 3 (MBO) order book storage and playback.
- [ ] **Fundamental & Alternative Data:**
  - [ ] Build ingestion pipeline for 10-K/10-Q financial statements.
  - [ ] Implement "Knowledge Date" lag handling to prevent look-ahead bias in fundamental factors.
  - [ ] Set up NoSQL store (Elasticsearch/MongoDB) for Alternative Data (News/Sentiment).
- [ ] **Unified Feature Store:**
  - [ ] Implement Offline Store (S3/Parquet) for batch training.
  - [ ] Implement Online Store (Redis) for real-time inference.
  - [ ] Ensure calculation logic consistency between Research and Production.

## Workstream 2: Advanced Alpha Research
**Goal:** Refine alpha generation to isolate idiosyncratic returns and improve signal stability.

- [ ] **Residual Forecasting:**
  - [ ] Implement factor neutralization logic to target the residual ($\epsilon$).
  - [ ] Ensure Alpha targets are calculated using the same Risk Model as the Optimizer.
- [ ] **Signal Processing Enhancements:**
  - [ ] **Forecast Shaping:** Implement exponential and linear decay functions for discrete event signals.
  - [ ] **Normalization:** Implement Rank Transformation to map signals to a uniform distribution (0-1).
- [ ] **Rigorous Validation Protocols:**
  - [ ] Implement Sliding and Expanding Window validation frameworks.
  - [ ] Build support for non-sequential validation (Striping/K-Fold) for low-granularity data.
- [ ] **Stacking & Meta-ML:**
  - [ ] Implement a Meta-Model (e.g., Ridge Regression) for dynamic signal combination based on realized performance.

## Workstream 3: Factor Risk & Non-Linear Optimization
**Goal:** Implement professional risk modeling and realistic cost-aware optimization.

- [ ] **Structured Factor Models:**
  - [ ] Implement Statistical Risk Models (PCA) for dynamic correlation capture.
  - [ ] Implement Fundamental Factor Models (e.g., Barra-style) for explainable risk.
- [ ] **Non-Linear Cost Modeling:**
  - [ ] Implement the "Square Root Law" for market impact modeling.
  - [ ] Integrate SOCP (Second-Order Cone Programming) solvers (e.g., Mosek) for non-convex cost handling.
- [ ] **Soft Constraint Framework:**
  - [ ] Refactor Hard Constraints into Soft Penalties using Lagrange multipliers.
  - [ ] Add specific penalties for Leverage, Turnover, and Sector/Factor Neutrality.

## Workstream 4: High-Fidelity Execution & Attribution
**Goal:** Build event-driven execution algorithms and rigorous performance analysis.

- [ ] **Execution Algorithms:**
  - [ ] Implement VWAP (Volume Weighted Average Price) execution logic.
  - [ ] Implement TWAP (Time Weighted Average Price) with randomization.
  - [ ] Build Implementation Shortfall (IS) algo to optimize the Impact vs. Risk trade-off.
- [ ] **Event-Driven OMS:**
  - [ ] Implement a formal FSM (Finite State Machine) for order lifecycles.
  - [ ] Build race-condition handling for asynchronous broker fills and cancellations.
- [ ] **Post-Trade Analysis & TCA:**
  - [ ] Build TCA engine to calculate Slippage against Arrival Price and Interval VWAP.
  - [ ] Implement PnL Attribution: Decompose returns into Factor, Sector, and Alpha model contributions.
- [ ] **Institutional Connectivity:**
  - [ ] Implement FIX Protocol connectivity for low-latency broker communication.
