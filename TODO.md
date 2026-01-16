# Project Roadmap: Institutional Systematic Trading System

This roadmap outlines the transition from a PoC to an institutional-grade systematic hedge fund architecture, as defined in `hedge_fund_guide.md`.

## Workstream 1: Institutional Data Infrastructure
**Goal:** Implement high-integrity, bitemporal data systems for symbology, universe, and multi-type market data.

- [x] **Temporal ISM (Security Master):**
  - [x] Implement relational schema for `Internal_ID` history (FB -> META mapping).
  - [x] Add support for institutional identifiers: CUSIP, ISIN, SEDOL, and BBG FIGI.
- [x] **Point-in-Time (PIT) Universe:**
  - [x] Build a PIT universe management system to eliminate survivorship bias.
  - [x] Implement `get_universe(date)` to reconstruct tradable sets historically.
- [x] **Market Data & Event Consolidation:**
  - [x] Standardize on `Bar` (OHLCV) market data.
  - [x] Consolidated Fundamental & Alternative data into generic `Event` structure.
  - [x] Implement "Knowledge Date" lag handling to prevent look-ahead bias.
- [x] **Unified Feature Store:**
  - [x] Move derived logic (e.g., `returns_raw`, `returns_residual`) into feature registry.
  - [x] Implement calculation logic consistency between Research and Production.

## Workstream 2: Advanced Alpha Research
**Goal:** Refine alpha generation to isolate idiosyncratic returns and improve signal stability.

- [x] **Residual Forecasting:**
  - [x] Implement factor neutralization logic (demeaning) via `returns_residual` feature.
- [x] **Signal Processing Enhancements:**
  - [x] **Forecast Shaping:** Implement exponential and linear decay functions for discrete event signals.
  - [x] **Normalization:** Implement Rank Transformation to map signals to a uniform distribution (0-1).
- [x] **Rigorous Validation Protocols:**
  - [x] Implement Sliding and Expanding Window validation frameworks.
- [x] **Stacking & Meta-ML:**
  - [x] Implement a simple linear combination of signals with fixed weights.

## Workstream 3: Factor Risk & Non-Linear Optimization
**Goal:** Implement professional risk modeling and realistic cost-aware optimization.

- [x] **Structured Factor Models:**
  - [x] Implement Statistical Risk Models (PCA) for dynamic correlation capture.
- [x] **Non-Linear Cost Modeling:**
  - [x] Implement the "Square Root Law" for market impact modeling.
- [x] **Soft Constraint Framework:**
  - [x] Refactor Hard Constraints into Soft Penalties using Lagrange multipliers.
  - [x] Add specific penalties for Leverage, Turnover, and Sector/Factor Neutrality.

## Workstream 4: High-Fidelity Execution & Attribution
**Goal:** Build event-driven execution algorithms and rigorous performance analysis.

- [x] **Execution Algorithms:**
  - [x] Implement VWAP/TWAP execution logic.
- [x] **Event-Driven OMS:**
  - [x] Implement a formal FSM (Finite State Machine) for order lifecycles.
- [x] **Post-Trade Analysis & TCA:**
  - [x] Build TCA engine to calculate Slippage and Alpha Decay.
- [x] **Institutional Connectivity:**
  - [x] Implement FIX Protocol connectivity.
