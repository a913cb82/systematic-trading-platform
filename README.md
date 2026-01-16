# Trading System

This project implements a systematic hedge fund architecture, focused on building a robust, data-driven investment platform. The system is designed to handle the end-to-end pipeline from data ingestion to trade execution.

## Core Modules

### 1. Data Platform
The foundation of the system, ensuring data integrity and preventing bias.
- **Internal Security Master (ISM):** Manages stable identifiers across corporate actions and ticker changes.
- **Market Data Engine:** Stores and retrieves high-frequency tick and bar data using high-performance columnar storage.
- **Bitemporal Modeling:** Distinguishes between event time and knowledge time to eliminate look-ahead bias.
- **Feature Store:** Centralized repository for consistent feature calculation in both research and production.

### 2. Alpha Generation
Focuses on forecasting idiosyncratic returns (alpha).
- **Signal Models:** Predictive models targeting residual returns after accounting for market and sector factors.
- **Backtesting Engine:** An event-driven simulation environment for rigorous out-of-sample vetting and performance analysis.
- **Signal Combination:** Aggregates multiple strategies using performance-based weighting and stacking techniques.

### 3. Portfolio Optimization & Risk
Translates forecasts into optimal capital allocations.
- **Utility Maximization:** Solves Quadratic Programming (QP) problems to balance expected returns, risk penalties, and transaction costs.
- **Risk Models:** Employs structured factor models (Fundamental and Statistical) to manage covariance and exposure.
- **Constraint Management:** Implements soft constraints for leverage, position limits, and turnover control.

### 4. Execution & Operations
Manages the lifecycle of orders and market connectivity.
- **Execution Algorithms:** Minimizes implementation shortfall using VWAP, TWAP, and Smart Order Routing (SOR).
- **Order Management System (OMS):** A state machine for order tracking, safety limits, and broker reconciliation.
- **Post-Trade Analysis:** Performs Transaction Cost Analysis (TCA) and PnL attribution to refine the investment process.

## Design Philosophy
- **Data Integrity First:** Priority is given to accurate, point-in-time data representation.
- **Modular Architecture:** Each component is decoupled to allow for independent scaling and upgrades.
- **Research to Production Consistency:** Ensuring that the logic used in backtesting is identical to what runs in live trading.

## Development

### Setup
The project manages dependencies via `pyproject.toml`.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies including dev tools
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
The project uses `pytest` to run tests. Some tests are written using the standard `unittest` library but are compatible with `pytest`.

```bash
source .venv/bin/activate
pytest
```

## Code Quality
We use `pre-commit` to ensure code quality standards are met. This includes **Ruff** for linting and formatting, and **Mypy** for static type checking, along with automated testing via **Pytest**.

### Running Quality Checks
To manually run all checks on all files:
```bash
source .venv/bin/activate
pre-commit run --all-files
```

This ensures adherence to the project's 79-character line length standard and verifies that all tests pass before code is committed.

## Quick Start (Institutional Simulation)
To run the end-to-end institutional trading simulation (Multi-alpha combination, PCA risk modeling, and VWAP execution):
```bash
source .venv/bin/activate
python hedge_fund_full_stack.py
```
This will run a full lifecycle simulation including data ingestion, alpha generation, optimization, and post-trade analysis.
