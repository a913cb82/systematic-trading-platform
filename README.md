# Systematic Trading System

This project implements a professional systematic hedge fund architecture for a data-driven investment platform. The system handles the end-to-end pipeline from data ingestion to algorithmic trade execution.

## Project Structure

The codebase is organized into modular packages within the `src/` directory:

### 1. `src/core/` (The Engine)
Primary business logic and mathematical engines.
- **`data_platform.py`**: Manages bitemporal Security Master, PIT universe reconstruction, and corporate action adjustments.
- **`alpha_engine.py`**: Base `AlphaModel` framework and `SignalProcessor` for Z-scoring, winsorization, and signal decay.
- **`portfolio_manager.py`**: QP Optimizer using a **PCA-based Statistical Risk Model** and a **Soft-Constraint Framework** for leverage, turnover, and neutrality.
- **`execution_handler.py`**: Orchestrates order lifecycle through an event-driven FSM and algorithmic slicing.
- **`risk_model.py`**: Shared PCA-based factor risk logic.

### 2. `src/backtesting/` (Simulation)
Formalized research and simulation tools.
- **`engine.py`**: Multi-day simulation loop with risk model updates and rebalancing cycles.
- **`analytics.py`**: Performance metrics (Sharpe, Drawdown) and PnL attribution.
- **`demo.py`**: End-to-end backtest simulation demonstrating the full stack.

### 3. `src/gateways/` (Connectivity)
External interactions and data provider implementations.
- **`base.py`**: Abstract interfaces for `DataProvider` and `ExecutionBackend`.
- **`alpaca.py`**: Implementation for the Alpaca Markets API.

### 4. `src/alpha_library/` (Research)
Alpha research artifacts and reusable signals.
- **`features.py`**: Centralized registry of technical and fundamental indicators (SMA, RSI, MACD, Residual Returns).
- **`models.py`**: Signal generation models (Momentum, Value).

## Design Philosophy
- **Data Integrity**: Bitemporal modeling to distinguish between Event Time and Knowledge Time, eliminating look-ahead bias.
- **Factor Neutrality**: Focus on isolating idiosyncratic returns (alpha) by forecasting residuals against statistical risk factors.
- **Consistency**: Shared feature registry and calculation logic ensure backtested performance matches live execution.

## Development

### Setup
Dependencies are managed via `pyproject.toml`.

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
```bash
source .venv/bin/activate
pytest
```

## Code Quality
We use `pre-commit` to enforce standards via **Ruff** and **Mypy**.

### Running Quality Checks
```bash
source .venv/bin/activate
pre-commit run --all-files
```

## Quick Start (Simulation)
To run the full intra-day simulation (Multi-alpha combination, PCA risk modeling, and PnL attribution):
```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/backtesting/demo.py
```
This runs a multi-day lifecycle including data syncing, risk model calculation, and performance analytics.
