# Systematic Trading System

This project implements a professional systematic hedge fund architecture for a data-driven investment platform. The system handles the end-to-end pipeline from data ingestion to algorithmic trade execution.

## Project Structure

The codebase is organized into modular packages within the `src/` directory:

### 1. `src/core/` (The Engine)
Primary business logic and mathematical engines.
- **`types.py`**: Centralized core dataclasses (Bar, Order, Security) and Enums (Timeframe, OrderState).
- **`data_platform.py`**: Manages bitemporal Security Master, stateless bar aggregation, and ratio/difference corporate action adjustments.
- **`alpha_engine.py`**: Base `AlphaModel` framework with thread-safe `contextvars` and a centralized feature registry.
- **`portfolio_manager.py`**: QP Optimizer supporting **Total Expected Return reconstruction** using a statistical risk model.
- **`execution_handler.py`**: Robust asynchronous manager using a centralized background worker for spaced-out order slicing.
- **`risk_model.py`**: PCA-based factor risk and realized factor return calculation.

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
- **Simplicity & Legibility**: Designed as a substantially complete demo that remains easy to read, avoiding the bloat of production systems while maintaining high-fidelity logic.
- **Data Integrity**: Bitemporal modeling to distinguish between Event Time and Knowledge Time, eliminating look-ahead bias.
- **Type Safety**: Centralized `types.py` and robust `Timeframe` enums ensure consistency across research and production.
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

## Quick Start

### 1. Simulation (Backtest)
To run the full intra-day simulation (Multi-alpha combination, PCA risk modeling, and PnL attribution):
```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/backtesting/demo.py
```
This runs a multi-day lifecycle including data syncing, risk model calculation, and performance analytics.

### 2. Live Trading (Paper)
To run the system against Alpaca's Paper Trading API:

1.  Copy `.env.example` to `.env` and fill in your credentials:
    ```bash
    cp .env.example .env
    # Edit .env with your APCA_API_KEY_ID and APCA_API_SECRET_KEY
    ```
2.  Run the live demo:
    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    python src/live_demo.py
    ```
    The system will automatically load the credentials from `.env` and switch from Mock to Live mode.
