# Institutional Systematic Trading System

This project implements a professional systematic hedge fund architecture, focused on building a robust, data-driven investment platform. The system is designed to handle the end-to-end pipeline from high-integrity data ingestion to algorithmic trade execution.

## Project Structure

The codebase is organized into four main modules within the `src/` directory:

### 1. `src/core/` (The Engine)
The primary business logic and mathematical engines of the system.
- **`data_platform.py`**: Manages the bitemporal Security Master, PIT universe reconstruction, and corporate action adjustments.
- **`alpha_engine.py`**: Provides the base `AlphaModel` framework and `SignalProcessor` for Z-scoring, winsorization, and signal decay.
- **`portfolio_manager.py`**: Implements a QP Optimizer using a **PCA-based Statistical Risk Model** and a **Soft-Constraint Framework** for leverage, turnover, and neutrality.
- **`execution_handler.py`**: Orchestrates the order lifecycle through an event-driven FSM and algorithmic slicing.

### 2. `src/gateways/` (Connectivity)
Handles all external interactions and data provider implementations.
- **`base.py`**: Abstract interfaces for `DataProvider` and `ExecutionBackend`.
- **`alpaca.py`**: Concrete implementation for the Alpaca Markets API.

### 3. `src/alpha_library/` (Research)
A dedicated space for alpha research artifacts and reusable signals.
- **`features.py`**: A centralized registry of technical and fundamental indicators (SMA, RSI, MACD, Residual Returns).
- **`models.py`**: Reusable signal generation models (Momentum, Value).

### 4. `src/backtester_demo.py`
A comprehensive end-to-end simulation that demonstrates the full stack in an intra-day institutional environment.

## Design Philosophy
- **Data Integrity First**: Rigorous bitemporal modeling to distinguish between Event Time and Knowledge Time, eliminating look-ahead bias.
- **Factor Neutrality**: Focus on isolating idiosyncratic returns (alpha) by forecasting residual returns against statistical risk factors.
- **Research to Production Consistency**: Shared feature registry and calculation logic ensure that backtested performance matches live execution.

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
The system maintains high test coverage using `pytest`.

```bash
source .venv/bin/activate
pytest
```

## Code Quality
We use `pre-commit` to enforce institutional standards. This includes **Ruff** for linting/formatting and **Mypy** for static type checking.

### Running Quality Checks
```bash
source .venv/bin/activate
pre-commit run --all-files
```

## Quick Start (Institutional Simulation)
To run the full intra-day simulation (Multi-alpha combination, PCA risk modeling, and VWAP execution):
```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/backtester_demo.py
```
This runs a 14-interval intra-day lifecycle including data syncing, risk model calculation, and post-trade TCA.
