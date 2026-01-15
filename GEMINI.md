# Systematic Trading System - Gemini Context

This directory contains a systematic hedge fund architecture project. It is designed as a data-driven investment platform covering the end-to-end pipeline from data ingestion to trade execution.

## Project Structure & Current State

- **Roadmap & Tasks:** Refer to `TODO.md` to understand the current progress, active workstreams, and upcoming tasks. This is the primary source of truth for the project's evolution.
- **Architecture Guide:** Detailed architectural principles, mathematical foundations, and implementation strategies are documented in `hedge_fund_guide.md`.
- **High-Level Overview:** See `README.md` for a summary of core modules and the design philosophy.

## Technical Framework

- **Language:** Python.
- **Environment:** A virtual environment is used (typically in `.venv/`).
- **Dependencies:** Refer to `requirements.txt` for the list of required libraries.
- **Interfaces:** Key inter-workstream contracts are defined using Python `TypedDict` and abstract base classes. Formal signatures and data structures are maintained in the "Key Inter-Workstream Interfaces" section of `TODO.md`.

## Development Principles

- **Bitemporal Integrity:** All data modeling must distinguish between Event Time and Knowledge Time to eliminate look-ahead bias.
- **Modularity:** Workstreams (Data, Alpha, Portfolio, Execution) are decoupled and communicate via fixed interfaces to allow for parallel development and independent scaling.
- **Research-to-Production Consistency:** Logic used in backtesting must be identical to what runs in live trading to prevent online-offline skew.

## Key Commands

- **Environment Setup:** `source .venv/bin/activate && pip install -r requirements.txt`
- **Tests/Execution:** (Refer to `TODO.md` or specific module READMEs for task-specific execution commands as the project matures).
