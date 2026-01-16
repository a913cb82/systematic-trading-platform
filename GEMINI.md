# Systematic Trading System - Gemini Context

This repository contains a modular, data-driven investment platform designed for systematic trading. It encompasses the end-to-end pipeline from data ingestion and feature engineering to portfolio optimization and execution.

## Core Documentation

- **Philosophy & Blueprint:** `ARCHITECTURE.md` contains the primary technical specifications, mathematical foundations, and bitemporal data integrity rules.
- **Operational Status:** `TODO.md` tracks active workstreams, roadmap progress, and current development tasks.
- **Onboarding:** `README.md` provides a high-level summary of core modules and setup instructions.

## Technical Framework

- **Runtime:** Python 3.10+.
- **Environment:** Managed via standard Python virtual environments (typically `.venv/`).
- **Dependencies:** Defined in `pyproject.toml`.
- **Quality Control:** Adheres to strict `pre-commit` hooks including `ruff` for linting/formatting, `mypy` for static type checking, and `pytest` for verification.
- **Contract Definition:** Workstreams communicate via fixed interfaces using abstract base classes (ABCs) and structured data types (e.g., `dataclasses`, `TypedDict`) to ensure decoupling and parallel development.

## Development Principles

- **Bitemporal Integrity:** All data modeling distinguishes between *Event Time* and *Knowledge Time* to eliminate look-ahead bias.
- **Research-to-Production Consistency:** Logic used in backtesting must be identical to what runs in live trading to prevent online-offline skew.
- **Modularity:** The four primary workstreams (Data, Alpha, Portfolio, Execution) are decoupled. Any change in one module should not break another if the inter-workstream interfaces are respected.

## Standard Workflow

- **Environment Setup:** `pip install -e ".[dev]"`
- **Verification:** `pytest`
- **Linting:** `pre-commit run --all-files`
