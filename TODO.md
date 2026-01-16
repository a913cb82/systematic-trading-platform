# Project Roadmap: High-Integrity Systematic Enhancements

This roadmap focuses on integration between alpha generation and risk management, while improving research fidelity.

## Workstream 1: Advanced Feature Pipeline
- [x] **Recursive Feature Dependencies**: Improve the `@feature` decorator in `src/core/alpha_engine.py` to allow features to depend on other registered features.

## Workstream 2: Alpha-Risk Consistency
- [x] **Risk-Aware Residuals**: Update `src/alpha_library/features.py` to use `RiskModel.get_residual_returns` for computing `returns_residual`, ensuring targets are idiosyncratic.

## Workstream 3: Simulation & Attribution
- [x] **Formal Backtest Engine**: Encapsulate multi-day simulation logic in `src/backtesting/engine.py`.
- [x] **Performance Analytics**: Automated calculation of Sharpe Ratio, Drawdown, and PnL Attribution in `src/backtesting/analytics.py`.
- [ ] **Extended Validation**: Build support for Expanding Window cross-validation using the `BacktestValidator`.
