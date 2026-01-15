This document outlines the architecture for building a systematic hedge fund. In quantitative finance, data integrity is the primary constraint. If the input data does not accurately reflect historical reality, the strategy will fail regardless of the mathematical sophistication applied.

---

# Section I: The Data Platform

## 1. Reference Data
Reference data provides the context for price and volume. It is required to link datasets, handle corporate events, and prevent survivorship bias.

### A. Symbology Mapping
Tickers are unstable identifiers. Companies change names (e.g., Facebook `FB` $\to$ Meta `META`), merge, or delist. Relying on exchange tickers as primary keys in a database creates data integrity issues.

**Implementation:**
An **Internal Security Master (ISM)** is required. This is a relational table linking a permanent, internal Unique ID (UUID) to external identifiers over specific time ranges.

**Table Structure:**
*   `Internal_ID`: 1001 (Immutable)
*   `Ticker`: FB
*   `Exchange`: NASDAQ
*   `Start_Date`: 2012-05-18
*   `End_Date`: 2022-06-08
*   `Next_ID_Pointer`: 1001 (Self-reference, remains the same entity)

*New Row for Change:*
*   `Internal_ID`: 1001
*   `Ticker`: META
*   `Start_Date`: 2022-06-09
*   `End_Date`: NULL (Active)

*Mappings required:* CUSIP, ISIN, SEDOL, RIC (Reuters), BBG_FIGI (Bloomberg).

### B. Corporate Actions (Splits, Dividends, Mergers)
Price data is discontinuous due to corporate actions. A 2-for-1 split reduces the price by 50% without a corresponding loss in value. A simple multiplication factor is insufficient for all use cases.

**Ratio vs. Difference Adjustments:**
1.  **Ratio Adjustment:** Preserves percentage returns but changes absolute price levels.
    *   *Use Case:* Backtesting strategies that trade on percentage returns.
2.  **Difference Adjustment:** Preserves absolute price differences but distorts percentage returns.
    *   *Use Case:* Derivatives pricing or absolute spread trading.

**Implementation:**
Store Raw Prices and corporate action metadata separately. Do not overwrite raw data.
1.  **Raw Price Table:** Open, High, Low, Close, Volume (as traded).
2.  **Corporate Action Table:** Date, Symbol, Action_Type (Split/Div), Value.
3.  **On-the-fly View:** Generate the adjusted series dynamically in memory based on the specific consumption need (Execution requires Raw; Backtesting requires Ratio).

### C. Universe Management & Survivorship Bias
Using a current constituent list (e.g., current S&P 500) for historical backtesting introduces survivorship bias, as it excludes companies that went bankrupt or were removed.

**Implementation:**
Implement a "Point-in-Time" (PIT) universe table:
*   `Date`: 2008-01-01, `Symbol`: LEH, `In_Index`: True
*   `Date`: 2008-09-15, `Symbol`: LEH, `In_Index`: False

**Rule:** The backtester must query the universe configuration valid *as of* the simulation date.

---

## 2. Market Data

### A. Data Types
1.  **Ticks (Time & Sales):** Individual trade execution records. Required for High-Frequency Trading (HFT) and Transaction Cost Analysis (TCA).
2.  **Bars (OHLCV):** Aggregations.
    *   *Time-Bars:* Standard fixed intervals (e.g., 5 mins).
    *   *Volume/Tick Bars:* A new bar is formed every $N$ shares or $N$ trades. This normalizes information flow, as highly active periods generate more bars.
3.  **Order Book:**
    *   **Level 1 (BBO):** Best Bid and Offer.
    *   **Level 2 (Depth):** Top price levels with aggregate volume.
    *   **Level 3 (MBO):** Market by Order. Specific limit orders in the queue.

### B. Storage Architecture
Standard row-oriented SQL databases are often inefficient for large-scale tick data.

**Architecture Recommendations:**
*   **Live/Hot Data:** KDB+, DolphinDB, or ClickHouse.
*   **Historical/Cold Data:** Parquet files stored in object storage (AWS S3 or MinIO). Parquet is columnar and highly compressible.
*   **Partitioning:** By Date, Region, and Asset Class (e.g., `s3://market-data/equities/usa/2023-11-25/ticks.parquet`).

---

## 3. Fundamental Data

Fundamental data involves metrics derived from company financial statements, such as Balance Sheets, Income Statements, and Cash Flow Statements. Unlike market data, this data is low-frequency and notoriously subject to restatements.

**Implementation:**
*   **Storage:** Relational Database (PostgreSQL) is usually sufficient given the lower volume.
*   **Standardization:** Different accounting standards (GAAP vs. IFRS) must be normalized.
*   **Lag Handling:** Essential to map data to the "Knowledge Date" (when the 10-K was published), not the "Period End Date" (end of the quarter), to prevent look-ahead bias.

---

## 4. Alternative Data

Alternative data refers to non-traditional datasets used to gain an informational edge, sourced from, e.g., News, Social Media or PDF reports.

**Implementation:**
*   **Storage:** Use NoSQL stores (e.g., Elasticsearch, MongoDB) or the Data Lakehouse for raw ingestion.
*   **Entity Recognition (NER):** Extract entities immediately upon ingestion and map to the **ISM**.
*   **Signal Extraction:** Only the derived numerical signal (e.g., `sentiment_score`) moves to the high-performance time-series stores.

---

## 5. Data Integrity: Bitemporal Modeling

Data is frequently restated (e.g., a company corrects a previous earnings report). To prevent look-ahead bias, the database architecture for *all* data types (Fundamental, Alternative, and corrected Market data) must distinguish between two timestamps:

1.  **Event Time (Effective Date):** When the event happened (e.g., Q3 Earnings period end).
2.  **Knowledge Time (Publication Date):** When the data was available to the public (e.g., Earnings release date).

**Query Logic:**
`SELECT * FROM earnings WHERE timestamp_event <= '2023-09-30' AND timestamp_knowledge <= '2023-10-01'`

## 6. The Feature Store

To bridge the gap between Research (Batch/Historical) and Production (Real-time/Inference), a **Feature Store** is mandatory.

**The Problem:** Calculating a feature like `14-day RSI` in Python during backtesting, but implementing it in C++ during execution leads to "Online-Offline Skew."

**The Solution:**
1.  **Feature Registry:** Define the logic once.
2.  **Offline Store:** Computes features historically for training (from S3).
3.  **Online Store:** Computes features in real-time for low-latency inference.
4.  **Consistency:** The Feature Store ensures the logic is identical in both environments.

---

# Section II: Alpha Generation (Forecasting)

## 1. The Forecast Target

The objective is to generate conditional expectations of returns.

### A. The Problem with Price and Raw Returns
*   **Price:** Non-stationary. Statistical properties (mean, variance) change over time.
*   **Raw Returns:** Dominated by market beta. Predicting raw returns often results in a model that mimics the benchmark index rather than generating alpha.

### B. The Solution: Residual Returns
The goal is to isolate the **idiosyncratic** movement of the asset ($\epsilon$). To ensure consistency between alpha generation and portfolio construction, the residual return must be calculated using the **same Risk Model** that will be used in the Optimizer (Section III).

$$ r_{stock} = \beta_{market} \cdot r_{market} + \beta_{sector} \cdot r_{sector} + \dots + \epsilon $$

**Implementation:**
Regress asset returns against the returns for factors defined in your Risk Model. The target variable for the forecasting model is the residual ($\epsilon$).

---

## 2. Feature Engineering

Feature engineering is a primary determinant of model performance.

### A. Stationarity & Log Returns
Financial time series exhibit memory but are non-stationary.
*   **Log Returns:** Standard input transformation ($ \ln(P_t) - \ln(P_{t-1}) $) to approximate stationarity.

### B. Categories of Features

1.  **Technical:**
    *   *Indicators:* RSI, Bollinger Bands, MACD.
    *   *Candlesticks:* Discrete pattern recognition (e.g., Doji, Engulfing, Hammer) encoded as boolean flags or categorical integers.
2.  **Fundamental (Point-in-Time):** Price-to-Book, Earnings Yield, Debt-to-Equity.
3.  **Microstructure:** Bid-Ask Spread, Volume Imbalance, VWAP distance.
4.  **Alternative:** Sentiment scores.

---

## 3. Signal Generation: The Models

### A. Model Types: Cycle vs. Event
1.  **Cycle Models:** Forecast at every regular time step (e.g., every 5 minutes).
2.  **Event Models:** Forecast only when a specific condition is met (e.g., an earnings release, a specific candlestick pattern, or a macro announcement).

### B. Sample Weighting
When training models, not all training samples are equal.
*   **Liquidity Weighting:** It is more important to forecast accurately for liquid stocks, as they allow for larger position sizing. A common technique is to set sample weights proportional to $\sqrt{\text{ADV}}$ (Average Daily Volume).
*   **Inverse Volatility:** Weighting samples by inverse volatility reduces the influence of high-noise periods.

### C. Validation
Validation protocols must strictly preserve the temporal sequence of data to avoid look-ahead bias (data leakage).

1. Windowing: Windowing enforces that the training window ends exactly at the forecast date (T_{train}), providing the most accurate out-of-sample performance estimates and capturing regime shifts.
   * Sliding Window: The training period is fixed in duration (e.g., last 5 years). This ensures a constant, recent sample size, avoiding excessive influence from old market regimes.
   * Expanding Window: The training period starts at $T_0$ and expands to $T_{train}$. This leverages all available history, improving statistical power, but may be dominated by old data.

2. Non-Sequential Methods: When the dataset size is insufficient for stable Windowing (data starvation, niche assets), non-sequential methods are sometimes used to increase statistical stability, despite the risk of weak data leakage.
   * K-Fold Cross-Validation: Splits the time series into K non-sequential segments for train/test rotation.
   * Striping: Uses non-contiguous periods (e.g., alternating months) for training to maximize sample size.

Final models validated this way require a large, contiguous hold-out set (e.g., final 1-2 years) that was never used in any training or validation step.

---

## 4. Signal Processing

### A. Forecast Shaping
For **Event Models**, the signal occurs at a single point in time. To be usable by a portfolio optimizer, this discrete event must be converted into a continuous signal.
*   **Implementation:** Apply a decay function (exponential or linear) to the prediction.
    $$ \text{Signal}_t = \text{Prediction}_{t_0} \times e^{-\lambda (t - t_0)} $$
    The half-life of the decay should match the estimated information persistence of the event.

### B. Normalization
Raw model outputs often have different distributions. Before combination, signals must be brought to a comparable scale.
*   **Z-Scoring:** $z = (x - \mu) / \sigma$. This centers the signal and scales it by volatility.
*   **Rank Transformation:** Map signals to a uniform distribution (0 to 1) based on rank. Robust to outliers.

### C. Winsorization
Clip signals at a statistical threshold (e.g., 99th percentile) to prevent a single asset from dominating the optimization.

---

## 5. Model Selection

Model selection acts as the final gate prior to signal combination (Section II.7). It requires rigorous performance evaluation on a completely unseen block of data to confirm generalization capacity.

### A. Out-of-Sample (OOS) Vetting
Models are critically evaluated against a dedicated, contiguous **hold-out set** (e.g., the latest 1 year of data). This OOS period must be excluded from all prior training and validation runs (Section II.3.C).
 * **Isolation:** Model parameters are fixed based on in-sample training. No further hyperparameter tuning or feature engineering is permitted using OOS data.
 * **Decision Mandate:** Failure to meet minimum performance criteria on the OOS set (e.g., Net Sharpe below hurdle, excessive factor exposure) results in the idea being abandoned. No iterative OOS tuning is permitted.

### B. Performance Review
Promotion decisions rely on a multivariate review of statistics vectors from the Backtesting Engine (Section II.7.D/E):
 * **Risk-Adjusted Performance:** Assess Net Sharpe and Information Ratio (IR). Alpha capture must be statistically significant and factor-benchmark neutral.
 * **Cost Efficiency:** Compare Gross vs. Net Sharpe. Low Gross-to-Net conversion mandates rejection or decay rate re-engineering (Section II.4.A), indicating alpha erosion by simulated transaction costs ($\Gamma_t$).
 * **Risk Profile & Exposure:** Review OOS Maximum Drawdown (MDD), VaR/CVaR, and Factor Betas. Betas must confirm PnL isolation to the forecasted residual ($\epsilon$), preventing unintended bets against the Risk Model.
 * **Correlation Analysis:** Evaluate returns correlation against promoted models and factor indices. Low correlation is preferred for maximizing diversification in Signal Combination.

---

## 6. Signal Combination

When running multiple strategies, they must be aggregated.

### A. Equal Weighting
$$ \text{Final Signal} = \frac{1}{N} \sum_{i=1}^{N} \text{Normalized\_Signal}_i $$
Simple and often robust against overfitting, hard to beat in noisy environments. Note that inputs must be normalized (e.g., Z-scored) first so high-volatility models do not dominate.

### B. Performance-Based Weighting
Weight models based on their backtested efficacy.
*   **Inverse Variance:** $w_i \propto 1 / \sigma^2_i$. Allocates less to unstable models.
*   **Sharpe Ratio Weighting:** $w_i \propto \text{Sharpe}_i$. Allocates more to models with better risk-adjusted returns.

### C. Stacking Regression
Train a meta-ML model (e.g., Ridge Regression) that takes the outputs of individual models as features and predicts the realized residual return. This learns to weight models dynamically based on their predictive power and correlations.
*   **Risk:** This carries a high risk of overfitting. Careful regularuzation and strict validation is required.

### D. Simulation-Based Optimization
Run backtest simulations for the portfolio with varying model weights.
*   **Method:** Use an optimization algorithm (e.g., Bayesian or Grid Search) to explore the high-dimensional weight space.
*   **Risk:** This is computationally expensive and carries a high risk of overfitting (mining for lucky weights). Limiting the exploration space and strict validation is required.

---

## 7. Backtesting
The backtesting engine is the foundation of research, providing a simulated, realistic environment to evaluate the entire investment process, from data quality to execution costs. Its primary purpose is to estimate the out-of-sample performance of the combined alpha, risk, and cost model before live deployment.

### A. Engine Architecture
The Event-Driven Backtester serves as the high-fidelity core for simulation, required for accurately modeling transaction costs, market microstructure, and time-dependent alpha signals (Mid-Frequency to HFT). The less complex Vectorized Backtester is primarily for preliminary, low-granularity testing. It's results can be validated by comparing against the Event-Driven core.

1. **Event-Driven Backtester (Core System):** Operates on a queue of discrete events (e.g., trade, quote update, corporate action). This is required to simulate:
   * **Race Conditions:** Correctly ordering market events and trading decisions.
   * **Time Dependencies:** The price of a new trade is dependent on the most recent market events.
2. **Vectorized Backtester (Optimization):** Operates on pre-aggregated bar data (e.g., daily OHLCV). It is faster but simplifies simulation, making the entire strategy logic execute simultaneously at the end of the bar.

### B. Data Integrity
The engine must enforce Point-in-Time (PIT) consistency throughout the simulation run to prevent look-ahead bias at all stages:
1. **Market Data:** When retrieving market data for day D, the system uses the price as of D adjusted by corporate actions effective before D.
2. **Fundamental/Alternative Data:** Any feature derived from restated data (e.g., earnings) must be mapped to its Knowledge Time ($T_{knowledge}$) and must only be visible in the simulation after $T_{knowledge}$.

$$\text{Feature}_{\text{simulated}} = \text{Feature}_{\text{actual}} \quad \text{ONLY IF} \quad T_{\text{sim}} \ge T_{\text{knowledge}}$$

3. **Universe:** The set of tradable assets must be defined by the Point-in-Time Universe Table (Section I.1.C).

### C. Full Portfolio Simulation Loop
For a complete and realistic simulation of the fund's PnL, the backtester must integrate the Portfolio Optimization step at every rebalancing interval.
At each simulation step $t$:
1. **Generate Forecast:** Calculate the combined $\text{Final\_Signal}_t$ (Section II.6).
2. **Estimate Risk/Cost:** Calculate the $\Sigma_t$ and $\Gamma$ models (Section III).
3. **Solve QP:** The engine calls the QP solver to determine the ideal weights $w_t$ based on the objective function $U(w)$ (Section III.1).
4. **Simulate Execution:** The trade list ($w_t - w_{t-1}$) is passed to a simulated Execution Algorithm (Section IV.2) which applies the market impact and slippage costs ($\Gamma$) to calculate the final realized execution price.
5. **Calculate PnL:** The realized PnL is calculated based on simulated trades and mark-to-market prices.

### D. Required Simulation Outputs
The primary output of the backtesting engine is a set of time-series vectors synchronized to a fixed rebalancing frequency (e.g., every 5 minutes, daily). These vectors form the foundation for external analysis.
1. **Position Vector ($w_t$):** The vector of realized capital allocations for all assets in the universe at the end of the rebalancing interval. This represents the actual holdings.
2. **Returns Vector ($\Delta P_t$):** The PnL generated over the interval, broken down into its key components:
   * **Gross Return:** PnL generated from price movement before any transaction costs. Required for calculating Gross Sharpe and isolating the true predictive power of the Alpha model.
   * **Transaction Costs ($\Gamma_t$):** The simulated costs and slippage (Section III.1.C) incurred during the rebalance.
   * **Net Return:** Gross Return minus Transaction Costs. This is the realized PnL.

### E. Model Selection Statistics (Refining Section II.5)
The engine's output time series vectors (Position, Gross Return, Net Return) must be distilled into actionable metrics for model selection, focusing on risk-adjusted, cost-aware, and factor-neutral performance.
| Category | Metric | Description |
|---|---|---|
| Return | Sharpe Ratio | $(\text{Mean Return} - \text{Risk-Free Rate}) / \text{Volatility}$. Measures risk-adjusted performance. |
| Risk | Maximum Drawdown (MDD) | The largest peak-to-trough decline during the simulation period. |
| Risk | Value-at-Risk (VaR) / Conditional VaR (CVaR) | Statistical measures of expected losses over a specified period at a given confidence level. |
| Quality | Information Ratio (IR) | $\text{Alpha} / \text{Tracking Error}$. Measures performance relative to a defined benchmark or factor model. |
| Efficiency | Gross vs. Net Sharpe | Compares the Sharpe Ratio calculated using the Gross Return vector (isolated alpha performance) to the Sharpe Ratio using the Net Return vector (realized performance). |
| Behavior | Turnover Rate | The total value of assets bought or sold over a period, expressed as a fraction of the portfolio value. |
| Exposure | Factor Betas | Regression of strategy returns against the factors in the Risk Model (e.g., Market, Value, Momentum). Essential to confirm the strategy is isolated to the residual ($\epsilon$). |

---

# Section III: Portfolio Optimization & Risk

## 1. The Objective Function

The goal is to determine the optimal capital allocation ($w$) by solving a **Quadratic Programming (QP)** problem.

We seek to **Maximize Utility ($U$)**:

$$ U(w) = \underbrace{w^T \mu}_{\text{Return}} - \underbrace{\frac{1}{2} \lambda w^T \Sigma w}_{\text{Risk Penalty}} - \underbrace{\Gamma(w, w_{t-1})}_{\text{Transaction Costs}} $$

### A. Component: Expected Return ($w^T \mu$)
The optimizer requires the **Total Expected Return** ($\mu$) for the asset. Since the Alpha models forecast **Residual Returns**, we must reconstruct the total expectation using the Risk Model factors:
$$ \mu_i = \sum (\beta_{i,k} \cdot E[f_k]) + E[\epsilon_i] $$
*   $E[\epsilon_i]$: The output from your Alpha model.
*   $E[f_k]$: Expected return of the factor (often 0 for pure alpha, or hedged out).

### B. Component: The Risk Penalty ($\frac{1}{2} \lambda w^T \Sigma w$)
*   $\lambda$: Risk aversion parameter.
*   $\Sigma$: The $N \times N$ **Covariance Matrix**.

### C. Component: Transaction Costs ($\Gamma(w, w_{t-1})$)
Costs must be modeled to prevent the optimizer from chasing marginal alpha that is eaten by fees/slippage. It generally consists of two parts:
1.  **Linear Costs (Spread/Commissions):** $k \sum |w_{t,i} - w_{t-1,i}|$
2.  **Quadratic Costs (Market Impact):** $c \sum (w_{t,i} - w_{t-1,i})^2$

In reality, market impact typically follows a "Square Root Law" or Power Law, incorporating this makes the optimization non-convex, requiring specialized solvers.
$$ \text{Impact Cost} \propto \sigma \cdot \text{ADV} \cdot \left( \frac{\text{Trade Size}}{\text{ADV}} \right)^{0.6} $$

The optimizer requires $w_{t-1}$ (current positions) to calculate the cost of moving to the new ideal portfolio $w_t$.

## 2. Risk Models: Estimating $\Sigma$

For large universes ($N > T$), the sample covariance matrix is ill-conditioned. **Structured Factor Models** are used.

$$ \Sigma_{\text{factor}} = B \Sigma_f B^T + D $$

**Model Selection:**
1.  **Fundamental Models (e.g., Barra):** Factors are human-interpretable (Growth, Value, Industry).
    *   *Pro:* Explainable risk.
    *   *Con:* Slow to react to new market regimes.
2.  **Statistical Models (e.g., PCA):** Factors are derived mathematically from price history.
    *   *Pro:* Adapts quickly to new correlations.
    *   *Con:* "Black box" factors; difficult to explain why risk increased.
3.  **Blended/Shrinkage Models:**
    *   Robust funds often blend Fundamental and Statistical models, or use Shrinkage Estimators (e.g., Ledoit-Wolf) which calculate a weighted average of the Sample Covariance and a structured target (like a Factor model) to balance bias and variance.

## 3. Constraints

### A. The Infeasibility Problem
Hard constraints (e.g., `Sum(w) == 0`) can cause the solver to fail entirely if the market moves wildly or data is imperfect.

### B. Soft Constraints
Instead of strictly enforcing limits, implement them as heavy penalties in the objective function using Lagrange multipliers (penalty scalars, $\rho$). This allows the solver to find a solution that *minimally violates* the constraint during extreme stress.

**Implementation:**
$$ \text{Minimize } \dots + \rho_{\text{leverage}} \times (\sum |w_i| - L)^2 $$

*   **Budget / Net Exposure:** Soft constraint targeting Dollar Neutral.
*   **Gross Leverage:** $\sum |w_i| \leq L$.
*   **Position Limits:** $w_i \cdot \text{Capital} \leq 0.01 \cdot \text{ADV}_i$.
*   **Turnover Constraint:** $\Theta = \rho_{\text{turnover}} \sum |w_t - w_{t-1}|$. This penalizes excessive trading. By tuning $\rho_{\text{turnover}}$, the fund controls the trade-off between alpha capture and trading activity.

## 4. Implementation (Solver)

**Stack:** Python `CVXPY` with solvers like `OSQP` (Open Source) or `Mosek`/`Gurobi` (Commercial).

If using more accurate power law constraints (e.g. market impact), since Power Laws are convex but not quadratic, standard QP solvers are insufficient. Use **SOCP (Second-Order Cone Programming)** solvers (e.g., Mosek, ECOS) to handle these realistic cost constraints.

---

# Section IV: Execution & Operations

## 1. The Execution Hierarchy
1.  **Parent Order:** Generated by the Portfolio Optimizer. ("Buy 50,000 shares MSFT").
2.  **Child Orders:** Generated by the Execution Engine ("Buy 100 shares now").

**Goal:** Minimize **Implementation Shortfall** (Difference between Arrival Price and Average Execution Price).

## 2. Execution Algorithms

### A. VWAP (Volume Weighted Average Price)
Executes orders proportional to historical volume profiles (Smile Curve). Useful for passive liquidation of positions over a full day.

### B. TWAP (Time Weighted Average Price)
Executes evenly over time. Requires randomization of intervals to prevent detection by predatory HFT algorithms.

### C. Implementation Shortfall (IS)
Optimizes the trade-off between **Market Impact** (trading too fast) and **Market Risk** (trading too slow). If the price moves in the favorable direction, the algo trades passively; if it moves away, the algo trades aggressively to complete the order.
*   **Calculation:**
    $$ IS = P_{\text{executed}} - P_{\text{arrival}} $$
    $$ IS = \text{Delay Cost} + \text{Trading Cost} + \text{Opportunity Cost (unfilled part)} $$

### D. Smart Order Routing (SOR)
Simply sending a large order to a single broker is inefficient. An **SOR** slices child orders across multiple venues (Lit exchanges like NYSE/NASDAQ, and Dark Pools) to find liquidity while minimizing information leakage.

## 3. The Order Management System (OMS)

The OMS acts as the state machine between the strategy and the broker.

### A. State Management
Orders transition through states: `PENDING_SUBMIT` $\to$ `SUBMITTED` $\to$ `PARTIALLY_FILLED` $\to$ `FILLED` (or `CANCELLED`). The system must handle race conditions (e.g., receiving a Fill after sending a Cancel).

### B. Reconciliation
Discrepancies between internal ledgers and broker records are inevitable.
*   **Process:** At T+0, query broker positions (Source of Truth), overwrite internal records and report discrepancy for manual review.

### C. Safety Limits
1.  **Fat Finger:** Reject orders $>$ $X$% of ADV.
2.  **Message Rate:** Stop trading if order count exceeds thresholds.
3.  **Kill Switch:** Liquidate positions if Intraday Drawdown > Limit.
4.  **Price Collar:** Reject trades significantly outside the current BBO.

## 4. Connectivity (FIX vs. REST)

*   **REST APIs:** Simple, but higher latency.
*   **FIX Protocol:** The industry standard. Socket-based, streaming text messages.

## 5. Post-Trade Analysis & Attribution

### A. TCA (Transaction Cost Analysis)
Compare execution against benchmarks:
*   **Arrival Price:** Price when the decision was made.
*   **Interval VWAP:** Volume-weighted price during the trade window.
*   **Slippage:** Decomposed into Spread Capture, Market Impact, and Alpha Decay (did the price move because we traded, or because the market moved?).

### B. PnL Attribution Breakdown
1.  **Factor/Sector/Region Attribution:** Linear decomposition.
2.  **Alpha Model Attribution:** Calculate contribution as $\text{PnL}_{\text{Sim A}} - \text{PnL}_{\text{Sim B}}$ (Marginal Simulation; Full Portfolio - Portfolio without Model X).

---
