# Implementation Plan: Alt/BTC Trend Follow Ratio Strategy

# Goal Description
Build an automated trading system based on the Alt/BTC ratio trend following strategy using **Test-Driven Development (TDD)**. The system will fetch data, calculate indicators, optimize portfolio weights, and execute backtests.

## User Review Required
> [!IMPORTANT]
> **TDD Approach**: We will write tests *before* implementing the core logic.
> **API Keys**: Public APIs (CCXT) will be used initially.
> **Data Storage**: Local CSV/Parquet in `data/`.

## Proposed Changes

### Directory Structure
```
crypt4/
├── config/
│   └── settings.yaml       # Configuration (symbols, windows, thresholds)
├── data/
│   ├── raw/                # Raw OHLCV data
│   └── processed/          # Cleaned/Merged data
├── logs/                   # Audit logs
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data fetching (CCXT), validation, interpolation
│   ├── strategy.py         # Signal generation, Regime detection
│   ├── portfolio.py        # Risk Parity, Beta Hedge optimization
│   ├── backtester.py       # WFO engine, Metrics calculation
│   ├── executor.py         # Order generation (Mock for now)
│   └── utils.py            # Logging, Error handling
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_strategy.py
│   ├── test_portfolio.py
│   ├── test_backtester.py
│   └── test_integration.py
├── main.py                 # CLI Entry point
├── requirements.txt        # Dependencies
└── README.md
```

### [Module] Data Ingestion (`src/data_loader.py`)
#### [NEW] `DataLoader` Class
-   **TDD Specs (`tests/test_data_loader.py`)**:
    -   `test_fetch_ohlcv_retry`: Mock CCXT network error, verify retry logic (3 times) and backoff.
    -   `test_validate_data_interpolation`: Provide data with gaps, verify `ffill` works for <3 gaps and raises error for >3 gaps.
    -   `test_initial_fetch`: Verify fetching 2 years of data on first run.

### [Module] Strategy Logic (`src/strategy.py`)
#### [NEW] `Strategy` Class
-   **TDD Specs (`tests/test_strategy.py`)**:
    -   `test_regime_detection`: Verify Bullish/Bearish logic based on Ratio vs MA_Trend.
    -   `test_signal_generation`: Verify Entry/Exit signals based on Z-Score thresholds.
    -   `test_signal_conflict`: Verify Sell signal priority when Buy/Sell occur simultaneously.
    -   `test_regime_change_close`: Verify "Close All" signal generation upon regime switch.

### [Module] Portfolio Construction (`src/portfolio.py`)
#### [NEW] `PortfolioOptimizer` Class
-   **TDD Specs (`tests/test_portfolio.py`)**:
    -   `test_risk_parity_weights`: Verify weights sum to 1.0 and are inversely proportional to volatility.
    -   `test_hedge_ratio_grid_search`: Verify selection of best hedge ratio from [0.8, 0.9, 1.0, 1.1, 1.2] * beta.

### [Module] Backtesting (`src/backtester.py`)
#### [NEW] `Backtester` Class
-   **TDD Specs (`tests/test_backtester.py`)**:
    -   `test_wfo_splitting`: Verify correct splitting of In-Sample (2y) and Out-of-Sample (3m) periods with rolling.
    -   `test_metrics_calculation`: Verify Sharpe Ratio and Max Drawdown calculations against known values.

### [Module] Configuration & Utils
-   **`config/settings.yaml`**: Define default parameters (UTC timezone, 0.11% fee, etc.).
-   **`src/utils.py`**: Implement logging with `chmod 600` permission setting.

## Verification Plan

### Automated Tests (TDD)
-   Run `pytest` continuously.
-   **Coverage Goal**: > 90% branch coverage for core logic.

### Manual Verification
-   **Visual Inspection**: Plot indicators and signals.
-   **Log Review**: Check audit logs for correct format and permission settings.
