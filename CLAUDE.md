# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**crypt4** is an institutional-grade cryptocurrency trading system that implements a trend-following strategy based on the Alt/BTC ratio. The system uses Z-score signal generation within detected market regimes (bullish/bearish) to identify entry and exit points, with Walk-Forward Optimization (WFO) for rigorous strategy validation.

**Key Characteristics:**
- Multi-symbol portfolio (8 altcoins + BTC hedge)
- Risk Parity allocation with Beta-based hedging
- Real OHLCV data from Binance via CCXT
- Comprehensive backtest validation with Go/No-Go criteria (Sharpe > 1.0, Max DD < 20%)
- Modular pipeline architecture with clear separation of concerns

## Essential Development Commands

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v

# Run tests for a specific module
pytest tests/test_strategy.py -v

# Run a single test
pytest tests/test_strategy.py::TestStrategy::test_detect_regime_bullish -v

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# Run only integration tests
pytest tests/ -m integration -v

# Run tests excluding slow tests
pytest tests/ -m "not slow" -v
```

### Executing the System

```bash
# Activate virtual environment
source venv/bin/activate

# Run backtest with cached data
python -m src --mode backtest --config config/settings.yaml

# Run backtest with fresh data (bypass cache)
python -m src --mode backtest --config config/settings.yaml --no-cache

# Run with custom date range and capital
python -m src --mode backtest --config config/settings.yaml \
  --start-date 2023-01-01 --end-date 2024-12-31 --capital 100000.0

# Live mode (data fetching only, no order execution)
python -m src --mode live --config config/settings.yaml

# Dry run (simulation without execution)
python -m src --mode backtest --dry-run --config config/settings.yaml
```

## Architecture Overview

### System Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ CLI Entry Point (src/cli.py)                                    │
│ - Argument parsing                                              │
│ - Application orchestration                                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│ Configuration (src/config.py)                                   │
│ - Load settings.yaml (strategy params, portfolio, WFO config)  │
│ - Validate sections and parameter types                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│ Data Loader (src/data_loader.py)                               │
│ - Fetch OHLCV from Binance via CCXT                            │
│ - Cache in data/raw/*.parquet                                  │
│ - Retry logic (exponential backoff, 3 attempts)                │
│ - Gap interpolation (max 3 consecutive)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│ Strategy Module (src/strategy.py)                              │
│ - Detect market regime (Bullish/Bearish/Neutral)               │
│ - Calculate Z-scores for signal generation                     │
│ - Generate Entry/Exit/StopLoss signals                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│ Portfolio Optimizer (src/portfolio.py)                         │
│ - Risk Parity weighting (inverse volatility)                   │
│ - Beta-based hedge ratio optimization                          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                    ┌──┴──────────────────────┐
                    │                         │
        ┌───────────▼────────────┐   ┌───────▼──────────────┐
        │ Backtester (backtest)  │   │ Executor (live)      │
        │ - WFO engine           │   │ - Order execution    │
        │ - Metrics calculation  │   │ - Position tracking  │
        │ - Go/No-Go validation  │   │ - P&L calculation    │
        └────────────────────────┘   └──────────────────────┘
```

### Walk-Forward Optimization (WFO) Architecture

The core validation methodology splits data into rolling windows:

- **In-Sample (IS)**: 730 days (2 years) - used to identify/optimize strategy parameters
- **Out-of-Sample (OOS)**: 90 days (3 months) - used to validate performance with unseen data
- **Rolling Shift**: 90 days - windows advance by 3 months at a time
- **Go/No-Go Criteria**: OOS Sharpe > 1.0 AND Max Drawdown < 20% (prevents overfitting)

Each WFO window: ISO period determines parameters → OOS period validates with Go/No-Go check.

## Module Responsibilities

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| **cli.py** | Entry point, orchestration | `CliApp`, `main()` |
| **config.py** | Load/validate settings.yaml | `load_config()`, `Config` |
| **data_loader.py** | Fetch, cache, validate OHLCV | `DataLoader`, `ExponentialBackoff` |
| **strategy.py** | Regime detection, signal generation | `Strategy`, `Signal`, `Regime` enum |
| **portfolio.py** | Weight optimization, hedging | `PortfolioOptimizer`, `RiskParityWeights` |
| **backtester.py** | WFO engine, metrics | `Backtester`, `walk_forward_optimize()` |
| **executor.py** | Order execution, P&L tracking | `Executor`, `Order`, `Position` |
| **utils.py** | Logging, timestamps, retry logic | `AuditLogger`, `ExponentialBackoff`, `get_utc_timestamp()` |

## Configuration System (config/settings.yaml)

The YAML configuration file drives all system behavior. Key sections:

```yaml
universe:          # Asset universe (altcoins + BTC hedge)
strategy:          # Z-score parameters (z_entry, z_target, z_stop)
portfolio:         # Weight method (risk_parity) and rebalance frequency
wfo:              # WFO parameters (730d IS, 90d OOS, 90d shift)
execution:        # Mode (backtest/live), fees, slippage, timezone
data:             # CCXT settings, retry logic, gap interpolation
validation:       # Go/No-Go thresholds (Sharpe > 1.0, DD < 20%)
logging:          # Audit log directory and permissions
```

**Configuration Flow**: `settings.yaml` → `Config` object → passed to all modules for parameter access.

## Testing Strategy

### Organization & Methodology

**Structure**: 200 tests across 9 test modules (75% code coverage, 11 second execution)

**Test Naming Convention** (TC-N: Normal, TC-B: Boundary, TC-A: Abnormal):
```python
def test_detect_regime_bullish_tc_n_typical_case(self):
    """TC-N: Normal bullish regime detection"""

def test_zscore_calculation_tc_b_zero_stddev(self):
    """TC-B: Boundary case with zero standard deviation"""

def test_order_execution_tc_a_insufficient_balance(self):
    """TC-A: Abnormal case - error handling"""
```

**BDD Style** (Given-When-Then):
```python
# Given: initial state
strategy = Strategy(config)
# When: action
signal = strategy.generate_signal(data)
# Then: assertion
assert signal.type == SignalType.ENTRY
```

**Pytest Markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

**Coverage by Module**:
- utils.py: 93% | backtester.py: 87% | data_loader.py: 87% | executor.py: 78%
- config.py: 76% | strategy.py: 75% | portfolio.py: 69% | cli.py: 43%

## Key Development Patterns

### 1. Modular Pipeline Design
Each module has a single responsibility and passes data forward:
- No circular dependencies
- Modules are independently testable
- Configuration injected via constructors

### 2. Value Objects for Type Safety
Use explicit value objects instead of dicts/tuples:
```python
Signal(type=SignalType.ENTRY, z_score=2.5, regime=Regime.BULLISH)
Order(symbol='ETH/USDT', side='BUY', quantity=10, price=1234.56)
Position(symbol='ETH/USDT', quantity=10, entry_price=1234.56, pnl=100)
```

### 3. Enum-Based State Management
Use Python enums for regime/signal types:
```python
class Regime(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
```

### 4. Defensive Programming
- Validate inputs in constructors with explicit error messages
- Use guard clauses for None checks
- Raise typed exceptions (ValueError, TypeError, RuntimeError)

### 5. CCXT Data Integration Pattern
- Fetch from Binance with `DataLoader.fetch_ohlcv(symbol, limit)`
- Automatic cache in `data/raw/{symbol}.parquet`
- Fallback to cache on network failures
- Exponential backoff retry (1s → 2s → 4s, max 30s)

### 6. Audit Logging
- AuditLogger writes CSV with secure permissions (chmod 600)
- Log all trading decisions with context (z_score, regime, reason)
- UTC timestamps for all events

### 7. WFO Validation Pattern
```python
# Core pattern used in backtester.py
for is_period, oos_period in wfo_windows:
    # Optimize on IS data
    best_params = optimize_on(is_period)
    # Validate on OOS data
    metrics = evaluate_on(oos_period, best_params)
    # Check Go/No-Go criteria
    if metrics.sharpe > 1.0 and metrics.max_dd < 0.20:
        is_valid = True  # Pass validation
```

## Development Workflow Tips

1. **Test First**: Write tests before implementation. Use TC-N/TC-B/TC-A for comprehensive coverage.
2. **Config-Driven Changes**: Modify behavior via settings.yaml when possible, not code.
3. **Data Caching**: Use `--no-cache` flag only when testing fresh data fetching.
4. **Error Messages**: Always include context (symbol, z_score, regime) in exceptions.
5. **Coverage**: Aim for >80% coverage. Check `htmlcov/index.html` after running tests.
6. **Git Commits**: Include strategy/data flow context in commit messages for future reference.

## Critical Files to Understand

To understand how the system works end-to-end:
1. [src/cli.py](src/cli.py) - Entry point, orchestrates all modules
2. [src/backtester.py](src/backtester.py) - WFO engine (core algorithm)
3. [src/strategy.py](src/strategy.py) - Regime detection and signal logic
4. [config/settings.yaml](config/settings.yaml) - All configurable parameters
5. [tests/test_integration.py](tests/test_integration.py) - End-to-end flow examples
