# Alt/BTC Ratio Trend-Following Trading System

An institutional-grade cryptocurrency trading system implementing trend-following strategy with Walk-Forward Optimization (WFO) validation for the Alt/BTC ratio.

## Project Overview

- **Strategy**: Trend-following on Alt/BTC ratio with Z-score signal generation
- **Universe**: Multi-symbol support (ETH, SOL, XRP, DOGE, ADA, DOT, MATIC, LTC) + BTC hedge
- **Portfolio**: Risk Parity allocation with Beta-based hedging
- **Validation**: Walk-Forward Optimization (WFO) with IS/OOS testing
- **Execution**: CLI interface for backtest and live trading modes

## Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Backtest (Real Data from Binance)

```bash
# Basic backtest with real data
python -m src.cli --mode backtest --config config/settings.yaml

# Backtest with specific date range
python -m src.cli --mode backtest \
  --config config/settings.yaml \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --capital 100000.0

# Force fresh data (skip cache)
python -m src.cli --mode backtest \
  --config config/settings.yaml \
  --no-cache

# Verbose output with detailed logging
python -m src.cli --mode backtest \
  --config config/settings.yaml \
  --verbose
```

### Live Trading Mode

```bash
python -m src.cli --mode live --config config/settings.yaml
```

### Dry Run (Simulation without execution)

```bash
python -m src.cli --mode backtest --dry-run --config config/settings.yaml
```

### Python Script Usage

```python
from src.config import load_config
from src.data_loader import DataLoader
from src.strategy import Strategy
from src.portfolio import PortfolioOptimizer
from src.backtester import Backtester

# Load configuration
config = load_config("config/settings.yaml")

# Fetch real data
loader = DataLoader()
btc_data = loader.fetch_ohlcv("BTC/USDT", limit=820)  # 730 IS + 90 OOS
eth_data = loader.fetch_ohlcv("ETH/USDT", limit=820)

# Initialize components
strategy = Strategy(config)
portfolio = PortfolioOptimizer()
backtester = Backtester(config, starting_capital=100000.0)

# Run backtest
results = backtester.run_wfo(btc_data, strategy, portfolio)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Go/No-Go: {'PASS' if results['is_valid'] else 'FAIL'}")
```

## Project Structure

```
crypt4/
├── config/
│   └── settings.yaml           # Configuration file
├── results/                    # Output directory for backtest results
├── src/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point and orchestration
│   ├── config.py               # Configuration loading and validation
│   ├── data_loader.py          # OHLCV data loading and caching
│   ├── strategy.py             # Signal generation and regime detection
│   ├── portfolio.py            # Portfolio optimization and rebalancing
│   ├── backtester.py           # WFO engine and metrics calculation
│   ├── executor.py             # Order execution and position tracking
│   └── utils.py                # Utilities (logging, timestamps, backoff)
│
├── tests/
│   ├── test_cli.py             # CLI tests (28 tests)
│   ├── test_config.py          # Config tests (12 tests)
│   ├── test_data_loader.py     # DataLoader tests (20 tests)
│   ├── test_strategy.py        # Strategy tests (19 tests)
│   ├── test_portfolio.py       # Portfolio tests (16 tests)
│   ├── test_backtester.py      # Backtester tests (32 tests)
│   ├── test_executor.py        # Executor tests (35 tests)
│   ├── test_integration.py     # Integration tests (24 tests)
│   └── test_utils.py           # Utils tests (14 tests)
│
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── TESTING.md                  # Testing documentation
└── .gitignore
```

## Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Module Tests

```bash
python -m pytest tests/test_strategy.py -v
```

### Generate Coverage Report

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Statistics

- **Total Tests**: 200
- **Pass Rate**: 100%
- **Coverage**: 75%
- **Execution Time**: ~11 seconds

## Configuration

Configure strategy parameters in `config/settings.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `z_entry` | Entry signal Z-score threshold | 2.0 |
| `z_target` | Target exit Z-score | 1.0 |
| `z_stop` | Stop loss Z-score | 3.0 |
| `trend_window` | Long-term MA window (days) | 200 |
| `short_window` | Short-term MA window (days) | 20 |
| `fee_pct` | Trading fee percentage | 0.1% |
| `slippage_pct` | Slippage percentage | 0.05% |

See [config/settings.yaml](config/settings.yaml) for detailed configuration.

## Strategy Specification

### Market Regime Detection

- **Bullish**: `Ratio > MA_Trend`
- **Bearish**: `Ratio < MA_Trend`
- **Neutral**: `Ratio ≈ MA_Trend` (maintains previous regime)

### Signal Generation

#### Bullish Regime
- Entry: `Z_Score < -z_entry` (LONG)
- Take Profit: `Z_Score > z_target`
- Stop Loss: `abs(Z_Score) > z_stop`

#### Bearish Regime
- Entry: `Z_Score > z_entry` (SHORT)
- Take Profit: `Z_Score < -z_target`
- Stop Loss: `abs(Z_Score) > z_stop`

### Portfolio Construction

- **Risk Parity**: Inverse volatility weighting (monthly rebalance)
- **Beta Hedge**: Grid search for optimal hedge ratio (quarterly update)

### Walk-Forward Optimization

- In-Sample: 730 days (2 years)
- Out-of-Sample: 90 days (3 months)
- Rolling: 90-day shift
- Go/No-Go Criteria: Sharpe > 1.0 AND Max Drawdown < 20%

## Module Documentation

### src/cli.py
- CLI argument parsing and application orchestration
- Supports backtest and live modes
- 28 test cases, 43% coverage

### src/strategy.py
- Regime detection and Z-score signal generation
- Handles bullish, bearish, and neutral markets
- 19 test cases, 75% coverage

### src/portfolio.py
- Risk Parity portfolio allocation
- Beta-based hedging optimization
- 16 test cases, 69% coverage

### src/backtester.py
- Walk-Forward Optimization engine
- Sharpe ratio and max drawdown metrics
- Go/No-Go validation (Sharpe > 1.0, DD < 20%)
- 32 test cases, 87% coverage

### src/executor.py
- Order execution with fee and slippage modeling
- Position tracking and P&L calculation
- Trade history logging
- 35 test cases, 78% coverage

### src/data_loader.py
- OHLCV data loading and validation
- CCXT integration for real exchange data (Binance)
- Automatic caching with parquet format
- Gap interpolation (max 3 consecutive)
- Exponential backoff retry logic (3 attempts)
- 20 test cases, 87% coverage

### src/config.py
- Configuration file loading and validation
- Section validation (strategy, execution, universe, validation)
- 12 test cases, 76% coverage

### src/utils.py
- AuditLogger: CSV-based audit logging with chmod 600
- ExponentialBackoff: Configurable retry delay
- get_utc_timestamp(): UTC timestamp generation
- 14 test cases, 93% coverage

### tests/test_integration.py
- End-to-end pipeline validation
- Component interaction testing
- Error propagation verification
- 24 test cases

## Data Source

### CCXT Integration

The system uses **CCXT (CryptoCurrency eXchange Trading)** to fetch real OHLCV data from Binance:

**Features**:
- Real-time data from Binance spot market
- Automatic rate limiting (respects API constraints)
- Transparent error handling with retries
- Parquet-based caching for performance
- Fallback to cache on network failures

**Usage**:

```python
from src.data_loader import DataLoader

loader = DataLoader()

# Fetch real data from Binance (uses cache if available)
btc_data = loader.fetch_ohlcv("BTC/USDT", limit=730, use_cache=True)
eth_data = loader.fetch_ohlcv("ETH/USDT", limit=730, use_cache=True)

# Force fresh data (no cache)
btc_fresh = loader.fetch_ohlcv("BTC/USDT", limit=730, use_cache=False)
```

**Supported Symbols**:
- BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT, ADA/USDT, DOT/USDT, MATIC/USDT, LTC/USDT
- Any other symbol available on Binance spot market

**Caching**:
- Data cached in `data/raw/{SYMBOL}.parquet`
- Automatic fallback to cache if network fails
- Configurable via `use_cache` parameter

---

## Error Handling

### Data Loading Errors
- Retry logic: Up to 3 attempts with exponential backoff (1s → 2s → 4s, max 30s)
- Cache fallback: Uses cached data if all retries fail
- Gap interpolation: Up to 3 consecutive missing periods (forward fill)
- Validation: Comprehensive OHLCV data checks (NaN, Inf, price relationships)

### Execution Errors
- Insufficient balance detection
- Invalid symbol handling
- Position consistency validation

### Configuration Errors
- Section validation (required vs optional)
- Parameter type checking
- Default value fallbacks

## Logging

The system uses Python's standard logging module:

```
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

**Log Levels**:
- **DEBUG**: Detailed execution flow
- **INFO**: Major milestones
- **WARNING**: Potential issues
- **ERROR**: Error conditions

## Future Enhancements

1. Multiple timeframes for signal confirmation
2. Machine learning-based regime detection
3. Advanced risk metrics (VaR, Sortino ratio)
4. Full exchange integration for live trading
5. Tick-level backtesting with realistic fills

## License

MIT License

## Support

For issues, questions, or feature requests, please open an issue on GitHub or refer to TESTING.md for detailed testing documentation.
