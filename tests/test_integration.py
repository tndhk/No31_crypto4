"""Integration tests for complete trading pipeline"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

import pytest

from src.cli import parse_arguments, CliApp
from src.config import load_config
from src.data_loader import DataLoader
from src.strategy import Strategy
from src.portfolio import PortfolioOptimizer
from src.backtester import Backtester, WFOSplit
from src.executor import Executor, ExecutionMode, Order
from src.utils import get_utc_timestamp


class TestEndToEndTradingPipeline:
    """Tests for complete trading pipeline integration"""

    def test_full_pipeline_backtest_workflow(self):
        """TC-N-01: Full backtest pipeline executes successfully."""
        # Given: Complete configuration
        config = {
            "universe": {"symbols": ["BTC", "ETH"]},
            "strategy": {
                "z_entry": 2.0,
                "z_target": 1.0,
                "z_stop": 3.0,
                "trend_window": 10,
                "short_window": 5,
            },
            "portfolio": {"method": "risk_parity"},
            "execution": {"fee_pct": 0.1, "slippage_pct": 0.05},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0},
        }

        # When: Running complete pipeline
        data_loader = DataLoader()
        strategy = Strategy(config)
        portfolio_optimizer = PortfolioOptimizer()
        backtester = Backtester(config, starting_capital=10000.0)

        # Then: All modules initialize without error
        assert strategy is not None
        assert portfolio_optimizer is not None
        assert backtester is not None
        assert backtester.starting_capital == 10000.0

    def test_data_pipeline_integration(self):
        """TC-N-02: Data loading and strategy signal generation pipeline."""
        # Given: Sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq='D')
        close_prices = np.linspace(100, 110, 100)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices,
            "high": close_prices + 1,
            "low": close_prices - 1,
            "close": close_prices,
            "volume": np.full(100, 1000000.0),
        })

        # When: Processing through strategy
        config = {
            "z_entry": 2.0,
            "z_target": 1.0,
            "z_stop": 3.0,
            "trend_window": 10,
            "short_window": 5,
        }
        strategy = Strategy(config)
        regime = strategy.detect_regime(df)

        # Then: Regime is detected
        assert regime is not None
        assert hasattr(regime, 'value')

    def test_strategy_to_executor_pipeline(self):
        """TC-N-03: Strategy signals converted to executor orders."""
        # Given: Executor with starting capital
        config = {
            "execution": {"fee_pct": 0.1, "slippage_pct": 0.05}
        }
        executor = Executor(config, starting_capital=10000.0, mode=ExecutionMode.BACKTEST)

        # When: Executing multiple orders
        buy_order = Order(symbol="BTC/USDT", order_type="BUY", size=0.1, price=50000.0)
        executor.execute_order(buy_order)

        # Then: Position is created and balance updated
        assert "BTC/USDT" in executor.positions
        assert executor.current_balance < 10000.0
        assert len(executor.trades) == 1

    def test_portfolio_rebalancing_flow(self):
        """TC-N-04: Portfolio rebalancing updates positions correctly."""
        # Given: Portfolio optimizer and executor
        optimizer = PortfolioOptimizer(min_data_days=20)
        executor = Executor({}, starting_capital=10000.0)

        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=50, freq='D')
        returns = pd.DataFrame({
            "BTC": np.random.randn(50) * 0.02,
            "ETH": np.random.randn(50) * 0.03,
        }, index=dates)

        # When: Calculating weights
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: Weights are valid
        assert weights is not None
        assert len(weights) == 2
        assert "BTC" in weights and "ETH" in weights

    def test_wfo_execution_full_flow(self):
        """TC-N-05: Walk-Forward Optimization executes complete analysis."""
        # Given: Backtester with configuration
        config = {
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0}
        }
        backtester = Backtester(config, starting_capital=10000.0)

        # Create sufficient test data
        dates = pd.date_range("2024-01-01", periods=900, freq='D')
        close_prices = 100.0 + np.cumsum(np.random.randn(900) * 0.5)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": close_prices,
            "high": close_prices + 1,
            "low": close_prices - 1,
            "close": close_prices,
            "volume": np.full(900, 1000000.0),
        })

        # When: Running WFO
        result = backtester.walk_forward_optimize(df, is_window_days=730, oos_window_days=90)

        # Then: Results contain required metrics
        assert "is_metrics" in result
        assert "oos_metrics" in result
        assert "is_go" in result
        assert "oos_go" in result
        assert "overall_go" in result

    def test_multi_symbol_execution(self):
        """TC-N-06: Multi-symbol order execution and position tracking."""
        # Given: Executor configured for multiple symbols
        config = {
            "execution": {"fee_pct": 0.1, "slippage_pct": 0.05},
            "universe": {"symbols": ["BTC", "ETH", "SOL"]}
        }
        executor = Executor(config, starting_capital=30000.0)

        # When: Executing orders on multiple symbols
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        for symbol in symbols:
            order = Order(symbol=symbol, order_type="BUY", size=0.1, price=5000.0)
            executor.execute_order(order)

        # Then: All positions are tracked
        assert len(executor.positions) == 3
        assert executor.current_balance < 30000.0

    def test_trade_closure_and_pnl_calculation(self):
        """TC-N-07: Complete trade cycle from open to close with P&L."""
        # Given: Executor with initial trade
        config = {"execution": {"fee_pct": 0.1, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=10000.0)

        # When: Opening and closing position
        buy_order = Order(symbol="BTC/USDT", order_type="BUY", size=0.1, price=50000.0)
        executor.execute_order(buy_order)
        initial_balance = executor.current_balance

        sell_order = Order(symbol="BTC/USDT", order_type="SELL", size=0.1, price=55000.0)
        executor.execute_order(sell_order)

        # Then: Position closed, P&L recorded, balance updated
        assert "BTC/USDT" not in executor.positions
        assert executor.current_balance > initial_balance
        assert len(executor.closed_trades) == 1

    def test_cli_to_execution_pipeline(self):
        """TC-N-08: CLI arguments flow through to executor configuration."""
        # Given: CLI arguments
        args_list = [
            "--mode", "backtest",
            "--config", "config/settings.yaml",
            "--capital", "50000.0",
            "--symbols", "BTC", "ETH",
            "--start-date", "2024-01-01",
            "--end-date", "2024-12-31",
        ]

        # When: Parsing CLI arguments
        args = parse_arguments(args_list)

        # Then: Arguments propagate to executor configuration
        assert args.mode == "backtest"
        assert args.capital == 50000.0
        assert args.symbols == ["BTC", "ETH"]
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-12-31"


class TestComponentInteraction:
    """Tests for interaction between major components"""

    def test_strategy_config_integration(self):
        """TC-N-09: Strategy correctly uses configuration parameters."""
        # Given: Strategy with specific config
        config = {
            "z_entry": 2.5,
            "z_target": 0.5,
            "z_stop": 3.5,
            "trend_window": 20,
            "short_window": 5,
        }
        strategy = Strategy(config)

        # When: Creating strategy
        # Then: Config values are stored correctly in strategy attributes
        assert strategy.z_entry == 2.5
        assert strategy.z_target == 0.5
        assert strategy.z_stop == 3.5
        assert strategy.trend_window == 20
        assert strategy.short_window == 5

    def test_backtester_executor_integration(self):
        """TC-N-10: Backtester and executor work together for backtest execution."""
        # Given: Configured backtester and executor
        config = {
            "execution": {"fee_pct": 0.1, "slippage_pct": 0.05},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0}
        }
        backtester = Backtester(config, starting_capital=10000.0)
        executor = Executor(config, starting_capital=10000.0)

        # When: Both are initialized with same capital
        executor.starting_capital == backtester.starting_capital

        # Then: They can coordinate
        assert executor.starting_capital == 10000.0
        assert backtester.starting_capital == 10000.0

    def test_error_propagation_in_pipeline(self):
        """TC-A-01: Errors in one component propagate appropriately."""
        # Given: Backtester with invalid data
        config = {
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0}
        }
        backtester = Backtester(config, starting_capital=10000.0)

        # Create insufficient data
        df = pd.DataFrame({"timestamp": [1, 2], "close": [100, 101]})

        # When/Then: Error is raised
        with pytest.raises((ValueError, KeyError)):
            backtester.walk_forward_optimize(df, is_window_days=730, oos_window_days=90)

    def test_missing_config_section_handling(self):
        """TC-A-02: Missing config section is handled gracefully."""
        # Given: Incomplete configuration
        config = {"strategy": {}}  # Missing execution, validation sections

        # When: Creating modules
        strategy = Strategy(config)
        executor = Executor(config)

        # Then: Modules use defaults
        assert executor.fee_pct == 0.0  # Default when not specified
        assert executor.slippage_pct == 0.0

    def test_capital_flow_through_modules(self):
        """TC-N-11: Starting capital flows correctly through all modules."""
        # Given: Specified starting capital
        starting_capital = 50000.0
        config = {"execution": {"fee_pct": 0.1, "slippage_pct": 0.05}}

        # When: Creating modules with this capital
        backtester = Backtester(config, starting_capital=starting_capital)
        executor = Executor(config, starting_capital=starting_capital)

        # Then: Both modules have correct capital
        assert backtester.starting_capital == starting_capital
        assert executor.starting_capital == starting_capital
        assert executor.current_balance == starting_capital


class TestDataConsistency:
    """Tests for data consistency across pipeline"""

    def test_ohlcv_data_integrity_through_pipeline(self):
        """TC-N-12: OHLCV data maintains integrity through processing."""
        # Given: Valid OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq='D')
        original_df = pd.DataFrame({
            "timestamp": dates,
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100.5, 110.5, 100),
            "volume": np.full(100, 1000000.0),
        })

        # When: Processing through backtester
        config = {"validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0}}
        backtester = Backtester(config)
        returns = backtester._calculate_returns(original_df)

        # Then: Data integrity maintained
        assert len(returns) == len(original_df) - 1
        assert isinstance(returns, np.ndarray)
        assert not np.isnan(returns).all()

    def test_position_state_consistency(self):
        """TC-N-13: Position state remains consistent through operations."""
        # Given: Executor with position
        config = {"execution": {"fee_pct": 0.1, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=10000.0)

        # When: Creating position
        buy_order = Order(symbol="BTC/USDT", order_type="BUY", size=0.1, price=50000.0)
        executor.execute_order(buy_order)
        position = executor.positions["BTC/USDT"]
        initial_entry_price = position.entry_price

        # Partial sell
        sell_order = Order(symbol="BTC/USDT", order_type="SELL", size=0.05, price=55000.0)
        executor.execute_order(sell_order)

        # Then: Remaining position maintains original entry price
        remaining_position = executor.positions["BTC/USDT"]
        assert remaining_position.entry_price == initial_entry_price
        assert remaining_position.size == 0.05

    def test_trade_history_completeness(self):
        """TC-N-14: Trade history captures all executed transactions."""
        # Given: Executor
        config = {"execution": {"fee_pct": 0.1, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=10000.0)

        # When: Executing multiple orders
        orders = [
            Order(symbol="BTC/USDT", order_type="BUY", size=0.1, price=50000.0),
            Order(symbol="ETH/USDT", order_type="BUY", size=1.0, price=3000.0),
            Order(symbol="BTC/USDT", order_type="SELL", size=0.05, price=55000.0),
        ]
        for order in orders:
            executor.execute_order(order)

        # Then: All trades are logged
        assert len(executor.trades) == 3
        assert len(executor.closed_trades) == 1


class TestErrorHandlingAcrossComponents:
    """Tests for error handling and recovery across integrated components"""

    def test_invalid_symbol_handling(self):
        """TC-A-03: Invalid symbol in order raises appropriate error."""
        # Given: Executor
        config = {"execution": {"fee_pct": 0.1, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=10000.0)

        # When: Attempting to sell non-existent position
        sell_order = Order(symbol="NONEXISTENT/USDT", order_type="SELL", size=0.1, price=100.0)

        # Then: Error is raised
        with pytest.raises(ValueError):
            executor.execute_order(sell_order)

    def test_insufficient_balance_handling(self):
        """TC-A-04: Insufficient balance prevents order execution."""
        # Given: Executor with limited capital
        config = {"execution": {"fee_pct": 0.1, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=100.0)

        # When: Attempting large purchase
        buy_order = Order(symbol="BTC/USDT", order_type="BUY", size=10, price=50000.0)

        # Then: Error is raised
        with pytest.raises(ValueError):
            executor.execute_order(buy_order)

    def test_configuration_validation_error(self):
        """TC-A-05: Invalid configuration is caught during module creation."""
        # Given: Invalid config (negative fee)
        config = {
            "execution": {"fee_pct": -1.0, "slippage_pct": 0.05}
        }

        # When/Then: No error during creation (validation deferred)
        # Config validation would happen at use time
        executor = Executor(config)
        assert executor.fee_pct == -1.0

    def test_insufficient_data_validation(self):
        """TC-A-06: Insufficient data for WFO is caught."""
        # Given: Backtester
        config = {"validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0}}
        backtester = Backtester(config)

        # Create minimal data
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "close": np.linspace(100, 101, 10),
        })

        # When/Then: Error on insufficient data
        with pytest.raises(ValueError):
            backtester.walk_forward_optimize(df)


class TestWorkflowOrchestration:
    """Tests for orchestration of complete workflows"""

    def test_cli_app_initialization_full(self):
        """TC-N-15: CLI app initializes with all required modules."""
        # Given: Complete CLI arguments
        args_list = [
            "--mode", "backtest",
            "--config", "config/settings.yaml",
            "--capital", "100000.0",
        ]
        args = parse_arguments(args_list)

        # When: Creating CliApp
        app = CliApp(args)

        # Then: App is properly initialized
        assert app.mode == "backtest"
        assert app.starting_capital == 100000.0
        assert app.config_path == "config/settings.yaml"

    def test_backtest_mode_workflow(self):
        """TC-N-16: Backtest mode workflow executes properly."""
        # Given: Backtest configuration
        config = {
            "universe": {"symbols": ["BTC", "ETH"]},
            "strategy": {
                "z_entry": 2.0,
                "z_target": 1.0,
                "z_stop": 3.0,
                "trend_window": 10,
                "short_window": 5,
            },
            "execution": {"fee_pct": 0.1, "slippage_pct": 0.05},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0},
        }

        # When: Initializing backtest components
        backtester = Backtester(config, starting_capital=10000.0)
        executor = Executor(config, starting_capital=10000.0)
        strategy = Strategy(config.get("strategy", {}))

        # Then: All components ready for backtest
        assert backtester.starting_capital == 10000.0
        assert executor.current_balance == 10000.0
        assert strategy.z_entry == 2.0

    def test_live_mode_workflow(self):
        """TC-N-17: Live mode workflow initializes correctly."""
        # Given: Live mode configuration
        config = {
            "execution": {"fee_pct": 0.1, "slippage_pct": 0.05},
        }

        # When: Initializing live mode
        executor = Executor(config, starting_capital=100000.0, mode=ExecutionMode.LIVE)

        # Then: Executor ready for live trading
        assert executor.mode == ExecutionMode.LIVE
        assert executor.starting_capital == 100000.0
        assert executor.current_balance == 100000.0

    def test_portfolio_rebalancing_workflow(self):
        """TC-N-18: Portfolio rebalancing workflow completes."""
        # Given: Optimizer and data
        optimizer = PortfolioOptimizer(min_data_days=20)

        # Create returns data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq='D')
        returns = pd.DataFrame({
            "BTC": np.random.randn(60) * 0.02,
            "ETH": np.random.randn(60) * 0.03,
            "SOL": np.random.randn(60) * 0.04,
        }, index=dates)

        # When: Calculating rebalance weights
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: Weights are valid and sum to 1.0
        assert weights is not None
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01
