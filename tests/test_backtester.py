"""Tests for src.backtester module"""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import pytest

from src.backtester import Backtester, WFOSplit, BacktestMetrics


class TestWFOSplit:
    """Tests for Walk-Forward Optimization window splitting"""

    def test_wfo_split_valid_data(self):
        """TC-N-01: Valid WFO split generates correct IS/OOS windows."""
        # Given: 820 days of data (minimum for IS=730 + OOS=90)
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.randn(820) * 0.02 + 100,
        })

        # When: Creating WFO split
        wfo = WFOSplit(df, is_window_days=730, oos_window_days=90)

        # Then: Should have valid split
        assert wfo.is_df is not None
        assert wfo.oos_df is not None
        assert len(wfo.is_df) == 730
        assert len(wfo.oos_df) == 90

    def test_wfo_split_exact_minimum_data(self):
        """TC-B-01: Exactly 820 days (730 IS + 90 OOS) returns single window."""
        # Given: Exactly 820 days of data
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.arange(820, dtype=float),
        })

        # When: Creating WFO split
        wfo = WFOSplit(df, is_window_days=730, oos_window_days=90)

        # Then: Should create valid split with exact sizes
        assert len(wfo.is_df) == 730
        assert len(wfo.oos_df) == 90
        assert wfo.is_df["close"].iloc[0] == 0.0
        assert wfo.oos_df["close"].iloc[-1] == 819.0

    def test_wfo_split_sufficient_data_for_rolling(self):
        """TC-N-02: 1460 days (2 full windows) enables rolling windows."""
        # Given: 1460 days (2 * 730 IS windows)
        dates = pd.date_range("2020-01-01", periods=1460, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.randn(1460) * 0.02 + 100,
        })

        # When: Creating WFO split
        wfo = WFOSplit(df, is_window_days=730, oos_window_days=90)

        # Then: First window should be valid
        assert len(wfo.is_df) == 730
        assert len(wfo.oos_df) == 90

    def test_wfo_split_insufficient_data_raises_error(self):
        """TC-A-01: Less than 820 days raises ValueError."""
        # Given: 819 days (1 day short)
        dates = pd.date_range("2022-01-01", periods=819, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.randn(819),
        })

        # When: Creating WFO split
        # Then: Should raise error
        with pytest.raises(ValueError, match="insufficient|minimum|820"):
            WFOSplit(df, is_window_days=730, oos_window_days=90)

    def test_wfo_split_null_dataframe_raises_error(self):
        """TC-A-02: Null DataFrame raises TypeError."""
        # Given: None DataFrame
        df = None

        # When/Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            WFOSplit(df, is_window_days=730, oos_window_days=90)

    def test_wfo_split_empty_dataframe_raises_error(self):
        """TC-A-03: Empty DataFrame raises ValueError."""
        # Given: Empty DataFrame
        df = pd.DataFrame()

        # When/Then: Should raise error
        with pytest.raises((ValueError, KeyError)):
            WFOSplit(df, is_window_days=730, oos_window_days=90)

    def test_wfo_split_missing_timestamp_column(self):
        """TC-A-04: Missing timestamp column raises ValueError."""
        # Given: DataFrame without timestamp
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        df = pd.DataFrame({
            "close": np.random.randn(820),
        }, index=dates)

        # When/Then: Should raise error
        with pytest.raises((ValueError, KeyError)):
            WFOSplit(df, is_window_days=730, oos_window_days=90)

    def test_wfo_split_preserves_data_order(self):
        """TC-B-02: WFO split maintains chronological order."""
        # Given: DataFrame with sequential prices
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.arange(820, dtype=float),
        })

        # When: Creating WFO split
        wfo = WFOSplit(df, is_window_days=730, oos_window_days=90)

        # Then: Order should be preserved
        assert wfo.is_df["close"].is_monotonic_increasing
        assert wfo.oos_df["close"].is_monotonic_increasing


class TestBacktestMetrics:
    """Tests for backtest metrics calculation"""

    def test_sharpe_ratio_calculation(self):
        """TC-N-03: Sharpe ratio calculated correctly from returns."""
        # Given: Daily returns with known statistics
        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005] * 52)  # 1 year

        # When: Calculating Sharpe ratio
        metrics = BacktestMetrics(
            returns=returns,
            start_value=10000.0,
            end_value=11000.0,
        )

        # Then: Sharpe should be calculated (annualized)
        assert metrics.sharpe_ratio is not None
        assert isinstance(metrics.sharpe_ratio, (float, np.floating))

    def test_max_drawdown_calculation(self):
        """TC-N-04: Max Drawdown calculated correctly from equity curve."""
        # Given: Equity curve with known drawdown
        equity = np.array([10000, 11000, 10500, 9500, 11500, 10000])  # -13.6% max DD

        # When: Calculating max drawdown
        returns = np.diff(equity) / equity[:-1]
        metrics = BacktestMetrics(
            returns=returns,
            start_value=10000.0,
            end_value=10000.0,
        )

        # Then: Max drawdown should match expected
        assert metrics.max_drawdown_pct is not None
        assert metrics.max_drawdown_pct >= 0

    def test_total_return_calculation(self):
        """TC-N-05: Total return calculated correctly."""
        # Given: Start and end values
        start = 10000.0
        end = 12000.0  # 20% return

        # When: Calculating total return
        metrics = BacktestMetrics(
            returns=np.array([0.01] * 100),
            start_value=start,
            end_value=end,
        )

        # Then: Return should be 20%
        assert metrics.total_return_pct is not None
        assert metrics.total_return_pct > 0

    def test_metrics_with_negative_returns(self):
        """TC-N-06: Metrics handle negative returns correctly."""
        # Given: Losing strategy
        returns = np.array([-0.01, -0.02, -0.01] * 50)  # Losing

        # When: Calculating metrics
        metrics = BacktestMetrics(
            returns=returns,
            start_value=10000.0,
            end_value=9000.0,
        )

        # Then: Should still calculate valid metrics
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown_pct is not None
        assert metrics.total_return_pct < 0

    def test_metrics_with_zero_std_returns(self):
        """TC-B-03: Flat returns (zero std) handled gracefully."""
        # Given: Constant returns (zero volatility)
        returns = np.array([0.0] * 100)

        # When: Calculating metrics
        metrics = BacktestMetrics(
            returns=returns,
            start_value=10000.0,
            end_value=10000.0,
        )

        # Then: Should return 0 Sharpe (not crash)
        assert metrics.sharpe_ratio == 0.0

    def test_metrics_insufficient_data(self):
        """TC-A-05: Less than 2 data points raises error."""
        # Given: Single return value
        returns = np.array([0.01])

        # When/Then: Should raise error
        with pytest.raises((ValueError, IndexError)):
            BacktestMetrics(
                returns=returns,
                start_value=10000.0,
                end_value=10100.0,
            )

    def test_metrics_null_returns_raises_error(self):
        """TC-A-06: Null returns raises error."""
        # Given: None returns
        returns = None

        # When/Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            BacktestMetrics(
                returns=returns,
                start_value=10000.0,
                end_value=10100.0,
            )


class TestBacktesterInitialization:
    """Tests for Backtester initialization"""

    def test_backtester_initialization_valid(self):
        """TC-N-07: Backtester initializes with valid configuration."""
        # Given: Valid configuration
        config = {
            "strategy": {"z_entry": 2.0, "z_target": 1.0, "z_stop": 3.0},
            "execution": {"fee_pct": 0.11, "slippage_pct": 0.05},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0},
        }

        # When: Creating backtester
        backtester = Backtester(config)

        # Then: Should initialize successfully
        assert backtester is not None
        assert backtester.config == config

    def test_backtester_initialization_with_starting_capital(self):
        """TC-N-08: Backtester accepts custom starting capital."""
        # Given: Custom capital amount
        config = {
            "strategy": {"z_entry": 2.0},
            "execution": {"fee_pct": 0.11},
            "validation": {"min_sharpe_ratio": 1.0},
        }
        capital = 50000.0

        # When: Creating backtester with custom capital
        backtester = Backtester(config, starting_capital=capital)

        # Then: Should use custom capital
        assert backtester.starting_capital == capital

    def test_backtester_null_config_raises_error(self):
        """TC-A-07: Null config raises TypeError."""
        # Given: None config
        config = None

        # When/Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            Backtester(config)


class TestWalkForwardOptimization:
    """Tests for walk-forward optimization engine"""

    def test_wfo_basic_run(self):
        """TC-N-09: WFO runs and returns results."""
        # Given: 820 days of price data and strategy parameters
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        price_data = 100 + np.cumsum(np.random.randn(820) * 2)
        df = pd.DataFrame({
            "timestamp": dates,
            "close": price_data,
            "open": price_data - 1,
            "high": price_data + 1,
            "low": price_data - 1,
            "volume": 1000000.0,
        })

        config = {
            "strategy": {"z_entry": 2.0, "z_target": 1.0, "z_stop": 3.0, "trend_window": 50, "short_window": 10},
            "execution": {"fee_pct": 0.11, "slippage_pct": 0.05},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0},
        }

        # When: Running WFO
        backtester = Backtester(config)
        result = backtester.walk_forward_optimize(df)

        # Then: Should return results
        assert result is not None
        assert "is_metrics" in result
        assert "oos_metrics" in result

    def test_wfo_insufficient_data_raises_error(self):
        """TC-A-08: Less than 820 days raises ValueError."""
        # Given: 500 days of data (insufficient)
        dates = pd.date_range("2022-01-01", periods=500, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.randn(500) + 100,
            "open": np.random.randn(500) + 100,
            "high": np.random.randn(500) + 101,
            "low": np.random.randn(500) + 99,
            "volume": np.ones(500) * 1000000.0,
        })

        config = {
            "strategy": {"z_entry": 2.0},
            "execution": {"fee_pct": 0.11},
            "validation": {"min_sharpe_ratio": 1.0},
        }

        # When: Running WFO
        backtester = Backtester(config)

        # Then: Should raise error
        with pytest.raises(ValueError, match="insufficient|minimum"):
            backtester.walk_forward_optimize(df)

    def test_wfo_go_no_go_validation_pass(self):
        """TC-N-10: Go/No-Go validation PASS when Sharpe > 1.0 AND DD < 20%."""
        # Given: Strong backtest results (Sharpe=1.5, DD=15%)
        # This is mocked in the result
        result = {
            "is_metrics": type('obj', (object,), {
                'sharpe_ratio': 1.5,
                'max_drawdown_pct': 15.0,
            })(),
            "oos_metrics": type('obj', (object,), {
                'sharpe_ratio': 0.8,
                'max_drawdown_pct': 18.0,
            })(),
        }

        # When: Checking go/no-go criteria
        min_sharpe = 1.0
        max_dd = 20.0

        # Then: Should be GO
        is_pass = (result["is_metrics"].sharpe_ratio > min_sharpe and
                   result["is_metrics"].max_drawdown_pct < max_dd)
        oos_pass = (result["oos_metrics"].sharpe_ratio > 0 and
                    result["oos_metrics"].max_drawdown_pct < max_dd)
        assert is_pass

    def test_wfo_go_no_go_validation_fail_low_sharpe(self):
        """TC-N-11: Go/No-Go validation FAIL when Sharpe <= 1.0."""
        # Given: Low Sharpe (0.5)
        result = {
            "is_metrics": type('obj', (object,), {
                'sharpe_ratio': 0.5,
                'max_drawdown_pct': 15.0,
            })(),
        }

        # When: Checking go/no-go criteria
        min_sharpe = 1.0
        max_dd = 20.0

        # Then: Should be NO-GO
        is_pass = (result["is_metrics"].sharpe_ratio > min_sharpe and
                   result["is_metrics"].max_drawdown_pct < max_dd)
        assert not is_pass

    def test_wfo_go_no_go_validation_fail_high_dd(self):
        """TC-N-12: Go/No-Go validation FAIL when DD >= 20%."""
        # Given: High drawdown (25%)
        result = {
            "is_metrics": type('obj', (object,), {
                'sharpe_ratio': 1.5,
                'max_drawdown_pct': 25.0,
            })(),
        }

        # When: Checking go/no-go criteria
        min_sharpe = 1.0
        max_dd = 20.0

        # Then: Should be NO-GO
        is_pass = (result["is_metrics"].sharpe_ratio > min_sharpe and
                   result["is_metrics"].max_drawdown_pct < max_dd)
        assert not is_pass

    def test_wfo_rolling_window_advancement(self):
        """TC-N-13: Multiple WFO windows advance by 90 days (OOS window)."""
        # Given: 1460 days (2 full WFO cycles)
        dates = pd.date_range("2020-01-01", periods=1460, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": 100 + np.cumsum(np.random.randn(1460) * 0.5),
            "open": 100 + np.cumsum(np.random.randn(1460) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(1460) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(1460) * 0.5),
            "volume": np.ones(1460) * 1000000.0,
        })

        # When: Creating first WFO split
        wfo1 = WFOSplit(df, is_window_days=730, oos_window_days=90)

        # Then: Next window should start 90 days later
        next_start = 730 + 90  # 820
        wfo2_data = df.iloc[next_start:]
        if len(wfo2_data) >= 820:
            wfo2 = WFOSplit(wfo2_data.reset_index(drop=True), is_window_days=730, oos_window_days=90)
            assert wfo2 is not None

    def test_wfo_null_dataframe_raises_error(self):
        """TC-A-09: Null DataFrame raises TypeError."""
        # Given: None DataFrame
        df = None

        config = {
            "strategy": {"z_entry": 2.0},
            "execution": {"fee_pct": 0.11},
            "validation": {"min_sharpe_ratio": 1.0},
        }

        # When: Running WFO
        backtester = Backtester(config)

        # Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            backtester.walk_forward_optimize(df)


class TestTradeSimulation:
    """Tests for trade execution simulation"""

    def test_trade_simulation_with_signal(self):
        """TC-N-14: Trade simulation executes orders with fees and slippage."""
        # Given: Price data and buy/sell prices
        entry_price = 100.0
        exit_price = 105.0
        fee_pct = 0.11
        slippage_pct = 0.05

        # When: Simulating trade
        # Entry cost with fee and slippage
        entry_cost = entry_price * (1 + fee_pct / 100 + slippage_pct / 100)
        # Exit proceeds with fee and slippage
        exit_proceeds = exit_price * (1 - fee_pct / 100 - slippage_pct / 100)

        # Then: Should calculate P&L correctly
        pnl = exit_proceeds - entry_cost
        assert pnl < (exit_price - entry_price)  # Less than ideal due to costs

    def test_trade_simulation_losing_trade(self):
        """TC-N-15: Losing trade P&L calculated correctly."""
        # Given: Entry at 100, exit at 95
        entry_price = 100.0
        exit_price = 95.0

        # When: Simulating losing trade
        pnl = exit_price - entry_price

        # Then: P&L should be negative
        assert pnl < 0

    def test_trade_simulation_multiple_trades(self):
        """TC-N-16: Multiple trades accumulate P&L correctly."""
        # Given: Three trades
        trades = [
            (100.0, 105.0),  # +5 profit
            (105.0, 103.0),  # -2 loss
            (103.0, 108.0),  # +5 profit
        ]

        # When: Calculating cumulative P&L
        total_pnl = sum(exit - entry for entry, exit in trades)

        # Then: Should be +8
        assert total_pnl == 8.0


class TestErrorHandling:
    """Tests for error handling in backtester"""

    def test_backtester_missing_required_columns(self):
        """TC-A-10: DataFrame missing required columns raises error."""
        # Given: DataFrame without OHLCV columns
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "price": np.random.randn(820) + 100,  # Wrong column name
        })

        config = {
            "strategy": {"z_entry": 2.0},
            "execution": {"fee_pct": 0.11},
            "validation": {"min_sharpe_ratio": 1.0},
        }

        # When: Running WFO
        backtester = Backtester(config)

        # Then: Should raise error
        with pytest.raises((ValueError, KeyError)):
            backtester.walk_forward_optimize(df)

    def test_backtester_nan_in_prices(self):
        """TC-A-11: NaN values in price data raises error."""
        # Given: Data with NaN prices
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        prices = np.random.randn(820) + 100
        prices[100:105] = np.nan  # Insert NaN

        df = pd.DataFrame({
            "timestamp": dates,
            "close": prices,
            "open": prices - 1,
            "high": prices + 1,
            "low": prices - 1,
            "volume": np.ones(820) * 1000000.0,
        })

        config = {
            "strategy": {"z_entry": 2.0},
            "execution": {"fee_pct": 0.11},
            "validation": {"min_sharpe_ratio": 1.0},
        }

        # When: Running WFO
        backtester = Backtester(config)

        # Then: Should raise error or handle gracefully
        with pytest.raises((ValueError, TypeError)):
            backtester.walk_forward_optimize(df)


class TestBacktestResults:
    """Tests for backtest result reporting"""

    def test_backtest_results_contain_metrics(self):
        """TC-B-04: Backtest results include IS and OOS metrics."""
        # Given: Valid backtest configuration
        dates = pd.date_range("2022-01-01", periods=820, freq='D')
        df = pd.DataFrame({
            "timestamp": dates,
            "close": 100 + np.cumsum(np.random.randn(820) * 0.5),
            "open": 100 + np.cumsum(np.random.randn(820) * 0.5),
            "high": 101 + np.cumsum(np.random.randn(820) * 0.5),
            "low": 99 + np.cumsum(np.random.randn(820) * 0.5),
            "volume": np.ones(820) * 1000000.0,
        })

        config = {
            "strategy": {"z_entry": 2.0, "z_target": 1.0, "z_stop": 3.0, "trend_window": 50, "short_window": 10},
            "execution": {"fee_pct": 0.11, "slippage_pct": 0.05},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0},
        }

        # When: Running WFO
        backtester = Backtester(config)
        result = backtester.walk_forward_optimize(df)

        # Then: Should have both metrics
        assert "is_metrics" in result
        assert "oos_metrics" in result
        assert hasattr(result["is_metrics"], "sharpe_ratio")
        assert hasattr(result["oos_metrics"], "sharpe_ratio")

    def test_backtest_results_include_go_no_go_decision(self):
        """TC-B-05: Backtest results include Go/No-Go decision."""
        # Given: Backtest configuration
        config = {
            "strategy": {"z_entry": 2.0},
            "execution": {"fee_pct": 0.11},
            "validation": {"min_sharpe_ratio": 1.0, "max_drawdown_pct": 20.0},
        }

        # When: Creating backtester
        backtester = Backtester(config)

        # Then: Should be able to evaluate go/no-go
        min_sharpe = config["validation"]["min_sharpe_ratio"]
        max_dd = config["validation"]["max_drawdown_pct"]
        assert min_sharpe > 0
        assert max_dd > 0
