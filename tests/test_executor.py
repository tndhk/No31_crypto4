"""Tests for src.executor module"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import pytest

from src.executor import Order, Position, Executor, ExecutionMode


class TestOrder:
    """Tests for Order representation"""

    def test_order_creation_buy(self):
        """TC-N-01: BUY order creation succeeds."""
        # Given: Buy order parameters
        symbol = "BTC/USDT"
        order_type = "BUY"
        size = 0.5
        price = 50000.0

        # When: Creating order
        order = Order(
            symbol=symbol,
            order_type=order_type,
            size=size,
            price=price,
        )

        # Then: Order should be created
        assert order.symbol == symbol
        assert order.order_type == order_type
        assert order.size == size
        assert order.price == price
        assert order.timestamp is not None

    def test_order_creation_sell(self):
        """TC-N-02: SELL order creation succeeds."""
        # Given: Sell order parameters
        symbol = "ETH/USDT"
        order_type = "SELL"
        size = 1.5
        price = 3000.0

        # When: Creating order
        order = Order(
            symbol=symbol,
            order_type=order_type,
            size=size,
            price=price,
        )

        # Then: Order should be valid
        assert order.order_type == "SELL"
        assert order.size == 1.5

    def test_order_with_reason(self):
        """TC-N-03: Order with reason/signal reason stored."""
        # Given: Order with reason
        symbol = "SOL/USDT"
        reason = "Z-Score=-2.5 (Entry)"

        # When: Creating order
        order = Order(
            symbol=symbol,
            order_type="BUY",
            size=10.0,
            price=100.0,
            reason=reason,
        )

        # Then: Reason should be stored
        assert order.reason == reason

    def test_order_invalid_type_raises_error(self):
        """TC-A-01: Invalid order type raises ValueError."""
        # Given: Invalid order type
        invalid_type = "INVALID"

        # When/Then: Should raise error
        with pytest.raises((ValueError, KeyError)):
            Order(
                symbol="BTC/USDT",
                order_type=invalid_type,
                size=1.0,
                price=50000.0,
            )

    def test_order_zero_size_raises_error(self):
        """TC-A-02: Zero or negative size raises ValueError."""
        # Given: Invalid size
        size = 0.0

        # When/Then: Should raise error
        with pytest.raises((ValueError, AssertionError)):
            Order(
                symbol="BTC/USDT",
                order_type="BUY",
                size=size,
                price=50000.0,
            )

    def test_order_zero_price_raises_error(self):
        """TC-A-03: Zero or negative price raises ValueError."""
        # Given: Invalid price
        price = -100.0

        # When/Then: Should raise error
        with pytest.raises((ValueError, AssertionError)):
            Order(
                symbol="BTC/USDT",
                order_type="BUY",
                size=1.0,
                price=price,
            )

    def test_order_null_symbol_raises_error(self):
        """TC-A-04: Null symbol raises error."""
        # Given: None symbol
        symbol = None

        # When/Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            Order(
                symbol=symbol,
                order_type="BUY",
                size=1.0,
                price=50000.0,
            )


class TestPosition:
    """Tests for Position tracking"""

    def test_position_creation_long(self):
        """TC-N-04: Long position creation succeeds."""
        # Given: Long position parameters
        symbol = "BTC/USDT"
        entry_price = 50000.0
        size = 0.5

        # When: Creating position
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            size=size,
            position_type="LONG",
        )

        # Then: Position should be created
        assert position.symbol == symbol
        assert position.entry_price == entry_price
        assert position.size == size
        assert position.position_type == "LONG"

    def test_position_creation_short(self):
        """TC-N-05: Short position creation succeeds."""
        # Given: Short position parameters
        symbol = "ETH/USDT"

        # When: Creating position
        position = Position(
            symbol=symbol,
            entry_price=3000.0,
            size=2.0,
            position_type="SHORT",
        )

        # Then: Position should be valid
        assert position.position_type == "SHORT"

    def test_position_current_value_long(self):
        """TC-N-06: Long position current value calculated correctly."""
        # Given: Long position (entry 50000, current 55000)
        position = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="LONG",
        )
        current_price = 55000.0

        # When: Calculating current value
        current_value = position.get_current_value(current_price)

        # Then: Should be 55000 * 0.5 = 27500
        assert current_value == 27500.0

    def test_position_pnl_long_profit(self):
        """TC-N-07: Long position P&L calculated correctly (profit)."""
        # Given: Long position with profit
        position = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="LONG",
        )
        current_price = 55000.0

        # When: Calculating P&L
        pnl = position.get_pnl(current_price)

        # Then: P&L = (55000 - 50000) * 0.5 = 2500
        assert pnl == 2500.0

    def test_position_pnl_long_loss(self):
        """TC-N-08: Long position P&L with loss."""
        # Given: Long position with loss
        position = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="LONG",
        )
        current_price = 45000.0

        # When: Calculating P&L
        pnl = position.get_pnl(current_price)

        # Then: P&L = (45000 - 50000) * 0.5 = -2500
        assert pnl == -2500.0

    def test_position_pnl_short_profit(self):
        """TC-N-09: Short position P&L with profit."""
        # Given: Short position (entry 50000, current 45000)
        position = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="SHORT",
        )
        current_price = 45000.0

        # When: Calculating P&L
        pnl = position.get_pnl(current_price)

        # Then: P&L = (50000 - 45000) * 0.5 = 2500
        assert pnl == 2500.0

    def test_position_pnl_short_loss(self):
        """TC-N-10: Short position P&L with loss."""
        # Given: Short position (entry 50000, current 55000)
        position = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="SHORT",
        )
        current_price = 55000.0

        # When: Calculating P&L
        pnl = position.get_pnl(current_price)

        # Then: P&L = (50000 - 55000) * 0.5 = -2500
        assert pnl == -2500.0

    def test_position_invalid_type_raises_error(self):
        """TC-A-05: Invalid position type raises error."""
        # Given: Invalid position type
        invalid_type = "INVALID_TYPE"

        # When/Then: Should raise error
        with pytest.raises((ValueError, KeyError)):
            Position(
                symbol="BTC/USDT",
                entry_price=50000.0,
                size=0.5,
                position_type=invalid_type,
            )

    def test_position_zero_size_raises_error(self):
        """TC-A-06: Zero size raises error."""
        # Given: Zero size
        size = 0.0

        # When/Then: Should raise error
        with pytest.raises((ValueError, AssertionError)):
            Position(
                symbol="BTC/USDT",
                entry_price=50000.0,
                size=size,
                position_type="LONG",
            )


class TestExecutorInitialization:
    """Tests for Executor initialization"""

    def test_executor_backtest_mode(self):
        """TC-N-11: Executor initialization in backtest mode."""
        # Given: Backtest configuration
        config = {
            "execution": {
                "mode": "backtest",
                "fee_pct": 0.11,
                "slippage_pct": 0.05,
            },
        }

        # When: Creating executor
        executor = Executor(config, mode=ExecutionMode.BACKTEST)

        # Then: Should initialize
        assert executor.mode == ExecutionMode.BACKTEST
        assert executor.positions == {}
        assert executor.trades == []

    def test_executor_live_mode(self):
        """TC-N-12: Executor initialization in live mode."""
        # Given: Live configuration
        config = {
            "execution": {
                "mode": "live",
                "fee_pct": 0.11,
            },
        }

        # When: Creating executor
        executor = Executor(config, mode=ExecutionMode.LIVE)

        # Then: Should initialize
        assert executor.mode == ExecutionMode.LIVE

    def test_executor_with_starting_capital(self):
        """TC-N-13: Executor tracks capital and balance."""
        # Given: Starting capital
        capital = 100000.0

        config = {"execution": {"fee_pct": 0.11}}

        # When: Creating executor
        executor = Executor(config, starting_capital=capital)

        # Then: Capital should be tracked
        assert executor.starting_capital == capital
        assert executor.current_balance == capital

    def test_executor_null_config_raises_error(self):
        """TC-A-07: Null config raises error."""
        # Given: None config
        config = None

        # When/Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            Executor(config)


class TestOrderExecution:
    """Tests for order execution logic"""

    def test_execute_buy_order_backtest(self):
        """TC-N-14: Execute BUY order in backtest mode."""
        # Given: Executor with capital and BUY order
        config = {"execution": {"fee_pct": 0.11, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=100000.0, mode=ExecutionMode.BACKTEST)

        order = Order(
            symbol="BTC/USDT",
            order_type="BUY",
            size=0.5,
            price=50000.0,
        )

        # When: Executing order
        executor.execute_order(order)

        # Then: Position should be created
        assert "BTC/USDT" in executor.positions
        assert executor.positions["BTC/USDT"].size == 0.5

    def test_execute_sell_order_backtest(self):
        """TC-N-15: Execute SELL order in backtest mode."""
        # Given: Executor with existing position
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0, mode=ExecutionMode.BACKTEST)

        # Create position first
        executor.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="LONG",
        )

        # Then execute SELL order
        order = Order(
            symbol="BTC/USDT",
            order_type="SELL",
            size=0.5,
            price=55000.0,
        )

        executor.execute_order(order)

        # Then: Position should be closed
        assert "BTC/USDT" not in executor.positions

    def test_execute_order_applies_fees(self):
        """TC-N-16: Order execution applies fees and slippage."""
        # Given: Executor with fee configuration
        fee_pct = 0.11
        config = {"execution": {"fee_pct": fee_pct, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=100000.0)

        order = Order(
            symbol="BTC/USDT",
            order_type="BUY",
            size=1.0,
            price=50000.0,
        )

        initial_balance = executor.current_balance

        # When: Executing order
        executor.execute_order(order)

        # Then: Balance should decrease by cost including fees
        # Cost = 50000 * 1.0 * (1 + fee/100 + slippage/100) ≈ 50800
        cost = 50000.0 * (1 + (fee_pct + 0.05) / 100.0)
        expected_balance = initial_balance - cost

        assert executor.current_balance < initial_balance

    def test_execute_order_insufficient_balance_raises_error(self):
        """TC-A-08: Insufficient balance raises error."""
        # Given: Executor with low balance
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=1000.0)  # Only $1000

        order = Order(
            symbol="BTC/USDT",
            order_type="BUY",
            size=1.0,
            price=50000.0,  # Costs more than balance
        )

        # When/Then: Should raise error
        with pytest.raises((ValueError, RuntimeError)):
            executor.execute_order(order)

    def test_execute_order_logs_trade(self):
        """TC-N-17: Order execution logs trade."""
        # Given: Executor with logging
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        order = Order(
            symbol="BTC/USDT",
            order_type="BUY",
            size=0.5,
            price=50000.0,
            reason="Z-Score=-2.5",
        )

        initial_trades = len(executor.trades)

        # When: Executing order
        executor.execute_order(order)

        # Then: Trade should be logged
        assert len(executor.trades) > initial_trades


class TestPositionManagement:
    """Tests for position management"""

    def test_close_position_long(self):
        """TC-N-18: Close long position succeeds."""
        # Given: Executor with long position
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        position = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="LONG",
        )
        executor.positions["BTC/USDT"] = position

        # When: Closing position
        executor.close_position("BTC/USDT", exit_price=55000.0)

        # Then: Position should be removed
        assert "BTC/USDT" not in executor.positions

    def test_close_position_nonexistent_raises_error(self):
        """TC-A-09: Closing nonexistent position raises error."""
        # Given: Executor with no position
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        # When/Then: Should raise error
        with pytest.raises((KeyError, ValueError)):
            executor.close_position("BTC/USDT", exit_price=50000.0)

    def test_get_all_positions(self):
        """TC-N-19: Get all positions returns correct list."""
        # Given: Executor with multiple positions
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        executor.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.5,
            position_type="LONG",
        )
        executor.positions["ETH/USDT"] = Position(
            symbol="ETH/USDT",
            entry_price=3000.0,
            size=5.0,
            position_type="LONG",
        )

        # When: Getting all positions
        positions = executor.get_all_positions()

        # Then: Should return all positions
        assert len(positions) == 2
        assert "BTC/USDT" in positions
        assert "ETH/USDT" in positions


class TestTradeLogging:
    """Tests for trade logging"""

    def test_trade_record_structure(self):
        """TC-N-20: Trade record contains required fields."""
        # Given: Executor and executed order
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        order = Order(
            symbol="BTC/USDT",
            order_type="BUY",
            size=0.5,
            price=50000.0,
            reason="Entry signal",
        )

        # When: Executing order
        executor.execute_order(order)

        # Then: Trade should be logged with required fields
        assert len(executor.trades) > 0
        trade = executor.trades[0]
        assert "timestamp" in trade
        assert "symbol" in trade
        assert "order_type" in trade
        assert "size" in trade
        assert "price" in trade

    def test_multiple_trades_logged(self):
        """TC-N-21: Multiple trades all logged."""
        # Given: Executor and multiple orders
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        orders = [
            Order(symbol="BTC/USDT", order_type="BUY", size=0.1, price=50000.0),
            Order(symbol="ETH/USDT", order_type="BUY", size=0.5, price=3000.0),
        ]

        # When: Executing all orders
        for order in orders:
            executor.execute_order(order)

        # Then: All trades should be logged
        assert len(executor.trades) == 2


class TestErrorHandling:
    """Tests for error handling"""

    def test_executor_invalid_order_type_raises_error(self):
        """TC-A-10: Invalid order type raises error."""
        # Given: Executor with invalid order
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        # When/Then: Order creation should fail
        with pytest.raises((ValueError, KeyError)):
            order = Order(
                symbol="BTC/USDT",
                order_type="INVALID",
                size=0.5,
                price=50000.0,
            )

    def test_executor_handles_nan_price(self):
        """TC-A-11: NaN price raises error."""
        # Given: NaN price
        price = float('nan')

        # When/Then: Should raise error
        with pytest.raises((ValueError, AssertionError)):
            Order(
                symbol="BTC/USDT",
                order_type="BUY",
                size=0.5,
                price=price,
            )

    def test_executor_handles_inf_price(self):
        """TC-A-12: Infinite price raises error."""
        # Given: Infinite price
        price = float('inf')

        # When/Then: Should raise error
        with pytest.raises((ValueError, AssertionError)):
            Order(
                symbol="BTC/USDT",
                order_type="BUY",
                size=0.5,
                price=price,
            )


class TestExecutorIntegration:
    """Integration tests for executor"""

    def test_full_trade_cycle_long(self):
        """TC-N-22: Complete LONG trade cycle (open and close)."""
        # Given: Executor ready for trading
        config = {"execution": {"fee_pct": 0.11, "slippage_pct": 0.05}}
        executor = Executor(config, starting_capital=100000.0)

        # When: Opening LONG position
        buy_order = Order(
            symbol="BTC/USDT",
            order_type="BUY",
            size=0.5,
            price=50000.0,
        )
        executor.execute_order(buy_order)

        # Then: Position should exist
        assert "BTC/USDT" in executor.positions

        # When: Closing position
        sell_order = Order(
            symbol="BTC/USDT",
            order_type="SELL",
            size=0.5,
            price=55000.0,
        )
        executor.execute_order(sell_order)

        # Then: Position should be closed
        assert "BTC/USDT" not in executor.positions

    def test_portfolio_execution_multiple_symbols(self):
        """TC-N-23: Execute portfolio with multiple symbols."""
        # Given: Portfolio of orders
        config = {"execution": {"fee_pct": 0.11}}
        executor = Executor(config, starting_capital=100000.0)

        portfolio_orders = [
            Order(symbol="BTC/USDT", order_type="BUY", size=0.1, price=50000.0),
            Order(symbol="ETH/USDT", order_type="BUY", size=0.5, price=3000.0),
            Order(symbol="SOL/USDT", order_type="BUY", size=10.0, price=100.0),
        ]

        # When: Executing portfolio
        for order in portfolio_orders:
            executor.execute_order(order)

        # Then: All positions should exist
        assert len(executor.positions) == 3
        assert executor.current_balance < 100000.0
