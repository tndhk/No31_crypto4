"""Order execution engine for backtest and live trading modes"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np


class ExecutionMode(Enum):
    """Trading execution mode"""
    BACKTEST = "backtest"
    LIVE = "live"


class Order:
    """Order representation"""

    VALID_TYPES = {"BUY", "SELL"}

    def __init__(
        self,
        symbol: str,
        order_type: str,
        size: float,
        price: float,
        reason: str = "",
    ):
        """Initialize Order.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            order_type: "BUY" or "SELL"
            size: Order size in base asset
            price: Order price
            reason: Signal reason (e.g., "Z-Score=-2.5")

        Raises:
            TypeError: If symbol is None
            ValueError: If invalid order_type, zero/negative size or price
        """
        if symbol is None:
            raise TypeError("Symbol cannot be None")

        if order_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid order_type: {order_type}. Must be in {self.VALID_TYPES}")

        if size <= 0:
            raise ValueError(f"Size must be positive: {size}")

        if price <= 0 or np.isnan(price) or np.isinf(price):
            raise ValueError(f"Price must be positive and finite: {price}")

        self.symbol = symbol
        self.order_type = order_type
        self.size = size
        self.price = price
        self.reason = reason
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        return f"Order({self.symbol}, {self.order_type}, {self.size}, {self.price})"


class Position:
    """Position tracking"""

    VALID_POSITION_TYPES = {"LONG", "SHORT"}

    def __init__(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        position_type: str,
    ):
        """Initialize Position.

        Args:
            symbol: Trading pair
            entry_price: Entry price
            size: Position size
            position_type: "LONG" or "SHORT"

        Raises:
            ValueError: If invalid position_type or zero/negative size/price
        """
        if position_type not in self.VALID_POSITION_TYPES:
            raise ValueError(f"Invalid position_type: {position_type}")

        if size <= 0:
            raise ValueError(f"Size must be positive: {size}")

        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {entry_price}")

        self.symbol = symbol
        self.entry_price = entry_price
        self.size = size
        self.position_type = position_type
        self.entry_time = datetime.now()

    def get_current_value(self, current_price: float) -> float:
        """Get current position value.

        Args:
            current_price: Current market price

        Returns:
            Current value in USD
        """
        return current_price * self.size

    def get_pnl(self, current_price: float) -> float:
        """Calculate position P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in USD
        """
        if self.position_type == "LONG":
            return (current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - current_price) * self.size

    def get_pnl_pct(self, current_price: float) -> float:
        """Calculate position P&L percentage.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L percentage
        """
        pnl = self.get_pnl(current_price)
        return (pnl / (self.entry_price * self.size)) * 100.0

    def __repr__(self) -> str:
        return f"Position({self.symbol}, {self.position_type}, {self.size}@{self.entry_price})"


class Executor:
    """Order execution engine for backtest and live trading"""

    def __init__(
        self,
        config: Dict[str, Any],
        starting_capital: float = 10000.0,
        mode: ExecutionMode = ExecutionMode.BACKTEST,
    ):
        """Initialize Executor.

        Args:
            config: Configuration dictionary with execution settings
            starting_capital: Starting capital for backtest
            mode: ExecutionMode.BACKTEST or ExecutionMode.LIVE

        Raises:
            TypeError: If config is None
            ValueError: If invalid configuration
        """
        if config is None:
            raise TypeError("Config cannot be None")

        self.config = config
        self.starting_capital = starting_capital
        self.current_balance = starting_capital
        self.mode = mode

        # Extract fees from config
        self.fee_pct = config.get("execution", {}).get("fee_pct", 0.0)
        self.slippage_pct = config.get("execution", {}).get("slippage_pct", 0.0)

        # Position and trade tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []

    def execute_order(self, order: Order) -> bool:
        """Execute a single order.

        Args:
            order: Order to execute

        Returns:
            True if execution successful

        Raises:
            ValueError: If insufficient balance or invalid order
            RuntimeError: If execution fails
        """
        if order.order_type == "BUY":
            return self._execute_buy(order)
        elif order.order_type == "SELL":
            return self._execute_sell(order)
        else:
            raise ValueError(f"Invalid order type: {order.order_type}")

    def _execute_buy(self, order: Order) -> bool:
        """Execute BUY order.

        Args:
            order: BUY order

        Returns:
            True if successful

        Raises:
            ValueError: If insufficient balance
        """
        # Calculate cost with fees and slippage
        cost_multiplier = 1.0 + (self.fee_pct + self.slippage_pct) / 100.0
        total_cost = order.price * order.size * cost_multiplier

        if self.current_balance < total_cost:
            raise ValueError(
                f"Insufficient balance: {self.current_balance} < {total_cost}"
            )

        # Create or update position
        if order.symbol in self.positions:
            # Add to existing position
            old_pos = self.positions[order.symbol]
            new_size = old_pos.size + order.size
            new_entry_price = (
                (old_pos.entry_price * old_pos.size + order.price * order.size) / new_size
            )
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                entry_price=new_entry_price,
                size=new_size,
                position_type="LONG",
            )
        else:
            # Create new position
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                entry_price=order.price,
                size=order.size,
                position_type="LONG",
            )

        # Deduct from balance
        self.current_balance -= total_cost

        # Log trade
        self._log_trade(order)

        return True

    def _execute_sell(self, order: Order) -> bool:
        """Execute SELL order.

        Args:
            order: SELL order

        Returns:
            True if successful

        Raises:
            ValueError: If no position to sell
        """
        if order.symbol not in self.positions:
            raise ValueError(f"No position to sell for {order.symbol}")

        position = self.positions[order.symbol]

        if position.size < order.size:
            raise ValueError(
                f"Insufficient position size: {position.size} < {order.size}"
            )

        # Calculate proceeds with fees and slippage deduction
        cost_multiplier = 1.0 - (self.fee_pct + self.slippage_pct) / 100.0
        proceeds = order.price * order.size * cost_multiplier

        # Calculate P&L
        pnl = (order.price - position.entry_price) * order.size

        # Update or remove position
        if position.size == order.size:
            # Close entire position
            del self.positions[order.symbol]
        else:
            # Reduce position size
            new_size = position.size - order.size
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                entry_price=position.entry_price,
                size=new_size,
                position_type=position.position_type,
            )

        # Add proceeds to balance
        self.current_balance += proceeds

        # Log trade and closed trade
        self._log_trade(order)
        self._log_closed_trade(order, pnl, position.entry_price)

        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
    ) -> Dict[str, Any]:
        """Close a position at specified price.

        Args:
            symbol: Trading pair
            exit_price: Exit price

        Returns:
            Trade result dictionary

        Raises:
            KeyError: If position doesn't exist
            ValueError: If invalid exit_price
        """
        if symbol not in self.positions:
            raise KeyError(f"No position for {symbol}")

        if exit_price <= 0:
            raise ValueError(f"Exit price must be positive: {exit_price}")

        position = self.positions[symbol]

        # Create SELL order
        sell_order = Order(
            symbol=symbol,
            order_type="SELL",
            size=position.size,
            price=exit_price,
        )

        self._execute_sell(sell_order)

        # Calculate result
        pnl = (exit_price - position.entry_price) * position.size

        return {
            "symbol": symbol,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size": position.size,
            "pnl": pnl,
            "pnl_pct": (pnl / (position.entry_price * position.size)) * 100.0,
        }

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions.

        Returns:
            Dictionary of symbol -> Position
        """
        return self.positions.copy()

    def get_total_exposure(self, current_prices: Dict[str, float]) -> float:
        """Get total notional exposure.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Total exposure in USD
        """
        exposure = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                exposure += position.get_current_value(current_prices[symbol])
        return exposure

    def get_total_pnl(self, current_prices: Dict[str, float]) -> float:
        """Get total unrealized P&L.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Total P&L in USD
        """
        pnl = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                pnl += position.get_pnl(current_prices[symbol])
        return pnl

    def _log_trade(self, order: Order) -> None:
        """Log executed trade.

        Args:
            order: Executed order
        """
        trade = {
            "timestamp": order.timestamp,
            "symbol": order.symbol,
            "order_type": order.order_type,
            "size": order.size,
            "price": order.price,
            "reason": order.reason,
            "balance": self.current_balance,
        }
        self.trades.append(trade)

    def _log_closed_trade(
        self,
        order: Order,
        pnl: float,
        entry_price: float,
    ) -> None:
        """Log closed trade with P&L.

        Args:
            order: Closing order
            pnl: Realized P&L
            entry_price: Entry price for reference
        """
        closed_trade = {
            "timestamp": order.timestamp,
            "symbol": order.symbol,
            "entry_price": entry_price,
            "exit_price": order.price,
            "size": order.size,
            "pnl": pnl,
            "pnl_pct": (pnl / (entry_price * order.size)) * 100.0,
        }
        self.closed_trades.append(closed_trade)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get complete trade history.

        Returns:
            List of all executed trades
        """
        return self.trades.copy()

    def get_closed_trades(self) -> List[Dict[str, Any]]:
        """Get closed trade results.

        Returns:
            List of closed trades with P&L
        """
        return self.closed_trades.copy()

    def get_portfolio_stats(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get portfolio statistics.

        Args:
            current_prices: Dictionary of symbol -> current price

        Returns:
            Dictionary with portfolio metrics
        """
        total_value = self.current_balance + self.get_total_exposure(current_prices)
        total_pnl = self.get_total_pnl(current_prices)
        total_pnl_pct = (total_pnl / self.starting_capital) * 100.0

        return {
            "starting_capital": self.starting_capital,
            "current_balance": self.current_balance,
            "total_exposure": self.get_total_exposure(current_prices),
            "portfolio_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "num_positions": len(self.positions),
        }
