"""Walk-Forward Optimization backtester with metrics calculation"""

from typing import Dict, Optional, Any, Tuple, List
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class WFOSplit:
    """Walk-Forward Optimization window splitter"""

    def __init__(
        self,
        df: pd.DataFrame,
        is_window_days: int = 730,
        oos_window_days: int = 90,
        lookback_days: int = 0,
    ):
        """Initialize WFO split.

        Args:
            df: DataFrame with OHLCV data
            is_window_days: In-Sample window size (days)
            oos_window_days: Out-of-Sample window size (days)
            lookback_days: Lookback period for OOS indicator calculation (days)

        Raises:
            TypeError: If df is None
            ValueError: If insufficient data or missing columns
        """
        if df is None:
            raise TypeError("DataFrame cannot be None")

        if df.empty or len(df) == 0:
            raise ValueError("DataFrame is empty")

        if "timestamp" not in df.columns:
            raise ValueError("Missing required column: timestamp")

        min_required = is_window_days + oos_window_days
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} days < {min_required} days minimum "
                f"(IS={is_window_days} + OOS={oos_window_days})"
            )

        # Split into IS and OOS windows
        self.is_df = df.iloc[:is_window_days].reset_index(drop=True)
        
        # OOS window needs lookback data for indicator calculation
        oos_start_idx = max(0, is_window_days - lookback_days)
        self.oos_df = df.iloc[oos_start_idx:is_window_days + oos_window_days].reset_index(drop=True)
        
        self.is_window_days = is_window_days
        self.oos_window_days = oos_window_days
        self.lookback_days = is_window_days - oos_start_idx


class BacktestMetrics:
    """Backtest performance metrics calculation"""

    def __init__(
        self,
        returns: np.ndarray,
        start_value: float,
        end_value: float,
        risk_free_rate: float = 0.0,
    ):
        """Initialize metrics calculator.

        Args:
            returns: Array of daily returns
            start_value: Starting capital
            end_value: Ending capital
            risk_free_rate: Annual risk-free rate (default 0%)

        Raises:
            TypeError: If returns is None
            ValueError: If insufficient data
        """
        if returns is None:
            raise TypeError("Returns cannot be None")

        if len(returns) < 2:
            raise ValueError(f"Insufficient returns data: {len(returns)} < 2")

        self.returns = returns
        self.start_value = start_value
        self.end_value = end_value
        self.risk_free_rate = risk_free_rate

        # Calculate metrics
        self.total_return_pct = ((end_value - start_value) / start_value) * 100.0
        self.sharpe_ratio = self._calculate_sharpe_ratio()
        self.max_drawdown_pct = self._calculate_max_drawdown()

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio.

        Returns:
            Sharpe ratio (annualized)
        """
        if len(self.returns) < 2:
            return 0.0

        mean_return = np.mean(self.returns)
        std_return = np.std(self.returns)

        # Avoid division by zero
        if std_return == 0 or np.isnan(std_return):
            return 0.0

        # Annualize (assuming daily returns, 252 trading days)
        sharpe = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)
        return float(sharpe)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage.

        Returns:
            Maximum drawdown as percentage (0-100)
        """
        if len(self.returns) < 2:
            return 0.0

        # Build equity curve from returns
        equity = np.cumprod(1 + self.returns) * self.start_value

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown at each point
        drawdown = (equity - running_max) / running_max

        # Return maximum drawdown as percentage
        max_dd_pct = np.abs(np.min(drawdown)) * 100.0
        return float(max_dd_pct)

    def __repr__(self) -> str:
        return (
            f"BacktestMetrics(Sharpe={self.sharpe_ratio:.2f}, "
            f"DD={self.max_drawdown_pct:.2f}%, Return={self.total_return_pct:.2f}%)"
        )


class Backtester:
    """Walk-Forward Optimization backtester"""

    def __init__(
        self,
        config: Dict[str, Any],
        starting_capital: float = 10000.0,
    ):
        """Initialize Backtester.

        Args:
            config: Configuration dictionary with strategy, execution, validation sections
            starting_capital: Starting capital for backtest (default $10,000)

        Raises:
            TypeError: If config is None
            ValueError: If config is invalid
        """
        if config is None:
            raise TypeError("Config cannot be None")

        self.config = config
        self.starting_capital = starting_capital

    def _simulate_strategy_trades(self, df: pd.DataFrame) -> np.ndarray:
        """Simulate trades based on strategy signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Array of daily returns including signal-based trades
        """
        from src.strategy import Strategy

        strategy_config = self.config.get("strategy", {})
        strategy = Strategy(strategy_config)

        # Initialize returns array
        returns = np.zeros(len(df))

        # Track position state
        current_position = None  # 'LONG' or 'SHORT'
        entry_price = None
        entry_idx = None
        signal_count = 0

        # Iterate through data to generate signals
        for i in range(max(strategy_config.get("trend_window", 200),
                          strategy_config.get("short_window", 20)), len(df)):
            window_df = df.iloc[:i+1].copy()

            try:
                # Detect regime
                regime = strategy.detect_regime(window_df)

                # Generate signal
                signal = strategy.generate_signal(window_df, regime)

                if signal and signal.signal_type == "ENTRY":
                    signal_count += 1
                    current_price = df["close"].iloc[i]

                    # Execute entry
                    if signal.order_type == "BUY":
                        current_position = "LONG"
                        entry_price = current_price
                        entry_idx = i
                    elif signal.order_type == "SELL":
                        current_position = "SHORT"
                        entry_price = current_price
                        entry_idx = i

                elif signal and signal.signal_type == "EXIT":
                    # Exit trade
                    if current_position:
                        current_price = df["close"].iloc[i]
                        if current_position == "LONG":
                            trade_return = (current_price - entry_price) / entry_price
                        else:  # SHORT
                            trade_return = (entry_price - current_price) / entry_price

                        # Apply Leverage
                        leverage = self.config.get("execution", {}).get("leverage", 1.0)
                        trade_return *= leverage

                        # Apply Fees and Slippage (Round trip: Entry + Exit)
                        # Fees scale with leverage (applied to notional value)
                        fee_pct = self.config.get("execution", {}).get("fee_pct", 0.1)
                        slippage_pct = self.config.get("execution", {}).get("slippage_pct", 0.05)
                        
                        # Total cost percentage relative to capital
                        # (Fee + Slippage) * 2 (Entry/Exit) * Leverage
                        total_cost_pct = (fee_pct + slippage_pct) * 2 * leverage / 100.0
                        
                        trade_return -= total_cost_pct

                        # Apply the return across the trade period
                        if entry_idx is not None:
                            trade_length = i - entry_idx
                            if trade_length > 0:
                                daily_return = trade_return / trade_length
                                returns[entry_idx:i] = daily_return

                        current_position = None
                        entry_price = None
                        entry_idx = None

            except (ValueError, TypeError):
                # Skip bars with insufficient data
                continue

        # Log signal count
        logger.info(f"Strategy signals generated during backtest: {signal_count}")

        return returns

    def walk_forward_optimize(
        self,
        df: pd.DataFrame,
        is_window_days: int = 730,
        oos_window_days: int = 90,
    ) -> Dict[str, Any]:
        """Run Walk-Forward Optimization analysis.

        Args:
            df: DataFrame with OHLCV data
            is_window_days: In-Sample window size (default 730 days)
            oos_window_days: Out-of-Sample window size (default 90 days)

        Returns:
            Dictionary with IS/OOS metrics and Go/No-Go decision

        Raises:
            TypeError: If df is None
            ValueError: If insufficient data or validation fails
        """
        if df is None:
            raise TypeError("DataFrame cannot be None")

        if df.empty:
            raise ValueError("DataFrame is empty")

        # Validate required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check for NaN in price columns
        for col in ["open", "high", "low", "close"]:
            if df[col].isna().any():
                raise ValueError(f"NaN values found in {col} column")

        # Check minimum data requirement
        min_required = is_window_days + oos_window_days
        if len(df) < min_required:
            raise ValueError(
                f"Insufficient data: {len(df)} days < {min_required} days minimum"
            )

        # Determine lookback period from strategy config
        strategy_config = self.config.get("strategy", {})
        lookback = max(strategy_config.get("trend_window", 200),
                      strategy_config.get("short_window", 20))

        # Create WFO split with lookback
        wfo = WFOSplit(
            df, 
            is_window_days=is_window_days, 
            oos_window_days=oos_window_days,
            lookback_days=lookback
        )

        # Calculate strategy-based returns for each window
        # Phase 11-C: Use strategy signals for active trading instead of buy-and-hold
        logger.info("Simulating In-Sample trading with strategy signals...")
        is_returns = self._simulate_strategy_trades(wfo.is_df)

        logger.info("Simulating Out-of-Sample trading with strategy signals...")
        # OOS simulation returns full array including lookback
        oos_returns_full = self._simulate_strategy_trades(wfo.oos_df)
        # Slice to get only the actual OOS period returns
        oos_returns = oos_returns_full[wfo.lookback_days:]

        # Calculate ending values
        is_end_value = self.starting_capital * np.prod(1 + is_returns)
        oos_end_value = self.starting_capital * np.prod(1 + oos_returns)

        # Create metrics
        is_metrics = BacktestMetrics(
            returns=is_returns,
            start_value=self.starting_capital,
            end_value=is_end_value,
        )

        oos_metrics = BacktestMetrics(
            returns=oos_returns,
            start_value=self.starting_capital,
            end_value=oos_end_value,
        )

        # Determine Go/No-Go decision
        min_sharpe = self.config.get("validation", {}).get("min_sharpe_ratio", 1.0)
        max_dd = self.config.get("validation", {}).get("max_drawdown_pct", 20.0)

        is_go = (is_metrics.sharpe_ratio > min_sharpe and
                 is_metrics.max_drawdown_pct < max_dd)
        oos_go = (oos_metrics.sharpe_ratio > 0 and
                  oos_metrics.max_drawdown_pct < max_dd)

        return {
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "is_go": is_go,
            "oos_go": oos_go,
            "overall_go": is_go and oos_go,
        }

    def _calculate_returns(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate daily returns from price data.

        Args:
            df: DataFrame with close prices

        Returns:
            Array of daily returns
        """
        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]
        return returns

    def apply_fees_and_slippage(
        self,
        returns: np.ndarray,
        fee_pct: float,
        slippage_pct: float,
    ) -> np.ndarray:
        """Apply trading costs to returns.

        Args:
            returns: Original returns
            fee_pct: Trading fee as percentage
            slippage_pct: Slippage as percentage

        Returns:
            Returns adjusted for costs
        """
        # For each trade, deduct fees and slippage
        # Simplified: reduce returns by cost percentage
        cost_pct = (fee_pct + slippage_pct) / 100.0
        adjusted_returns = returns - cost_pct

        # Apply leverage if configured
        leverage = self.config.get("execution", {}).get("leverage", 1.0)
        if leverage > 1.0:
            # Leverage magnifies returns (both positive and negative)
            # Note: This is a simplified simulation that assumes constant leverage
            # and doesn't account for borrowing costs or liquidation risk
            adjusted_returns = adjusted_returns * leverage

        return adjusted_returns

    def validate_go_no_go(
        self,
        result: Dict[str, Any],
    ) -> bool:
        """Validate Go/No-Go criteria.

        Args:
            result: WFO result dictionary

        Returns:
            True if Go, False if No-Go
        """
        min_sharpe = self.config.get("validation", {}).get("min_sharpe_ratio", 1.0)
        max_dd = self.config.get("validation", {}).get("max_drawdown_pct", 20.0)

        is_metrics = result.get("is_metrics")
        if is_metrics is None:
            return False

        return (is_metrics.sharpe_ratio > min_sharpe and
                is_metrics.max_drawdown_pct < max_dd)
