"""Strategy logic for signal generation and regime detection"""

from enum import Enum
from typing import Dict, Optional, Any

import pandas as pd
import numpy as np


class Regime(Enum):
    """Market regime enumeration"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class Signal:
    """Trade signal representation"""

    def __init__(
        self,
        signal_type: str,
        signal_value: float,
        order_type: str,
        reason: str = "",
    ):
        """Initialize signal.

        Args:
            signal_type: ENTRY or EXIT or CLOSE_ALL
            signal_value: Z-Score or regime info
            order_type: BUY or SELL
            reason: Signal reason (e.g., "Z-Score=-2.5")
        """
        self.signal_type = signal_type
        self.signal_value = signal_value
        self.order_type = order_type
        self.reason = reason

    def __repr__(self) -> str:
        return f"Signal({self.signal_type}, {self.order_type}, {self.signal_value})"


class Strategy:
    """Implements trend-following strategy with regime detection and Z-score signals"""

    def __init__(
        self,
        config: Dict[str, Any],
        previous_regime: Optional[Regime] = None,
    ):
        """Initialize strategy.

        Args:
            config: Configuration dictionary with strategy parameters
            previous_regime: Previous market regime (for state continuity)
        """
        self.z_entry = config.get("z_entry", 2.0)
        self.z_target = config.get("z_target", 1.0)
        self.z_stop = config.get("z_stop", 3.0)
        self.trend_window = config.get("trend_window", 200)
        self.short_window = config.get("short_window", 20)

        self.previous_regime = previous_regime or Regime.NEUTRAL
        self.current_regime = self.previous_regime

    def detect_regime(self, df: pd.DataFrame) -> Regime:
        """Detect market regime based on ratio vs trend MA.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Regime (BULLISH, BEARISH, or NEUTRAL)

        Raises:
            TypeError: If df is None
            ValueError: If required columns missing or insufficient data
        """
        if df is None:
            raise TypeError("DataFrame cannot be None")

        if "close" not in df.columns:
            raise ValueError("Missing required column: close")

        if len(df) < self.trend_window:
            raise ValueError(
                f"Insufficient data for MA calculation: {len(df)} < {self.trend_window}"
            )

        # Calculate trend MA
        trend_ma = df["close"].rolling(window=self.trend_window).mean().iloc[-1]
        ratio = df["close"].iloc[-1]

        # Determine regime
        if ratio > trend_ma:
            self.current_regime = Regime.BULLISH
        elif ratio < trend_ma:
            self.current_regime = Regime.BEARISH
        else:
            # Maintain previous regime when ratio ≈ MA
            self.current_regime = self.previous_regime

        return self.current_regime

    def calculate_zscore(self, df: pd.DataFrame) -> float:
        """Calculate Z-Score for current price against short MA.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Z-Score value

        Raises:
            ValueError: If insufficient data
        """
        if len(df) < self.short_window:
            raise ValueError(
                f"Insufficient data for Z-Score calculation: {len(df)} < {self.short_window}"
            )

        short_ma = df["close"].rolling(window=self.short_window).mean().iloc[-1]
        std = df["close"].rolling(window=self.short_window).std().iloc[-1]

        # Avoid division by zero
        if std == 0 or np.isnan(std):
            return 0.0

        zscore = (df["close"].iloc[-1] - short_ma) / std
        return float(zscore)

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime: Regime,
    ) -> Optional[Signal]:
        """Generate trading signal based on regime and Z-Score.

        Args:
            df: DataFrame with OHLCV data
            regime: Current market regime

        Returns:
            Signal if conditions met, None otherwise

        Raises:
            ValueError: If data validation fails
        """
        if len(df) < max(self.trend_window, self.short_window):
            return None

        try:
            zscore = self.calculate_zscore(df)
        except ValueError:
            return None

        if regime == Regime.BULLISH:
            # Bullish regime: look for LONG opportunities
            if zscore < -self.z_entry:
                return Signal(
                    signal_type="ENTRY",
                    signal_value=zscore,
                    order_type="BUY",
                    reason=f"Z-Score={zscore:.2f}",
                )
            elif zscore > self.z_target:
                return Signal(
                    signal_type="EXIT",
                    signal_value=zscore,
                    order_type="BUY",
                    reason=f"Z-Score={zscore:.2f}",
                )
            elif abs(zscore) > self.z_stop:
                return Signal(
                    signal_type="EXIT",
                    signal_value=zscore,
                    order_type="BUY",
                    reason=f"Stop Loss (Z-Score={zscore:.2f})",
                )

        elif regime == Regime.BEARISH:
            # Bearish regime: look for SHORT opportunities
            if zscore > self.z_entry:
                return Signal(
                    signal_type="ENTRY",
                    signal_value=zscore,
                    order_type="SELL",
                    reason=f"Z-Score={zscore:.2f}",
                )
            elif zscore < -self.z_target:
                return Signal(
                    signal_type="EXIT",
                    signal_value=zscore,
                    order_type="SELL",
                    reason=f"Z-Score={zscore:.2f}",
                )
            elif abs(zscore) > self.z_stop:
                return Signal(
                    signal_type="EXIT",
                    signal_value=zscore,
                    order_type="SELL",
                    reason=f"Stop Loss (Z-Score={zscore:.2f})",
                )

        return None

    def handle_regime_change(
        self,
        previous_regime: Regime,
        new_regime: Regime,
    ) -> Optional[Signal]:
        """Generate CLOSE ALL signal on regime change.

        Args:
            previous_regime: Previous market regime
            new_regime: New market regime

        Returns:
            CLOSE ALL signal if regime changed, None otherwise
        """
        if previous_regime != new_regime and new_regime != Regime.NEUTRAL:
            return Signal(
                signal_type="CLOSE_ALL",
                signal_value=0.0,
                order_type="BOTH",
                reason=f"Regime change: {previous_regime.value} → {new_regime.value}",
            )

        return None

    def process_bar(
        self,
        df: pd.DataFrame,
    ) -> Optional[Signal]:
        """Process a complete bar and generate signals.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Signal if conditions met, None otherwise
        """
        try:
            # Detect current regime
            regime = self.detect_regime(df)

            # Check for regime change
            regime_signal = self.handle_regime_change(self.previous_regime, regime)
            if regime_signal:
                self.previous_regime = regime
                return regime_signal

            # Generate trading signal
            signal = self.generate_signal(df, regime)

            # Update previous regime
            self.previous_regime = regime

            return signal

        except (ValueError, TypeError):
            return None
