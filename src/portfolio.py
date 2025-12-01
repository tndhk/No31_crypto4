"""Portfolio optimization for Risk Parity and Beta Hedge"""

from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


class RiskParityWeights:
    """Risk Parity weight representation"""

    def __init__(self, weights: Dict[str, float]):
        """Initialize Risk Parity weights.

        Args:
            weights: Dictionary of symbol -> weight
        """
        self.weights = weights

    def __getitem__(self, key: str) -> float:
        """Get weight for symbol."""
        return self.weights[key]

    def __repr__(self) -> str:
        return f"RiskParityWeights({self.weights})"


class HedgeRatio:
    """Hedge ratio representation"""

    def __init__(self, hedge_ratio: float, sharpe_ratio: float = 0.0):
        """Initialize Hedge Ratio.

        Args:
            hedge_ratio: Hedge ratio multiplier (e.g., 1.0 * beta)
            sharpe_ratio: Sharpe ratio of hedged portfolio
        """
        self.hedge_ratio = hedge_ratio
        self.sharpe_ratio = sharpe_ratio

    def __repr__(self) -> str:
        return f"HedgeRatio({self.hedge_ratio:.2f}, Sharpe={self.sharpe_ratio:.2f})"


class PortfolioOptimizer:
    """Optimizes portfolio weights using Risk Parity and Beta Hedge strategies"""

    # Grid search multipliers for hedge ratio
    HEDGE_RATIO_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]

    def __init__(self, min_data_days: int = 30):
        """Initialize PortfolioOptimizer.

        Args:
            min_data_days: Minimum days of data required
        """
        self.min_data_days = min_data_days

    def calculate_risk_parity_weights(
        self,
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate Risk Parity weights based on volatility.

        Args:
            returns: DataFrame with symbol returns (each column is a symbol)

        Returns:
            Dictionary of symbol -> weight

        Raises:
            TypeError: If returns is None
            ValueError: If insufficient data or empty DataFrame
        """
        if returns is None:
            raise TypeError("Returns DataFrame cannot be None")

        if returns.empty or len(returns.columns) == 0:
            raise ValueError("Returns DataFrame is empty or has no columns")

        if len(returns) < self.min_data_days:
            raise ValueError(
                f"Insufficient data: {len(returns)} days < {self.min_data_days} minimum"
            )

        # Calculate volatility for each symbol
        volatilities = returns.std()

        # Handle zero volatility
        volatilities = volatilities.replace(0, volatilities[volatilities > 0].min() if any(volatilities > 0) else 1e-6)

        # Check for any NaN volatilities
        if volatilities.isna().any():
            raise ValueError("NaN values detected in volatility calculation")

        # Calculate inverse volatility weights
        inverse_volatilities = 1.0 / volatilities

        # Normalize to sum to 1.0
        weights = inverse_volatilities / inverse_volatilities.sum()

        return weights.to_dict()

    def calculate_beta(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Calculate beta of portfolio relative to benchmark.

        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns

        Returns:
            Beta value

        Raises:
            ValueError: If insufficient data
        """
        if len(portfolio_returns) < 30:
            raise ValueError(f"Insufficient data for beta calculation: {len(portfolio_returns)} < 30")

        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        # Avoid division by zero
        if benchmark_variance == 0:
            return 1.0

        beta = covariance / benchmark_variance
        return float(beta)

    def optimize_hedge_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        min_sharpe: float = 0.0,
    ) -> float:
        """Optimize hedge ratio using grid search on Sharpe ratio.

        Args:
            portfolio_returns: Array of portfolio returns
            benchmark_returns: Array of benchmark returns
            min_sharpe: Minimum Sharpe ratio threshold

        Returns:
            Optimal hedge ratio (multiplier for beta)
        """
        # Calculate base beta
        beta = self.calculate_beta(portfolio_returns, benchmark_returns)

        # Grid search over hedge ratio multipliers
        best_hedge_ratio = 1.0 * beta
        best_sharpe = float('-inf')

        for multiplier in self.HEDGE_RATIO_GRID:
            hedge_ratio = multiplier * beta

            # Calculate hedged returns
            hedged_returns = portfolio_returns - hedge_ratio * benchmark_returns

            # Calculate Sharpe ratio
            sharpe = self._calculate_sharpe_ratio(hedged_returns)

            # Track best hedge ratio
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_hedge_ratio = hedge_ratio

        return float(best_hedge_ratio)

    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Avoid division by zero
        if std_return == 0:
            return 0.0

        # Annualize (assuming daily returns)
        sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)

        return float(sharpe)

    def rebalance_weights(
        self,
        current_weights: Dict[str, float],
        new_returns: pd.DataFrame,
        rebalance_interval: str = "monthly",
    ) -> Dict[str, float]:
        """Rebalance portfolio weights based on new data.

        Args:
            current_weights: Current portfolio weights
            new_returns: New returns data
            rebalance_interval: Rebalancing frequency ("daily", "monthly", "quarterly")

        Returns:
            Updated weights
        """
        # Recalculate Risk Parity weights with new data
        new_weights = self.calculate_risk_parity_weights(new_returns)

        return new_weights

    def create_hedge_portfolio(
        self,
        asset_weights: Dict[str, float],
        hedge_ratio: float,
        btc_symbol: str = "BTC",
    ) -> Dict[str, float]:
        """Create hedged portfolio allocation.

        Args:
            asset_weights: Risk Parity weights for altcoins
            hedge_ratio: Hedge ratio for BTC hedge
            btc_symbol: Symbol for hedge asset (usually BTC)

        Returns:
            Portfolio weights including hedge
        """
        # Reduce altcoin weights to allocate to hedge
        hedge_allocation = min(hedge_ratio, 0.5)  # Cap hedge at 50% max

        # Adjust altcoin weights
        hedged_weights = {}
        for symbol, weight in asset_weights.items():
            hedged_weights[symbol] = weight * (1 - hedge_allocation)

        # Add BTC hedge
        hedged_weights[btc_symbol] = hedge_allocation

        # Normalize to sum to 1.0
        total = sum(hedged_weights.values())
        if total > 0:
            hedged_weights = {k: v / total for k, v in hedged_weights.items()}

        return hedged_weights
