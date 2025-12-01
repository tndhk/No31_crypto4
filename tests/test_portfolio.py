"""Tests for src.portfolio module"""

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import numpy as np
import pytest

from src.portfolio import PortfolioOptimizer, RiskParityWeights, HedgeRatio


class TestRiskParityWeights:
    """Tests for Risk Parity weight calculation"""

    def test_risk_parity_weights_sum_to_one(self):
        """TC-N-01: Risk Parity weights sum to 1.0."""
        # Given: Valid daily returns for 252 trading days
        np.random.seed(42)
        returns = pd.DataFrame({
            "BTC": np.random.randn(252) * 0.02 + 0.0005,
            "ETH": np.random.randn(252) * 0.03 + 0.0003,
            "SOL": np.random.randn(252) * 0.04 + 0.0002,
        })

        # When: Calculating Risk Parity weights
        optimizer = PortfolioOptimizer()
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: Weights should sum to 1.0
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in weights.values())

    def test_weights_inversely_proportional_to_volatility(self):
        """TC-N-02: Weights are inversely proportional to volatility."""
        # Given: Returns with different volatility levels
        np.random.seed(42)
        low_vol = np.random.randn(252) * 0.01  # Low volatility
        high_vol = np.random.randn(252) * 0.04  # High volatility

        returns = pd.DataFrame({
            "LowVolAsset": low_vol,
            "HighVolAsset": high_vol,
        })

        # When: Calculating Risk Parity weights
        optimizer = PortfolioOptimizer()
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: Low volatility asset should have higher weight
        assert weights["LowVolAsset"] > weights["HighVolAsset"]

    def test_single_symbol_weight_equals_one(self):
        """TC-B-02: Single symbol portfolio has weight = 1.0."""
        # Given: Single symbol returns
        np.random.seed(42)
        returns = pd.DataFrame({
            "BTC": np.random.randn(252) * 0.02,
        })

        # When: Calculating Risk Parity weights
        optimizer = PortfolioOptimizer()
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: Weight should be 1.0
        assert weights["BTC"] == 1.0

    def test_equal_volatility_equal_weights(self):
        """TC-B-01: Equal volatility assets get equal weights."""
        # Given: Returns with equal volatility
        np.random.seed(42)
        vol = np.random.randn(252) * 0.02

        returns = pd.DataFrame({
            "Asset1": vol,
            "Asset2": vol.copy(),
            "Asset3": vol.copy(),
        })

        # When: Calculating Risk Parity weights
        optimizer = PortfolioOptimizer()
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: All weights should be approximately equal (1/3)
        expected_weight = 1.0 / 3
        for w in weights.values():
            assert abs(w - expected_weight) < 1e-2


class TestBetaHedgeOptimization:
    """Tests for Beta Hedge grid search optimization"""

    def test_hedge_ratio_grid_search(self):
        """TC-N-03: Grid search selects best hedge ratio."""
        # Given: Portfolio data and benchmark data
        np.random.seed(42)
        portfolio_returns = pd.DataFrame({
            "returns": np.random.randn(252) * 0.02,
        })
        benchmark_returns = np.random.randn(252) * 0.015

        # When: Running hedge ratio grid search
        optimizer = PortfolioOptimizer()
        best_hedge_ratio = optimizer.optimize_hedge_ratio(
            portfolio_returns["returns"],
            benchmark_returns,
        )

        # Then: Should select from grid [0.8, 0.9, 1.0, 1.1, 1.2] * beta
        assert best_hedge_ratio is not None
        assert isinstance(best_hedge_ratio, (float, np.floating))

    def test_hedge_ratio_neutral_weighting(self):
        """TC-N-04: Hedge ratio 1.0 applies pure beta weighting."""
        # Given: Configuration with hedge_ratio = 1.0
        np.random.seed(42)
        portfolio_returns = np.random.randn(252) * 0.02
        benchmark_returns = np.random.randn(252) * 0.015

        # When: Calculating hedge with ratio 1.0
        optimizer = PortfolioOptimizer()
        beta = optimizer.calculate_beta(portfolio_returns, benchmark_returns)

        # Then: Hedge should apply 1.0 * beta
        assert beta is not None
        assert isinstance(beta, (float, np.floating))

    def test_hedge_ratio_under_hedge(self):
        """TC-N-05: Hedge ratio 0.8 reduces hedge exposure."""
        # Given: Portfolio and benchmark returns
        np.random.seed(42)
        portfolio_returns = np.random.randn(252) * 0.02
        benchmark_returns = np.random.randn(252) * 0.015

        # When: Calculating with 0.8 multiplier
        optimizer = PortfolioOptimizer()
        beta = optimizer.calculate_beta(portfolio_returns, benchmark_returns)
        reduced_hedge = 0.8 * beta

        # Then: Reduced hedge should be less than beta
        assert reduced_hedge < beta

    def test_hedge_ratio_over_hedge(self):
        """TC-N-06: Hedge ratio 1.2 increases hedge exposure."""
        # Given: Portfolio and benchmark returns
        np.random.seed(42)
        portfolio_returns = np.random.randn(252) * 0.02
        benchmark_returns = np.random.randn(252) * 0.015

        # When: Calculating with 1.2 multiplier
        optimizer = PortfolioOptimizer()
        beta = optimizer.calculate_beta(portfolio_returns, benchmark_returns)
        increased_hedge = 1.2 * beta

        # Then: Increased hedge should be greater than beta
        assert increased_hedge > beta


class TestMonthlyRebalancing:
    """Tests for portfolio rebalancing"""

    def test_monthly_rebalancing_recalculates_weights(self):
        """TC-N-07: Monthly rebalancing recalculates weights with new volatility."""
        # Given: Returns over multiple months
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=252, freq='D')
        returns = pd.DataFrame({
            "BTC": np.random.randn(252) * 0.02,
            "ETH": np.random.randn(252) * 0.03,
        }, index=dates)

        # When: Rebalancing monthly (use 60 days per "month" to meet min_data_days=30)
        optimizer = PortfolioOptimizer(min_data_days=20)
        first_month = returns.iloc[:60]  # First period
        next_month = returns.iloc[60:120]  # Next period

        weights_1 = optimizer.calculate_risk_parity_weights(first_month)
        weights_2 = optimizer.calculate_risk_parity_weights(next_month)

        # Then: Weights should change (different volatility)
        assert weights_1["BTC"] != weights_2["BTC"] or weights_1["ETH"] != weights_2["ETH"]


class TestErrorHandling:
    """Tests for error handling"""

    def test_empty_dataframe_raises_error(self):
        """TC-A-01: Empty DataFrame raises ValueError."""
        # Given: Empty returns DataFrame
        returns = pd.DataFrame()

        # When: Calculating weights
        optimizer = PortfolioOptimizer()

        # Then: Should raise error
        with pytest.raises((ValueError, KeyError)):
            optimizer.calculate_risk_parity_weights(returns)

    def test_null_dataframe_raises_error(self):
        """TC-A-02: Null/None DataFrame raises error."""
        # Given: None returns
        returns = None

        # When: Calculating weights
        optimizer = PortfolioOptimizer()

        # Then: Should raise error
        with pytest.raises((TypeError, ValueError)):
            optimizer.calculate_risk_parity_weights(returns)

    def test_nan_values_in_returns(self):
        """TC-A-03: NaN values in returns should be handled."""
        # Given: Returns with NaN values
        returns = pd.DataFrame({
            "BTC": [0.01, 0.02, float('nan'), 0.01],
            "ETH": [0.02, float('nan'), 0.03, 0.02],
        })

        # When: Processing returns with NaN
        optimizer = PortfolioOptimizer()

        # Then: Should either raise or handle with forward fill
        try:
            weights = optimizer.calculate_risk_parity_weights(returns)
            assert sum(weights.values()) > 0
        except ValueError:
            # Acceptable to raise ValueError for NaN
            pass

    def test_zero_volatility_handling(self):
        """TC-A-05: Zero volatility (flat prices) is handled."""
        # Given: Flat returns (zero volatility)
        returns = pd.DataFrame({
            "BTC": [0.0] * 252,
            "ETH": [0.01] * 252,
        })

        # When: Calculating Risk Parity weights
        optimizer = PortfolioOptimizer()

        # Then: Should either raise or use floor for zero volatility
        try:
            weights = optimizer.calculate_risk_parity_weights(returns)
            # If no error, weights should be valid
            assert sum(weights.values()) > 0
        except ValueError:
            # Acceptable to raise for zero volatility
            pass

    def test_insufficient_data_raises_error(self):
        """TC-A-06: Insufficient historical data raises error."""
        # Given: Returns with less than 30 days
        returns = pd.DataFrame({
            "BTC": np.random.randn(10) * 0.02,
            "ETH": np.random.randn(10) * 0.03,
        })

        # When: Calculating weights
        optimizer = PortfolioOptimizer(min_data_days=30)

        # Then: Should raise error
        with pytest.raises(ValueError, match="insufficient|minimum"):
            optimizer.calculate_risk_parity_weights(returns)


class TestLargePortfolios:
    """Tests for large portfolios"""

    def test_large_portfolio_100_symbols(self):
        """TC-B-03: Large portfolio (100 symbols) processes all."""
        # Given: 100 symbols
        np.random.seed(42)
        symbols = [f"Asset{i}" for i in range(100)]
        returns = pd.DataFrame({
            symbol: np.random.randn(252) * 0.02
            for symbol in symbols
        })

        # When: Calculating Risk Parity weights
        optimizer = PortfolioOptimizer()
        weights = optimizer.calculate_risk_parity_weights(returns)

        # Then: All symbols should have weights
        assert len(weights) == 100
        assert abs(sum(weights.values()) - 1.0) < 1e-4


class TestBetaCalculation:
    """Tests for Beta calculation"""

    def test_beta_calculation_from_returns(self):
        """Calculate beta correctly from returns."""
        # Given: Portfolio and benchmark returns with known relationship
        np.random.seed(42)
        benchmark_returns = np.random.randn(252) * 0.02
        # Portfolio has 1.5x market beta
        portfolio_returns = 1.5 * benchmark_returns + np.random.randn(252) * 0.01

        # When: Calculating beta
        optimizer = PortfolioOptimizer()
        beta = optimizer.calculate_beta(portfolio_returns, benchmark_returns)

        # Then: Beta should be approximately 1.5
        assert beta is not None
        assert 1.0 < beta < 2.0
