"""Tests for src.data_loader module"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pandas as pd
import pytest

from src.data_loader import DataLoader, DataValidator


class TestDataValidator:
    """Tests for data validation"""

    def test_validate_ohlcv_data_valid(self):
        """TC-N-04: Valid OHLCV data passes validation."""
        # Given: DataFrame with all required OHLCV columns
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000.0 + i * 10 for i in range(100)],
        })

        # When: Validating data
        validator = DataValidator()

        # Then: Should pass validation without error
        try:
            validator.validate(df)
        except Exception as e:
            pytest.fail(f"Validation failed unexpectedly: {e}")

    def test_validate_ohlcv_missing_column(self):
        """TC-A-04: Missing required OHLCV column raises error."""
        # Given: DataFrame missing 'close' column
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "volume": [1000.0 + i * 10 for i in range(100)],
        })

        # When: Validating data
        validator = DataValidator()

        # Then: Should raise ValueError
        with pytest.raises(ValueError, match="Missing required column"):
            validator.validate(df)

    def test_validate_ohlcv_empty_dataframe(self):
        """TC-B-03: Empty DataFrame raises error."""
        # Given: Empty DataFrame
        df = pd.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })

        # When: Validating data
        validator = DataValidator()

        # Then: Should raise ValueError
        with pytest.raises(ValueError, match="No data"):
            validator.validate(df)

    def test_validate_ohlcv_nan_values(self):
        """TC-A-05: NaN values in data."""
        # Given: DataFrame with NaN in price columns
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "open": [100.0, float('nan'), 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8, 100.9],
            "high": [101.0, 101.1, 101.2, 101.3, 101.4, 101.5, 101.6, 101.7, 101.8, 101.9],
            "low": [99.0, 99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9],
            "close": [100.5, 100.6, 100.7, 100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4],
            "volume": [1000.0 + i * 10 for i in range(10)],
        })

        # When: Validating data with NaN
        validator = DataValidator()

        # Then: Should detect NaN and handle appropriately
        with pytest.raises(ValueError, match="NaN|missing"):
            validator.validate(df)


class TestDataInterpolation:
    """Tests for data interpolation and gap handling"""

    def test_interpolate_gap_1_candle(self):
        """TC-N-05: Gap with 1 missing candle is filled with ffill."""
        # Given: DataFrame with 1 gap
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"]),
            "close": [100.0, 101.0, 103.0, 104.0],
        })

        # When: Interpolating gaps
        validator = DataValidator()

        # Then: Gap should be filled
        result = validator.interpolate(df)
        assert len(result) == 5
        assert result.iloc[2]["close"] == 101.0  # ffill value

    def test_interpolate_gap_2_candles(self):
        """TC-N-06: Gap with 2 missing candles is filled with ffill."""
        # Given: DataFrame with 2 gaps
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]),
            "close": [100.0, 101.0, 104.0, 105.0],
        })

        # When: Interpolating gaps
        validator = DataValidator()

        # Then: Gaps should be filled
        result = validator.interpolate(df)
        assert len(result) == 6
        assert result.iloc[2]["close"] == 101.0  # ffill
        assert result.iloc[3]["close"] == 101.0  # ffill

    def test_interpolate_gap_3_candles(self):
        """TC-N-07: Gap with 3 missing candles is filled with ffill (boundary)."""
        # Given: DataFrame with 3 gaps (maximum allowed)
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-06", "2024-01-07"]),
            "close": [100.0, 101.0, 105.0, 106.0],
        })

        # When: Interpolating gaps
        validator = DataValidator()

        # Then: Gaps should be filled (3 is the maximum)
        result = validator.interpolate(df)
        assert len(result) == 7
        assert result.iloc[2]["close"] == 101.0

    def test_interpolate_gap_4_candles_exceeds_limit(self):
        """TC-A-01: Gap with 4 missing candles exceeds limit."""
        # Given: DataFrame with 4 gaps (exceeds maximum of 3)
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-07", "2024-01-08"]),
            "close": [100.0, 101.0, 106.0, 107.0],
        })

        # When: Interpolating gaps
        validator = DataValidator()

        # Then: Should raise ValueError (gap too large)
        with pytest.raises(ValueError, match="gap|interpolation|consecutive"):
            validator.interpolate(df)

    def test_interpolate_gap_10_candles_exceeds_limit(self):
        """TC-A-02: Gap with 10 missing candles far exceeds limit."""
        # Given: DataFrame with 10 gaps
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-13", "2024-01-14"]),
            "close": [100.0, 101.0, 111.0, 112.0],
        })

        # When: Interpolating gaps
        validator = DataValidator()

        # Then: Should raise ValueError
        with pytest.raises(ValueError, match="gap|interpolation|consecutive"):
            validator.interpolate(df)


class TestDataLoaderFetch:
    """Tests for data fetching and retry logic"""

    def test_fetch_ohlcv_success(self):
        """TC-N-01: Valid OHLCV fetch succeeds."""
        # Given: Valid symbol list
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "raw"
            data_path.mkdir()

            loader = DataLoader(data_dir=str(tmpdir), min_data_days=10)

            # Mock CCXT to return valid data (30 candles)
            mock_ohlcv = [
                [1704067200000 + i * 86400000, 100.0 + i * 0.1, 101.0 + i * 0.1, 99.0 + i * 0.1, 100.5 + i * 0.1, 1000.0]
                for i in range(30)
            ]

            # When: Fetching data
            with patch.object(loader, '_fetch_from_exchange', return_value=mock_ohlcv):
                result = loader.fetch_ohlcv("BTC/USDT", limit=30)

            # Then: Should return DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 30
            assert "close" in result.columns

    def test_fetch_ohlcv_retry_on_failure(self):
        """TC-N-08: Network retry succeeds on second attempt."""
        # Given: DataLoader with failed first attempt
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir), min_data_days=10)

            mock_ohlcv = [
                [1704067200000 + i * 86400000, 100.0 + i * 0.1, 101.0 + i * 0.1, 99.0 + i * 0.1, 100.5 + i * 0.1, 1000.0]
                for i in range(30)
            ]

            # When: First call fails, second succeeds
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("Network error")
                return mock_ohlcv

            # Then: Should retry and succeed
            with patch.object(loader, '_fetch_from_exchange', side_effect=side_effect):
                result = loader.fetch_ohlcv("BTC/USDT", limit=30)
                assert isinstance(result, pd.DataFrame)
                assert call_count[0] == 2  # Verify 2 attempts made

    def test_fetch_ohlcv_multiple_retries(self):
        """TC-N-09: Multiple retries (2-3 attempts) succeed on third attempt."""
        # Given: DataLoader with multiple failed attempts
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir), min_data_days=10)

            mock_ohlcv = [
                [1704067200000 + i * 86400000, 100.0 + i * 0.1, 101.0 + i * 0.1, 99.0 + i * 0.1, 100.5 + i * 0.1, 1000.0]
                for i in range(30)
            ]

            # When: First two calls fail, third succeeds
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 3:
                    raise Exception("Network error")
                return mock_ohlcv

            # Then: Should retry 3 times and succeed
            with patch.object(loader, '_fetch_from_exchange', side_effect=side_effect):
                result = loader.fetch_ohlcv("BTC/USDT", limit=30)
                assert isinstance(result, pd.DataFrame)
                assert call_count[0] == 3

    def test_fetch_ohlcv_max_retries_exceeded(self):
        """TC-A-03: All network retries fail, exceeds max_retries."""
        # Given: DataLoader with persistent failures
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir), max_retries=3, min_data_days=10)

            # When: All attempts fail
            with patch.object(loader, '_fetch_from_exchange', side_effect=Exception("Network error")):
                # Then: Should raise exception after max_retries
                with pytest.raises(Exception, match="Network error|failed"):
                    loader.fetch_ohlcv("BTC/USDT", limit=30)

    def test_fetch_ohlcv_empty_symbol_list(self):
        """TC-A-06: Empty symbol list raises error."""
        # Given: Empty symbol parameter
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir))

            # When: Fetching with empty symbol
            # Then: Should raise ValueError
            with pytest.raises((ValueError, TypeError)):
                loader.fetch_ohlcv("", limit=100)

    def test_fetch_ohlcv_invalid_symbol(self):
        """TC-A-07: Invalid symbol format raises error."""
        # Given: Invalid symbol (e.g., very long string)
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir), min_data_days=10)

            # When: Attempting to fetch invalid symbol
            with patch.object(loader, '_fetch_from_exchange', side_effect=ValueError("Invalid symbol")):
                # Then: Should raise Exception (because retries occur, wrapping the ValueError)
                with pytest.raises(Exception, match="Failed|Invalid"):
                    loader.fetch_ohlcv("INVALID_SYMBOL_12345", limit=100)

    def test_fetch_ohlcv_insufficient_data(self):
        """TC-A-08: Insufficient historical data raises error."""
        # Given: Request for more data than available
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir), min_data_days=30)

            mock_ohlcv = [
                [1704067200000 + i * 86400000, 100.0 + i * 0.1, 101.0 + i * 0.1, 99.0 + i * 0.1, 100.5 + i * 0.1, 1000.0]
                for i in range(10)  # Only 10 days of data
            ]

            # When: Attempting to fetch minimum 30 days
            with patch.object(loader, '_fetch_from_exchange', return_value=mock_ohlcv):
                # Then: Should raise ValueError (insufficient data)
                with pytest.raises(ValueError, match="insufficient|data"):
                    loader.fetch_ohlcv("BTC/USDT", limit=10)


class TestDataLoaderInitialFetch:
    """Tests for initial data fetch behavior"""

    def test_initial_fetch_2_years(self):
        """TC-N-02: Initial fetch (no cached data) fetches 2 years."""
        # Given: No cached data exists
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir))

            # Generate 2 years of mock OHLCV data
            two_years_ago = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)
            mock_ohlcv = [
                [two_years_ago + i * 86400000, 100.0 + i * 0.01, 101.0 + i * 0.01, 99.0 + i * 0.01, 100.5 + i * 0.01, 1000.0]
                for i in range(730)
            ]

            # When: Performing initial fetch
            with patch.object(loader, '_fetch_from_exchange', return_value=mock_ohlcv):
                result = loader.fetch_ohlcv("BTC/USDT", limit=730)

            # Then: Should return 2 years of data
            assert isinstance(result, pd.DataFrame)
            assert len(result) >= 730

    def test_incremental_fetch_new_data(self):
        """TC-N-03: Incremental fetch (cached data exists) fetches only new data."""
        # Given: Cached data from 100 days ago
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir), min_data_days=5)

            # Create cached data file
            cache_path = Path(tmpdir) / "raw" / "BTC_USDT.parquet"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            old_date = datetime.now() - timedelta(days=100)
            cached_df = pd.DataFrame({
                "timestamp": pd.date_range(start=old_date - timedelta(days=30), periods=30, freq='D'),
                "close": [100.0 + i * 0.1 for i in range(30)],
            })
            cached_df.to_parquet(cache_path)

            # When: Performing incremental fetch
            new_ohlcv = [
                [int((datetime.now() - timedelta(days=i)).timestamp() * 1000), 100.0 + i * 0.1, 101.0 + i * 0.1, 99.0 + i * 0.1, 100.5 + i * 0.1, 1000.0]
                for i in range(10)
            ]

            with patch.object(loader, '_fetch_from_exchange', return_value=new_ohlcv):
                result = loader.fetch_ohlcv("BTC/USDT", limit=10)

            # Then: Should return new data only
            assert isinstance(result, pd.DataFrame)


class TestDataLoaderNullCases:
    """Tests for NULL/None boundary cases"""

    def test_fetch_ohlcv_null_symbol(self):
        """TC-B-01: Null/None symbol raises error."""
        # Given: None symbol parameter
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(data_dir=str(tmpdir))

            # When: Fetching with None
            # Then: Should raise TypeError or ValueError
            with pytest.raises((TypeError, ValueError)):
                loader.fetch_ohlcv(None, limit=100)

    def test_validate_null_dataframe(self):
        """TC-B-02: Null/None DataFrame raises error."""
        # Given: None DataFrame
        validator = DataValidator()

        # When: Validating None
        # Then: Should raise TypeError or ValueError
        with pytest.raises((TypeError, ValueError)):
            validator.validate(None)
