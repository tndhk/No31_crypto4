"""Data loading from CCXT exchanges with validation and interpolation"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from src.utils import ExponentialBackoff

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates and interpolates OHLCV data"""

    REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    MAX_CONSECUTIVE_GAPS = 3  # Maximum consecutive missing candles allowed

    def validate(self, df: Optional[pd.DataFrame]) -> None:
        """Validate OHLCV data structure and quality.

        Args:
            df: DataFrame to validate

        Raises:
            TypeError: If df is None
            ValueError: If validation fails
        """
        if df is None:
            raise TypeError("DataFrame cannot be None")

        if df.empty or len(df) == 0:
            raise ValueError("No data: DataFrame is empty")

        # Check required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Check for NaN values in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if df[col].isna().any():
                raise ValueError(f"NaN values found in {col} column")

    def interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing data rows using forward fill.

        Args:
            df: DataFrame with potential gaps

        Returns:
            DataFrame with gaps filled

        Raises:
            ValueError: If gap exceeds maximum consecutive missing candles
        """
        if df is None:
            raise TypeError("DataFrame cannot be None")

        # Create complete date range
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        min_time = df_sorted["timestamp"].min()
        max_time = df_sorted["timestamp"].max()

        # Generate complete date range (assuming daily data)
        complete_range = pd.date_range(start=min_time, end=max_time, freq='D')

        # Reindex to complete range
        df_sorted.set_index("timestamp", inplace=True)
        df_reindexed = df_sorted.reindex(complete_range)

        # Check for consecutive NaN rows to detect gaps
        # A gap of N missing dates means N consecutive NaN rows
        nan_mask = df_reindexed.isnull().any(axis=1)
        if nan_mask.any():
            # Find consecutive NaN groups
            nan_groups = (nan_mask != nan_mask.shift()).cumsum()
            for group_id, group in df_reindexed.groupby(nan_groups):
                if nan_mask[group.index].all():  # Check if entire group is NaN
                    gap_size = len(group)
                    if gap_size > self.MAX_CONSECUTIVE_GAPS:
                        raise ValueError(
                            f"Gap exceeds maximum {self.MAX_CONSECUTIVE_GAPS} consecutive missing candles. "
                            f"Found {gap_size} consecutive missing rows."
                        )

        # Forward fill to interpolate gaps
        df_filled = df_reindexed.ffill()

        # Reset index to make timestamp a column again
        df_filled.reset_index(inplace=True)
        df_filled.rename(columns={"index": "timestamp"}, inplace=True)

        return df_filled


class DataLoader:
    """Loads OHLCV data from CCXT exchanges"""

    def __init__(
        self,
        data_dir: str = "data",
        max_retries: int = 3,
        min_data_days: int = 30,
    ):
        """Initialize DataLoader.

        Args:
            data_dir: Base directory for data storage
            max_retries: Maximum retry attempts for failed fetches
            min_data_days: Minimum number of days of data required
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories if needed
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.min_data_days = min_data_days
        self.validator = DataValidator()
        self.backoff = ExponentialBackoff(
            initial_sec=1.0,
            multiplier=2.0,
            max_delay=30.0,
            max_retries=max_retries,
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        limit: int = 1000,
        timeframe: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol with retry logic and caching.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            limit: Number of candles to fetch
            timeframe: Candle timeframe (default "1d")
            use_cache: Use cached data if available (default True)

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If symbol is invalid or data is insufficient
            TypeError: If symbol is None
        """
        # Validate inputs
        if symbol is None:
            raise TypeError("Symbol cannot be None")

        if not symbol or symbol == "":
            raise ValueError("Symbol cannot be empty")

        # Check minimum data requirement
        if limit < self.min_data_days:
            raise ValueError(
                f"Insufficient data requested: {limit} days < {self.min_data_days} days minimum"
            )

        # Check cache first
        if use_cache:
            cached_df = self.load_from_cache(symbol)
            if cached_df is not None and len(cached_df) >= limit:
                logger.info(f"Using cached data for {symbol} ({len(cached_df)} records)")
                return cached_df.tail(limit)

        # Attempt fetch with retries
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                ohlcv_data = self._fetch_from_exchange(symbol, limit, timeframe)

                # Convert to DataFrame
                df = self._convert_to_dataframe(ohlcv_data)

                # Validate data
                self.validator.validate(df)

                # Interpolate gaps
                df = self.validator.interpolate(df)

                # Save to cache
                if use_cache:
                    self.save_to_cache(symbol, df)
                    logger.info(f"Cached {symbol} data ({len(df)} records)")

                return df

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.backoff.get_delay(attempt)
                    logger.warning(f"Attempt {attempt}/{self.max_retries} failed for {symbol}: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # All retries exhausted - try to use cache as fallback
                    logger.error(f"Failed to fetch {symbol} after {self.max_retries} retries: {str(last_error)}")
                    if use_cache:
                        cached_df = self.load_from_cache(symbol)
                        if cached_df is not None:
                            logger.warning(f"Using stale cached data for {symbol} as fallback")
                            return cached_df.tail(limit)
                    raise Exception(
                        f"Failed to fetch {symbol} after {self.max_retries} retries: {str(last_error)}"
                    ) from last_error

    def _fetch_from_exchange(
        self,
        symbol: str,
        limit: int,
        timeframe: str,
    ) -> List[List[Any]]:
        """Fetch OHLCV data from exchange using CCXT.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            limit: Number of candles to fetch
            timeframe: Candle timeframe (e.g., "1d", "1h")

        Returns:
            List of [timestamp, o, h, l, c, v] lists

        Raises:
            ValueError: If symbol is invalid
            Exception: For network/API errors
        """
        import ccxt
        import time

        # Initialize exchange (Binance for spot market)
        exchange = ccxt.binance({
            'enableRateLimit': True,  # Respect rate limits
            'timeout': 30000,         # 30 seconds timeout
            'options': {
                'defaultType': 'spot',  # Use spot market
            }
        })

        try:
            # Fetch OHLCV data
            # CCXT returns: [[timestamp_ms, open, high, low, close, volume], ...]
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            if not ohlcv or len(ohlcv) == 0:
                raise ValueError(f"No data returned for {symbol}")

            # Small delay for rate limiting (0.1s between calls)
            time.sleep(0.1)

            return ohlcv

        except ccxt.BadSymbol as e:
            raise ValueError(f"Invalid symbol: {symbol}") from e
        except ccxt.NetworkError as e:
            raise Exception(f"Network error fetching {symbol}: {str(e)}") from e
        except ccxt.ExchangeError as e:
            raise Exception(f"Exchange error fetching {symbol}: {str(e)}") from e

    def _convert_to_dataframe(self, ohlcv_data: List[List[Any]]) -> pd.DataFrame:
        """Convert OHLCV list to DataFrame.

        Args:
            ohlcv_data: List of [timestamp, o, h, l, c, v] lists

        Returns:
            DataFrame with OHLCV columns
        """
        df = pd.DataFrame(
            ohlcv_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Convert timestamp from milliseconds to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')

        return df

    def get_cached_path(self, symbol: str) -> Path:
        """Get path for cached symbol data.

        Args:
            symbol: Trading symbol

        Returns:
            Path to cached parquet file
        """
        # Convert symbol to filename (e.g., "BTC/USDT" -> "BTC_USDT.parquet")
        safe_symbol = symbol.replace("/", "_")
        return self.raw_dir / f"{safe_symbol}.parquet"

    def load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            DataFrame if cache exists, None otherwise
        """
        cache_path = self.get_cached_path(symbol)

        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                return None

        return None

    def save_to_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save data to cache.

        Args:
            symbol: Trading symbol
            df: DataFrame to cache
        """
        cache_path = self.get_cached_path(symbol)
        df.to_parquet(cache_path, index=False)
