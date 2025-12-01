"""Utility functions for logging, error handling, and retry logic"""

import os
import csv
import stat
from datetime import datetime
from pathlib import Path
from typing import Optional


class AuditLogger:
    """CSV-based audit logger for trade records"""

    def __init__(self, log_path: str):
        """Initialize audit logger.

        Args:
            log_path: Path to CSV audit log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create file if it doesn't exist with header
        if not self.log_path.exists():
            self._write_header()
            # Set permissions to 600 (read/write for owner only)
            os.chmod(self.log_path, 0o600)

    def _write_header(self):
        """Write CSV header row"""
        headers = [
            "timestamp",
            "symbol",
            "signal_type",
            "order_type",
            "price",
            "size",
            "reason",
        ]
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def log_entry(
        self,
        timestamp: str,
        symbol: str,
        signal_type: str,
        order_type: str,
        price: float,
        size: float,
        reason: str,
    ) -> None:
        """Log a trade entry to CSV.

        Args:
            timestamp: ISO 8601 timestamp (e.g., "2024-01-01 12:00:00 UTC")
            symbol: Trading symbol (e.g., "ETH")
            signal_type: ENTRY or EXIT
            order_type: BUY or SELL
            price: Trade price
            size: Trade size
            reason: Trade reason (Z-Score value, regime info, etc.)
        """
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "symbol",
                    "signal_type",
                    "order_type",
                    "price",
                    "size",
                    "reason",
                ],
            )
            writer.writerow({
                "timestamp": timestamp,
                "symbol": symbol,
                "signal_type": signal_type,
                "order_type": order_type,
                "price": price,
                "size": size,
                "reason": reason,
            })


def setup_audit_logger(log_path: str) -> AuditLogger:
    """Create and initialize audit logger.

    Args:
        log_path: Path to CSV audit log file

    Returns:
        AuditLogger instance with chmod 600 permissions
    """
    logger = AuditLogger(log_path)
    # Ensure permissions are 600
    os.chmod(log_path, 0o600)
    return logger


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format.

    Returns:
        Timestamp string in format "YYYY-MM-DD HH:MM:SS UTC"
    """
    try:
        # Python 3.12+: Use timezone-aware UTC
        from datetime import UTC
        now = datetime.now(UTC)
    except ImportError:
        # Python <3.12: Use utcnow()
        now = datetime.utcnow()
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


class ExponentialBackoff:
    """Exponential backoff retry strategy"""

    def __init__(
        self,
        initial_sec: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        max_retries: int = 3,
    ):
        """Initialize exponential backoff strategy.

        Args:
            initial_sec: Initial delay in seconds
            multiplier: Multiplication factor for each retry
            max_delay: Maximum delay in seconds
            max_retries: Maximum number of retry attempts
        """
        self.initial_sec = initial_sec
        self.multiplier = multiplier
        self.max_delay = max_delay
        self.max_retries = max_retries

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Attempt number (1-indexed)

        Returns:
            Delay in seconds, capped at max_delay
        """
        # Calculate exponential delay: initial * (multiplier ^ (attempt - 1))
        delay = self.initial_sec * (self.multiplier ** (attempt - 1))
        # Cap at max_delay
        return min(delay, self.max_delay)
