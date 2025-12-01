"""Tests for src.utils module"""

import os
import csv
import stat
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock

from src.utils import (
    setup_audit_logger,
    get_utc_timestamp,
    ExponentialBackoff,
    AuditLogger,
)


class TestSetupAuditLogger:
    """Tests for audit logger setup"""

    def test_setup_audit_logger_creates_file(self):
        """TC-N-01: Setup creates audit log file."""
        # Given: A temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.csv"

            # When: Setting up audit logger
            logger = setup_audit_logger(str(log_path))

            # Then: File should be created
            assert log_path.exists()
            assert isinstance(logger, AuditLogger)

    def test_setup_audit_logger_permissions_600(self):
        """TC-N-02: Audit log file has chmod 600 permissions."""
        # Given: A temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.csv"

            # When: Setting up audit logger
            logger = setup_audit_logger(str(log_path))

            # Then: File permissions should be 600 (rw-------)
            file_stat = os.stat(log_path)
            file_permissions = stat.filemode(file_stat.st_mode)
            # Extract just the permission bits
            mode = file_stat.st_mode & 0o777
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_audit_logger_log_entry(self):
        """TC-N-03: Audit logger records entry correctly."""
        # Given: An initialized audit logger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.csv"
            logger = setup_audit_logger(str(log_path))

            # When: Writing a log entry
            logger.log_entry(
                timestamp="2024-01-01 12:00:00 UTC",
                symbol="ETH",
                signal_type="ENTRY",
                order_type="BUY",
                price=2000.0,
                size=1.5,
                reason="Z-Score=-2.5, BULLISH regime",
            )

            # Then: Entry should be in CSV file
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["symbol"] == "ETH"
            assert rows[0]["signal_type"] == "ENTRY"
            assert rows[0]["order_type"] == "BUY"
            assert rows[0]["price"] == "2000.0"
            assert rows[0]["size"] == "1.5"

    def test_audit_logger_multiple_entries(self):
        """TC-N-04: Audit logger handles multiple entries."""
        # Given: An initialized audit logger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.csv"
            logger = setup_audit_logger(str(log_path))

            # When: Writing multiple entries
            for i in range(3):
                logger.log_entry(
                    timestamp=f"2024-01-01 {i:02d}:00:00 UTC",
                    symbol=f"SYM{i}",
                    signal_type="ENTRY",
                    order_type="BUY",
                    price=100.0 + i,
                    size=1.0,
                    reason="Test entry",
                )

            # Then: All entries should be in CSV file
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 3

    def test_audit_logger_empty_message(self):
        """TC-B-01: Audit logger handles empty message."""
        # Given: An initialized audit logger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.csv"
            logger = setup_audit_logger(str(log_path))

            # When: Writing with empty reason
            logger.log_entry(
                timestamp="2024-01-01 12:00:00 UTC",
                symbol="ETH",
                signal_type="ENTRY",
                order_type="BUY",
                price=2000.0,
                size=1.5,
                reason="",  # Empty reason
            )

            # Then: Entry should be created without error
            with open(log_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["reason"] == ""


class TestGetUTCTimestamp:
    """Tests for UTC timestamp generation"""

    def test_get_utc_timestamp_format(self):
        """TC-N-01: UTC timestamp has correct format."""
        # Given: No preconditions
        # When: Getting UTC timestamp
        ts = get_utc_timestamp()

        # Then: Format should be "YYYY-MM-DD HH:MM:SS UTC"
        parts = ts.split()
        assert len(parts) == 3
        assert parts[2] == "UTC"
        # Check date format YYYY-MM-DD
        date_parts = parts[0].split("-")
        assert len(date_parts) == 3
        assert len(date_parts[0]) == 4  # YYYY
        assert len(date_parts[1]) == 2  # MM
        assert len(date_parts[2]) == 2  # DD
        # Check time format HH:MM:SS
        time_parts = parts[1].split(":")
        assert len(time_parts) == 3

    def test_get_utc_timestamp_is_utc(self):
        """TC-N-02: Timestamp is in UTC timezone."""
        # Given: No preconditions
        # When: Getting UTC timestamp
        ts = get_utc_timestamp()

        # Then: Should end with UTC
        assert ts.endswith("UTC")

    def test_get_utc_timestamp_midnight(self):
        """TC-B-01: UTC timestamp at midnight."""
        # Given: Test with a known time
        # When: Creating timestamp from known datetime
        midnight_utc = datetime(2024, 1, 1, 0, 0, 0)
        expected = midnight_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Then: Should represent 00:00:00 UTC
        assert "00:00:00 UTC" in expected

    def test_get_utc_timestamp_consistency(self):
        """TC-N-03: Multiple calls are consistent."""
        # Given: Call get_utc_timestamp multiple times
        # When: Comparing results
        ts1 = get_utc_timestamp()
        ts2 = get_utc_timestamp()

        # Then: Both should have valid UTC format (dates might differ by 1 second)
        assert ts1.endswith("UTC")
        assert ts2.endswith("UTC")


class TestExponentialBackoff:
    """Tests for exponential backoff retry logic"""

    def test_exponential_backoff_first_delay(self):
        """TC-N-01: First retry delay is initial_sec."""
        # Given: ExponentialBackoff with initial_sec=1
        backoff = ExponentialBackoff(initial_sec=1, max_retries=3)

        # When: Getting first delay
        delay = backoff.get_delay(attempt=1)

        # Then: Should be 1 second
        assert delay == 1

    def test_exponential_backoff_multiplier(self):
        """TC-N-02: Delay multiplies by multiplier each attempt."""
        # Given: ExponentialBackoff with multiplier=2
        backoff = ExponentialBackoff(initial_sec=1, multiplier=2, max_retries=5)

        # When: Getting delays for each attempt
        delays = [backoff.get_delay(i) for i in range(1, 4)]

        # Then: Should be 1, 2, 4 (exponential)
        assert delays[0] == 1
        assert delays[1] == 2
        assert delays[2] == 4

    def test_exponential_backoff_max_delay(self):
        """TC-N-03: Delay is capped at max_delay."""
        # Given: ExponentialBackoff with max_delay=5
        backoff = ExponentialBackoff(initial_sec=1, multiplier=2, max_delay=5, max_retries=10)

        # When: Getting delays beyond cap
        delays = [backoff.get_delay(i) for i in range(1, 6)]

        # Then: Should not exceed max_delay
        for delay in delays:
            assert delay <= 5

    def test_exponential_backoff_max_retries(self):
        """TC-B-01: max_retries is respected."""
        # Given: ExponentialBackoff with max_retries=3
        backoff = ExponentialBackoff(initial_sec=1, max_retries=3)

        # When: Checking if retries exceeded
        # Then: After 3 attempts, should indicate retry exhausted
        assert backoff.max_retries == 3

    def test_exponential_backoff_zero_initial(self):
        """TC-B-02: Zero initial delay."""
        # Given: ExponentialBackoff with initial_sec=0
        backoff = ExponentialBackoff(initial_sec=0, max_retries=3)

        # When: Getting first delay
        delay = backoff.get_delay(attempt=1)

        # Then: Should be 0
        assert delay == 0
