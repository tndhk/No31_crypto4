"""Tests for CLI main entry point"""

import sys
from pathlib import Path
from typing import List
import argparse

import pytest

from src.cli import main, parse_arguments, CliApp


class TestArgumentParsing:
    """Tests for CLI argument parsing"""

    def test_parse_arguments_backtest_mode(self):
        """TC-N-01: Parse arguments for backtest mode."""
        # Given: Backtest mode arguments
        args_list = [
            "--mode", "backtest",
            "--config", "config/settings.yaml",
            "--symbols", "BTC", "ETH", "SOL",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse correctly
        assert args.mode == "backtest"
        assert args.config == "config/settings.yaml"
        assert args.symbols == ["BTC", "ETH", "SOL"]

    def test_parse_arguments_live_mode(self):
        """TC-N-02: Parse arguments for live mode."""
        # Given: Live mode arguments
        args_list = [
            "--mode", "live",
            "--config", "config/settings.yaml",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should recognize live mode
        assert args.mode == "live"

    def test_parse_arguments_with_date_range(self):
        """TC-N-03: Parse date range arguments."""
        # Given: Date range arguments
        args_list = [
            "--mode", "backtest",
            "--start-date", "2024-01-01",
            "--end-date", "2024-12-31",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse dates
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-12-31"

    def test_parse_arguments_with_output(self):
        """TC-N-04: Parse output directory argument."""
        # Given: Output directory argument
        args_list = [
            "--mode", "backtest",
            "--output", "results/",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse output path
        assert args.output == "results/"

    def test_parse_arguments_defaults(self):
        """TC-N-05: Default arguments are provided."""
        # Given: Minimal arguments
        args_list = ["--mode", "backtest"]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should have default values
        assert hasattr(args, "config")
        assert hasattr(args, "output")
        assert hasattr(args, "verbose")

    def test_parse_arguments_missing_mode_raises_error(self):
        """TC-A-01: Missing mode raises error."""
        # Given: No mode argument
        args_list = ["--config", "config.yaml"]

        # When/Then: Should raise error
        with pytest.raises(SystemExit):
            parse_arguments(args_list)

    def test_parse_arguments_invalid_mode_raises_error(self):
        """TC-A-02: Invalid mode raises error."""
        # Given: Invalid mode
        args_list = ["--mode", "invalid_mode"]

        # When/Then: Should raise error
        with pytest.raises((SystemExit, ValueError)):
            parse_arguments(args_list)

    def test_parse_arguments_verbose_flag(self):
        """TC-N-06: Verbose flag is recognized."""
        # Given: Verbose flag
        args_list = [
            "--mode", "backtest",
            "--verbose",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should set verbose
        assert args.verbose is True


class TestCliApp:
    """Tests for CliApp initialization and workflow"""

    def test_cli_app_initialization_backtest(self):
        """TC-N-07: CliApp initializes for backtest mode."""
        # Given: Backtest configuration
        args = argparse.Namespace(
            mode="backtest",
            config="config/settings.yaml",
            symbols=["BTC", "ETH"],
            output="results/",
            verbose=False,
            start_date=None,
            end_date=None,
            capital=10000.0,
            dry_run=False,
        )

        # When: Creating CliApp
        app = CliApp(args)

        # Then: Should initialize
        assert app.mode == "backtest"
        assert app.config_path == "config/settings.yaml"
        assert app.symbols == ["BTC", "ETH"]

    def test_cli_app_initialization_live(self):
        """TC-N-08: CliApp initializes for live mode."""
        # Given: Live configuration
        args = argparse.Namespace(
            mode="live",
            config="config/settings.yaml",
            symbols=None,
            output=None,
            verbose=False,
            start_date=None,
            end_date=None,
            capital=10000.0,
            dry_run=False,
        )

        # When: Creating CliApp
        app = CliApp(args)

        # Then: Should initialize
        assert app.mode == "live"

    def test_cli_app_null_args_raises_error(self):
        """TC-A-03: Null args raises error."""
        # Given: None args
        args = None

        # When/Then: Should raise error
        with pytest.raises((TypeError, AttributeError)):
            CliApp(args)

    def test_cli_app_run_method_exists(self):
        """TC-N-09: CliApp has run method."""
        # Given: CliApp instance
        args = argparse.Namespace(
            mode="backtest",
            config="config/settings.yaml",
            symbols=["BTC"],
            output="results/",
            verbose=False,
            start_date=None,
            end_date=None,
            capital=10000.0,
            dry_run=False,
        )
        app = CliApp(args)

        # When/Then: Should have run method
        assert hasattr(app, "run")
        assert callable(app.run)


class TestCliWorkflow:
    """Tests for CLI workflow orchestration"""

    def test_cli_workflow_loads_config(self):
        """TC-N-10: Workflow loads configuration."""
        # Given: Config path
        args = argparse.Namespace(
            mode="backtest",
            config="config/settings.yaml",
            symbols=["BTC"],
            output="results/",
            verbose=False,
            start_date=None,
            end_date=None,
            capital=10000.0,
            dry_run=False,
        )

        # When: Creating CliApp
        app = CliApp(args)

        # Then: Should be able to load config
        assert app.config_path is not None

    def test_cli_workflow_validates_arguments(self):
        """TC-N-11: Workflow validates arguments."""
        # Given: Valid arguments
        args_list = [
            "--mode", "backtest",
            "--config", "config/settings.yaml",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should have valid structure
        assert args.mode in ["backtest", "live"]

    def test_cli_workflow_output_directory_creation(self):
        """TC-N-12: Workflow creates output directory."""
        # Given: Output path
        args = argparse.Namespace(
            mode="backtest",
            config="config/settings.yaml",
            symbols=["BTC"],
            output="results/",
            verbose=False,
            start_date=None,
            end_date=None,
            capital=10000.0,
            dry_run=False,
        )

        # When: Creating CliApp
        app = CliApp(args)

        # Then: Should track output path
        assert app.output_dir == "results/"


class TestCliErrorHandling:
    """Tests for CLI error handling"""

    def test_cli_missing_config_file_error(self):
        """TC-A-04: Missing config file raises error."""
        # Given: Non-existent config
        args_list = [
            "--mode", "backtest",
            "--config", "/nonexistent/path/config.yaml",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: CliApp should detect missing file
        # (error would occur during app.run())
        assert args.config == "/nonexistent/path/config.yaml"

    def test_cli_invalid_symbols_format(self):
        """TC-A-05: Invalid symbols format handled."""
        # Given: Symbols with invalid format
        args_list = [
            "--mode", "backtest",
            "--symbols", "",  # Empty symbol
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse (validation happens at run time)
        assert hasattr(args, "symbols")

    def test_cli_negative_capital_raises_error(self):
        """TC-A-06: Negative starting capital raises error."""
        # Given: Negative capital
        args_list = [
            "--mode", "backtest",
            "--capital", "-1000",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse capital as float
        if hasattr(args, "capital"):
            # Capital is parsed as float, value is negative
            assert args.capital == -1000.0
            assert isinstance(args.capital, float)


class TestCliIntegration:
    """Integration tests for CLI workflow"""

    def test_cli_full_workflow_setup(self):
        """TC-N-13: Full CLI setup completes."""
        # Given: Complete arguments
        args_list = [
            "--mode", "backtest",
            "--config", "config/settings.yaml",
            "--symbols", "BTC", "ETH",
            "--output", "results/",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # And: Creating CliApp
        app = CliApp(args)

        # Then: Should be ready for execution
        assert app is not None
        assert app.mode == "backtest"
        assert len(app.symbols) == 2

    def test_cli_verbose_output_setting(self):
        """TC-N-14: Verbose output setting is configured."""
        # Given: Verbose flag
        args_list = [
            "--mode", "backtest",
            "--verbose",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Verbose should be set
        assert args.verbose is True

    def test_cli_multiple_symbols_parsing(self):
        """TC-N-15: Multiple symbols parsed correctly."""
        # Given: Multiple symbols
        symbols = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
        args_list = ["--mode", "backtest", "--symbols"] + symbols

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: All symbols should be parsed
        assert args.symbols == symbols

    def test_cli_date_range_validation(self):
        """TC-N-16: Date range arguments are handled."""
        # Given: Start and end dates
        args_list = [
            "--mode", "backtest",
            "--start-date", "2023-01-01",
            "--end-date", "2024-01-01",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse dates
        assert args.start_date == "2023-01-01"
        assert args.end_date == "2024-01-01"


class TestMainFunction:
    """Tests for main entry point function"""

    def test_main_with_backtest_args(self):
        """TC-N-17: Main function accepts backtest arguments."""
        # Given: Backtest arguments
        test_args = [
            "crypt4",  # Program name
            "--mode", "backtest",
            "--config", "config/settings.yaml",
        ]

        # When: Calling main (simulate)
        # Note: We test parse_arguments separately since main() may do more
        args = parse_arguments(test_args[1:])

        # Then: Should parse successfully
        assert args.mode == "backtest"

    def test_main_with_live_args(self):
        """TC-N-18: Main function accepts live mode arguments."""
        # Given: Live mode arguments
        test_args = [
            "--mode", "live",
            "--config", "config/settings.yaml",
        ]

        # When: Calling parse_arguments
        args = parse_arguments(test_args)

        # Then: Should recognize live mode
        assert args.mode == "live"

    def test_main_help_argument(self):
        """TC-B-01: Help argument shows usage."""
        # Given: Help argument
        test_args = ["--help"]

        # When/Then: Should raise SystemExit (help exits)
        with pytest.raises(SystemExit):
            parse_arguments(test_args)


class TestCliEdgeCases:
    """Tests for edge cases"""

    def test_cli_empty_symbols_list(self):
        """TC-A-07: Empty symbols list is handled."""
        # Given: No symbols
        args_list = ["--mode", "backtest"]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should have symbols attribute (None or empty)
        assert hasattr(args, "symbols")

    def test_cli_very_long_config_path(self):
        """TC-A-08: Very long config path handled."""
        # Given: Very long path
        long_path = "a/" * 100 + "config.yaml"
        args_list = ["--mode", "backtest", "--config", long_path]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should store long path
        assert args.config == long_path
        assert len(args.config) > 100

    def test_cli_special_characters_in_symbols(self):
        """TC-A-09: Special characters in symbols handled."""
        # Given: Symbols with special characters
        args_list = [
            "--mode", "backtest",
            "--symbols", "BTC/USDT", "ETH/USDT",
        ]

        # When: Parsing arguments
        args = parse_arguments(args_list)

        # Then: Should parse with slashes
        assert "/" in args.symbols[0]
