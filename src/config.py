"""Configuration management for trading system"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config(dict):
    """Configuration dictionary wrapper with validation support"""

    def __init__(self, data: Dict[str, Any]):
        """Initialize configuration from dictionary.

        Args:
            data: Configuration data dictionary
        """
        super().__init__(data)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        validator = ConfigValidator()
        validator.validate(dict(self))


class ConfigValidator:
    """Validates configuration parameters"""

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate strategy parameters
        if "strategy" in config:
            strategy = config["strategy"]
            self._validate_strategy(strategy)

        # Validate execution parameters
        if "execution" in config:
            execution = config["execution"]
            self._validate_execution(execution)

        # Validate universe parameters
        if "universe" in config:
            universe = config["universe"]
            self._validate_universe(universe)

        # Validate validation thresholds
        if "validation" in config:
            validation = config["validation"]
            self._validate_thresholds(validation)

    def _validate_strategy(self, strategy: Dict[str, Any]) -> None:
        """Validate strategy parameters.

        Args:
            strategy: Strategy configuration

        Raises:
            ValueError: If strategy parameters are invalid
        """
        # Check required fields
        required_fields = ["z_entry", "z_target", "z_stop"]
        for field in required_fields:
            if field not in strategy:
                raise KeyError(f"Missing required field: strategy.{field}")

        # Validate z_entry (must be positive)
        z_entry = strategy.get("z_entry")
        if z_entry is not None:
            if z_entry <= 0:
                raise ValueError(f"z_entry must be positive, got {z_entry}")

        # Validate z_target (must be positive)
        z_target = strategy.get("z_target")
        if z_target is not None:
            if z_target <= 0:
                raise ValueError(f"z_target must be positive, got {z_target}")

        # Validate z_stop (must be positive)
        z_stop = strategy.get("z_stop")
        if z_stop is not None:
            if z_stop <= 0:
                raise ValueError(f"z_stop must be positive, got {z_stop}")

        # Validate window sizes if present
        if "trend_window" in strategy and "short_window" in strategy:
            trend_window = strategy["trend_window"]
            short_window = strategy["short_window"]
            if trend_window <= short_window:
                raise ValueError(
                    f"trend_window ({trend_window}) must be > short_window ({short_window})"
                )

    def _validate_execution(self, execution: Dict[str, Any]) -> None:
        """Validate execution parameters.

        Args:
            execution: Execution configuration

        Raises:
            ValueError: If execution parameters are invalid
        """
        # Validate mode
        mode = execution.get("mode")
        if mode and mode not in ["backtest", "live"]:
            raise ValueError(f"Invalid execution mode: {mode}")

        # Validate fee percentage
        fee_pct = execution.get("fee_pct")
        if fee_pct is not None:
            if fee_pct < 0 or fee_pct > 100:
                raise ValueError(f"fee_pct must be between 0 and 100, got {fee_pct}")

        # Validate slippage percentage
        slippage_pct = execution.get("slippage_pct")
        if slippage_pct is not None:
            if slippage_pct < 0 or slippage_pct > 100:
                raise ValueError(f"slippage_pct must be between 0 and 100, got {slippage_pct}")

    def _validate_universe(self, universe: Dict[str, Any]) -> None:
        """Validate universe parameters.

        Args:
            universe: Universe configuration

        Raises:
            ValueError: If universe parameters are invalid
        """
        if "altcoins" in universe:
            altcoins = universe["altcoins"]

            # Validate min/max symbols
            min_symbols = altcoins.get("min_symbols", 1)
            max_symbols = altcoins.get("max_symbols", 50)
            if min_symbols > max_symbols:
                raise ValueError(
                    f"min_symbols ({min_symbols}) cannot be > max_symbols ({max_symbols})"
                )

            # Validate initial symbols list
            initial = altcoins.get("initial", [])
            if initial and (len(initial) < min_symbols or len(initial) > max_symbols):
                raise ValueError(
                    f"Initial symbols count ({len(initial)}) outside range "
                    f"[{min_symbols}, {max_symbols}]"
                )

    def _validate_thresholds(self, validation: Dict[str, Any]) -> None:
        """Validate validation thresholds.

        Args:
            validation: Validation configuration

        Raises:
            ValueError: If thresholds are invalid
        """
        # Validate min_sharpe_ratio
        min_sharpe = validation.get("min_sharpe_ratio")
        if min_sharpe is not None:
            if min_sharpe < 0:
                raise ValueError(f"min_sharpe_ratio must be non-negative, got {min_sharpe}")

        # Validate max_drawdown_pct
        max_dd = validation.get("max_drawdown_pct")
        if max_dd is not None:
            if max_dd <= 0 or max_dd > 100:
                raise ValueError(f"max_drawdown_pct must be between 0 and 100, got {max_dd}")


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config instance

    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        config_data = {}

    # Create Config instance
    config = Config(config_data)

    # Validate configuration
    config.validate()

    return config
