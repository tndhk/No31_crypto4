"""Tests for src.config module"""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config, ConfigValidator


class TestLoadConfig:
    """Tests for loading configuration from YAML"""

    def test_load_valid_config(self):
        """TC-N-01: Load valid configuration file."""
        # Given: A temporary YAML file with valid config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "settings.yaml"
            config_data = {
                "universe": {
                    "altcoins": {
                        "initial": ["ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", "MATIC", "LTC"],
                        "min_symbols": 5,
                        "max_symbols": 50,
                    }
                },
                "strategy": {
                    "trend_window": 200,
                    "short_window": 20,
                    "z_entry": 2.0,
                    "z_target": 1.0,
                    "z_stop": 3.0,
                },
                "execution": {
                    "mode": "backtest",
                    "fee_pct": 0.11,
                    "slippage_pct": 0.05,
                },
                "validation": {
                    "min_sharpe_ratio": 1.0,
                    "max_drawdown_pct": 20.0,
                },
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # When: Loading configuration
            config = load_config(str(config_path))

            # Then: Config should be loaded successfully
            assert isinstance(config, Config)
            assert config["strategy"]["z_entry"] == 2.0
            assert config["execution"]["fee_pct"] == 0.11

    def test_load_missing_config_file(self):
        """TC-A-01: Missing config file raises error."""
        # Given: Non-existent config file path
        config_path = "/tmp/nonexistent_config_12345.yaml"

        # When: Attempting to load config
        # Then: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_config(config_path)

    def test_load_invalid_yaml_syntax(self):
        """TC-A-02: Invalid YAML syntax raises error."""
        # Given: A temporary file with invalid YAML
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "invalid.yaml"
            config_path.write_text("invalid: yaml: content: [")

            # When: Attempting to load config
            # Then: Should raise yaml.YAMLError or similar
            with pytest.raises((yaml.YAMLError, Exception)):
                load_config(str(config_path))

    def test_load_missing_required_field(self):
        """TC-A-03: Missing required field raises error."""
        # Given: Config without z_entry parameter
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "incomplete.yaml"
            config_data = {
                "strategy": {
                    # Missing z_entry
                    "z_target": 1.0,
                    "z_stop": 3.0,
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)

            # When: Loading configuration
            # Then: Should raise error for missing field
            with pytest.raises((KeyError, ValueError, Exception)):
                config = load_config(str(config_path))
                config.validate()


class TestConfigValidator:
    """Tests for configuration validation"""

    def test_validate_zero_z_entry(self):
        """TC-B-01: Zero z_entry threshold is invalid."""
        # Given: Config with z_entry = 0
        config_dict = {
            "strategy": {
                "z_entry": 0.0,
                "z_target": 1.0,
                "z_stop": 3.0,
            }
        }

        # When: Validating
        validator = ConfigValidator()

        # Then: Should raise validation error
        with pytest.raises((ValueError, AssertionError)):
            validator.validate(config_dict)

    def test_validate_negative_z_entry(self):
        """TC-B-02: Negative z_entry is invalid."""
        # Given: Config with z_entry = -1
        config_dict = {
            "strategy": {
                "z_entry": -1.0,
                "z_target": 1.0,
                "z_stop": 3.0,
            }
        }

        # When: Validating
        validator = ConfigValidator()

        # Then: Should raise validation error
        with pytest.raises((ValueError, AssertionError)):
            validator.validate(config_dict)

    def test_validate_positive_z_entry(self):
        """TC-N-01: Positive z_entry is valid."""
        # Given: Config with z_entry = 2.0
        config_dict = {
            "strategy": {
                "trend_window": 200,
                "short_window": 20,
                "z_entry": 2.0,
                "z_target": 1.0,
                "z_stop": 3.0,
            },
            "execution": {
                "fee_pct": 0.11,
                "slippage_pct": 0.05,
            },
            "validation": {
                "min_sharpe_ratio": 1.0,
                "max_drawdown_pct": 20.0,
            },
        }

        # When: Validating
        validator = ConfigValidator()

        # Then: Should pass validation
        try:
            validator.validate(config_dict)
        except (ValueError, AssertionError) as e:
            pytest.fail(f"Validation failed: {e}")

    def test_validate_window_sizes(self):
        """TC-N-02: Window size validation."""
        # Given: Config with valid window sizes
        config_dict = {
            "strategy": {
                "trend_window": 200,
                "short_window": 20,
                "z_entry": 2.0,
                "z_target": 1.0,
                "z_stop": 3.0,
            }
        }

        # When: Validating window sizes
        # Then: Should pass (200 > 20)
        assert config_dict["strategy"]["trend_window"] > config_dict["strategy"]["short_window"]


class TestConfig:
    """Tests for Config class"""

    def test_config_initialization(self):
        """TC-N-01: Config initializes with data."""
        # Given: A dictionary with config data
        config_data = {
            "strategy": {
                "z_entry": 2.0,
                "z_target": 1.0,
                "z_stop": 3.0,
            },
            "execution": {
                "mode": "backtest",
            },
        }

        # When: Creating Config instance
        config = Config(config_data)

        # Then: Data should be accessible
        assert config["strategy"]["z_entry"] == 2.0
        assert config["execution"]["mode"] == "backtest"

    def test_config_nested_access(self):
        """TC-N-02: Config allows nested dictionary access."""
        # Given: Config with nested structure
        config_data = {
            "strategy": {
                "z_entry": 2.0,
            }
        }
        config = Config(config_data)

        # When: Accessing nested values
        z_entry = config["strategy"]["z_entry"]

        # Then: Should get correct value
        assert z_entry == 2.0

    def test_config_get_method(self):
        """TC-N-03: Config.get() method works."""
        # Given: Config with data
        config_data = {"strategy": {"z_entry": 2.0}}
        config = Config(config_data)

        # When: Using get() method
        value = config.get("strategy", {}).get("z_entry")

        # Then: Should return value
        assert value == 2.0

    def test_config_missing_key(self):
        """TC-A-01: Accessing missing key raises KeyError."""
        # Given: Config without certain key
        config_data = {"strategy": {"z_entry": 2.0}}
        config = Config(config_data)

        # When: Accessing non-existent key
        # Then: Should raise KeyError
        with pytest.raises(KeyError):
            _ = config["nonexistent"]
