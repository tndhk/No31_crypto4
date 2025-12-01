"""CLI main entry point for Alt/BTC Ratio Trading System"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Any, Dict
import logging

from src.config import load_config
from src.data_loader import DataLoader
from src.strategy import Strategy
from src.portfolio import PortfolioOptimizer
from src.backtester import Backtester
from src.executor import Executor, ExecutionMode
from src.utils import get_utc_timestamp


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of arguments (for testing), defaults to sys.argv[1:]

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If arguments are invalid
    """
    parser = argparse.ArgumentParser(
        description="Alt/BTC Ratio Trend-Following Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest mode
  python -m src.cli --mode backtest --config config/settings.yaml --symbols BTC ETH SOL

  # Live mode
  python -m src.cli --mode live --config config/settings.yaml

  # With date range
  python -m src.cli --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
        """,
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        required=True,
        choices=["backtest", "live"],
        help="Execution mode (backtest or live trading)",
    )

    # Optional arguments
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)",
    )

    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Trading symbols (e.g., BTC ETH SOL). If not specified, uses config default.",
    )

    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date for backtest (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for backtest (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory for results (default: results/)",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Starting capital for backtest (default: 10000.0)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no actual execution)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching and fetch fresh data",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    return parsed_args


class CliApp:
    """CLI application for trading system"""

    def __init__(self, args: argparse.Namespace):
        """Initialize CliApp.

        Args:
            args: Parsed command-line arguments

        Raises:
            TypeError: If args is None
            ValueError: If arguments are invalid
        """
        if args is None:
            raise TypeError("Arguments cannot be None")

        self.args = args
        self.mode = args.mode
        self.config_path = args.config
        self.symbols = args.symbols
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.output_dir = args.output
        self.starting_capital = args.capital
        self.verbose = args.verbose
        self.dry_run = args.dry_run
        self.no_cache = getattr(args, 'no_cache', False)

        # Configure logging level
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug(f"Verbose logging enabled")

        logger.info(f"Initializing CliApp in {self.mode} mode")

    def run(self) -> int:
        """Run the trading system.

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        try:
            logger.info(f"Starting trading system at {get_utc_timestamp()}")

            # Load configuration
            logger.info(f"Loading configuration from {self.config_path}")
            config = load_config(self.config_path)
            logger.info("Configuration loaded successfully")

            # Load data
            logger.info("Loading market data")
            data_loader = DataLoader()

            # Fetch BTC data for analysis (primary hedge asset)
            logger.info("Fetching BTC/USDT data...")
            btc_data = data_loader.fetch_ohlcv(
                "BTC/USDT",
                limit=820,  # 730 days IS + 90 days OOS
                use_cache=not self.no_cache
            )
            logger.info(f"BTC/USDT data loaded: {len(btc_data)} records")

            # Fetch ETH data for analysis
            logger.info("Fetching ETH/USDT data...")
            eth_data = data_loader.fetch_ohlcv(
                "ETH/USDT",
                limit=820,
                use_cache=not self.no_cache
            )
            logger.info(f"ETH/USDT data loaded: {len(eth_data)} records")

            logger.info("Market data loaded successfully")

            # Phase 11-B: Calculate Alt/BTC ratio for strategy analysis
            logger.info("Computing ETH/BTC ratio...")
            eth_btc_ratio = eth_data['close'] / btc_data['close']

            # Create ratio DataFrame for WFO analysis
            # This shifts analysis from absolute price to ratio, enabling better signal generation
            ratio_df = btc_data.copy()
            ratio_df['close'] = eth_btc_ratio
            ratio_df['open'] = eth_data['open'] / btc_data['open']
            ratio_df['high'] = eth_data['high'] / btc_data['high']
            ratio_df['low'] = eth_data['low'] / btc_data['low']
            logger.info(f"ETH/BTC ratio computed: {len(ratio_df)} records (range: {ratio_df['close'].min():.6f} to {ratio_df['close'].max():.6f})")

            # Create strategy
            logger.info("Initializing strategy")
            strategy = Strategy(config)
            logger.info(f"Strategy initialized with config: {config.get('strategy', {})}")

            # Create portfolio optimizer
            logger.info("Initializing portfolio optimizer")
            optimizer = PortfolioOptimizer()
            logger.info("Portfolio optimizer initialized")

            # Create backtester (for backtest mode)
            if self.mode == "backtest":
                logger.info("Backtest mode: Setting up backtester")
                backtester = Backtester(config, starting_capital=self.starting_capital)
                logger.info(
                    f"Backtester initialized with capital: ${self.starting_capital:,.2f}"
                )

                if self.dry_run:
                    logger.info("DRY RUN: Not executing backtest")
                    return 0

                logger.info("Running Walk-Forward Optimization with real data")
                try:
                    # Run WFO with ETH/BTC ratio for Alt strategy analysis
                    # WFO splits data into 730-day In-Sample and 90-day Out-of-Sample windows
                    wfo_results = backtester.walk_forward_optimize(
                        ratio_df,
                        is_window_days=730,
                        oos_window_days=90
                    )

                    # Log key results
                    logger.info(f"WFO Results:")

                    # Extract In-Sample metrics (primary evaluation)
                    is_metrics = wfo_results.get('is_metrics')
                    if is_metrics:
                        sharpe = is_metrics.sharpe_ratio
                        max_dd = is_metrics.max_drawdown_pct
                        total_ret = is_metrics.total_return_pct
                    else:
                        sharpe = 0.0
                        max_dd = 100.0
                        total_ret = 0.0

                    logger.info(f"  Sharpe Ratio: {sharpe:.4f}")
                    logger.info(f"  Max Drawdown: {max_dd:.4f}%")
                    logger.info(f"  Total Return: {total_ret:.4f}%")

                    # Check Go/No-Go criteria (In-Sample must pass)
                    is_go = wfo_results.get('is_go', False)
                    if is_go:
                        logger.info("✓ Go/No-Go Criteria: PASS (Sharpe > 1.0 AND Drawdown < 20%)")
                    else:
                        logger.warning("✗ Go/No-Go Criteria: FAIL")

                    logger.info("WFO completed successfully")
                except Exception as e:
                    logger.error(f"WFO execution failed: {str(e)}", exc_info=True)
                    return 1

            # Create executor (for live mode)
            elif self.mode == "live":
                logger.info("Live mode: Setting up executor")
                executor = Executor(
                    config,
                    starting_capital=self.starting_capital,
                    mode=ExecutionMode.LIVE,
                )
                logger.info("Executor initialized for live trading")

                if self.dry_run:
                    logger.info("DRY RUN: Not executing live trades")
                    return 0

                logger.info("Live trading enabled")

            # Create output directory
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created: {self.output_dir}")

            logger.info(f"Trading system completed at {get_utc_timestamp()}")
            return 0

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return 1
        except ValueError as e:
            logger.error(f"Invalid value: {e}")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI.

    Args:
        argv: Command-line arguments (for testing)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse arguments
        args = parse_arguments(argv)

        # Create and run app
        app = CliApp(args)
        return app.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130  # Standard exit code for SIGINT
    except SystemExit as e:
        # This is raised by argparse on error or help
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
