"""Trading scheduler for live execution"""

import logging
import time
import schedule
from datetime import datetime
from typing import Dict, Any, Optional

from src.data_loader import DataLoader
from src.strategy import Strategy
from src.portfolio import PortfolioOptimizer
from src.live_executor import LiveExecutor
from src.executor import Order

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Schedules and executes trading logic"""

    def __init__(
        self,
        config: Dict[str, Any],
        executor: LiveExecutor,
    ):
        """Initialize TradingScheduler.

        Args:
            config: Configuration dictionary
            executor: LiveExecutor instance
        """
        self.config = config
        self.executor = executor
        self.data_loader = DataLoader()
        self.strategy = Strategy(config)
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Schedule jobs
        self._setup_schedule()

    def _setup_schedule(self) -> None:
        """Setup scheduled jobs."""
        # Run trading logic every hour
        # In a real production system, this might be more frequent or event-driven
        schedule.every(1).hours.do(self.run_trading_cycle)
        
        logger.info("Scheduler setup complete. Trading cycle scheduled every 1 hour.")

    def run_trading_cycle(self) -> None:
        """Execute one trading cycle."""
        logger.info(f"Starting trading cycle at {datetime.utcnow()}")
        
        try:
            # 1. Sync state
            self.executor.sync_state()
            
            # 2. Fetch latest data
            # We need enough data for strategy indicators
            lookback = max(
                self.config["strategy"]["trend_window"],
                self.config["strategy"]["short_window"]
            ) + 100
            
            btc_data = self.data_loader.fetch_ohlcv("BTC/USDT", limit=lookback, use_cache=False)
            eth_data = self.data_loader.fetch_ohlcv("ETH/USDT", limit=lookback, use_cache=False)
            
            # 3. Calculate Ratio
            # Align dataframes
            common_idx = btc_data.index.intersection(eth_data.index)
            btc_data = btc_data.loc[common_idx]
            eth_data = eth_data.loc[common_idx]
            
            # Create ratio dataframe
            ratio_df = btc_data.copy()
            ratio_df['close'] = eth_data['close'] / btc_data['close']
            ratio_df['open'] = eth_data['open'] / btc_data['open']
            ratio_df['high'] = eth_data['high'] / btc_data['high']
            ratio_df['low'] = eth_data['low'] / btc_data['low']
            
            # 4. Generate Signal
            regime = self.strategy.detect_regime(ratio_df)
            signal = self.strategy.generate_signal(ratio_df, regime)
            
            if signal:
                logger.info(f"Signal detected: {signal}")
                
                # 5. Execute Signal
                # Determine size (simplified: use 10% of balance for now, or use portfolio optimizer)
                # For this MVP, we'll use a fixed allocation logic based on config
                
                # Example logic:
                # If BUY signal and no position -> Buy
                # If SELL signal and have position -> Sell
                
                symbol = "ETH/USDT" # We are trading ETH based on ratio
                current_price = eth_data['close'].iloc[-1]
                
                if signal.signal_type == "ENTRY":
                    # Check if we already have a position
                    positions = self.executor.get_all_positions()
                    if symbol not in positions:
                        # Calculate size
                        balance = self.executor.current_balance
                        allocation_pct = 0.5 # Allocate 50% of capital
                        size = (balance * allocation_pct) / current_price
                        
                        order = Order(
                            symbol=symbol,
                            order_type="BUY",
                            size=size,
                            price=current_price,
                            reason=f"Signal: {signal.signal_type}"
                        )
                        self.executor.execute_order(order)
                        
                elif signal.signal_type == "EXIT":
                    positions = self.executor.get_all_positions()
                    if symbol in positions:
                        position = positions[symbol]
                        order = Order(
                            symbol=symbol,
                            order_type="SELL",
                            size=position.size,
                            price=current_price,
                            reason=f"Signal: {signal.signal_type}"
                        )
                        self.executor.execute_order(order)
            else:
                logger.info("No signal generated.")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

    def start(self) -> None:
        """Start the scheduler loop."""
        logger.info("Starting scheduler loop...")
        
        # Run once immediately on start
        self.run_trading_cycle()
        
        while True:
            schedule.run_pending()
            time.sleep(1)
