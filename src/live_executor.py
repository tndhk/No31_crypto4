"""Live trading executor using CCXT"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

import ccxt
import pandas as pd

from src.executor import Executor, Order, Position, ExecutionMode

logger = logging.getLogger(__name__)


class LiveExecutor(Executor):
    """Executor for live trading on Binance"""

    def __init__(
        self,
        config: Dict[str, Any],
        dry_run: bool = False,
    ):
        """Initialize LiveExecutor.

        Args:
            config: Configuration dictionary
            dry_run: If True, do not send actual orders
        """
        super().__init__(config, mode=ExecutionMode.LIVE)
        self.dry_run = dry_run
        
        # Initialize exchange
        self.exchange = self._init_exchange()
        
        # Sync initial state
        self.sync_state()

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize CCXT exchange instance.

        Returns:
            CCXT exchange instance
        """
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET")

        if not self.dry_run and (not api_key or not secret):
            raise ValueError("BINANCE_API_KEY and BINANCE_SECRET must be set for live trading")

        exchange_config = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        }

        return ccxt.binance(exchange_config)

    def sync_state(self) -> None:
        """Synchronize local state with exchange."""
        try:
            if self.dry_run:
                logger.info("DRY RUN: Skipping state sync")
                return

            # Fetch balance
            balance = self.exchange.fetch_balance()
            self.current_balance = float(balance['USDT']['free'])
            
            # Fetch positions (non-zero balances)
            # Note: For spot, "positions" are just asset balances
            self.positions = {}
            for currency, amount in balance['total'].items():
                if currency != 'USDT' and amount > 0:
                    # Estimate value (simplified)
                    # In a real system, we'd fetch current price to calculate value
                    symbol = f"{currency}/USDT"
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        price = ticker['last']
                        value = amount * price
                        
                        # Only track significant positions (> $10)
                        if value > 10:
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                entry_price=price, # Approximate
                                size=amount,
                                position_type="LONG"
                            )
                    except Exception:
                        # Ignore assets that don't have USDT pair or fail to fetch
                        pass

            logger.info(f"State synced. Balance: ${self.current_balance:.2f}, Positions: {len(self.positions)}")

        except Exception as e:
            logger.error(f"Failed to sync state: {e}")
            if not self.dry_run:
                raise

    def execute_order(self, order: Order) -> bool:
        """Execute order on exchange.

        Args:
            order: Order to execute

        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info(f"DRY RUN: Would execute {order}")
            return True

        try:
            # Prepare order parameters
            symbol = order.symbol
            side = order.order_type.lower() # 'buy' or 'sell'
            amount = order.size
            
            # Execute market order (simplest for now)
            # For production, limit orders with smart pricing are better
            logger.info(f"Sending {side.upper()} order for {amount} {symbol}...")
            
            response = self.exchange.create_market_order(symbol, side, amount)
            
            logger.info(f"Order executed successfully: {response['id']}")
            
            # Update local state immediately
            self.sync_state()
            
            return True

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return False
