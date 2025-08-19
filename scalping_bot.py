#!/usr/bin/env python3
"""
Binance Futures Scalping Bot with Enhanced Risk Management
Optimized for quick in-and-out trades with tight spreads
"""

import os
import json
import sys
import time
import threading
import csv
import logging
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
from decimal import Decimal, ROUND_HALF_UP
import pytz
from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
import dotenv

dotenv.load_dotenv()


@dataclass
class ScalpingConfig:
    """Configuration optimized for scalping strategy."""
    symbol: str = 'ETHUSDC'
    base_quantity: float = 0.01
    max_position_size: float = 0.5  # Smaller positions for scalping
    take_profit_pct: float = 0.05  # Tight take profit for scalping
    stop_loss_pct: float = 0.1  # Tight stop loss for scalping
    direction: str = 'BUY'
    max_orders: int = 5  # Fewer orders for scalping
    wait_time: int = 15  # Faster execution
    min_spread_pct: float = 0.02  # Tighter spread requirement
    max_daily_loss: float = 50.0  # Lower daily loss limit
    position_size_scaling: bool = True
    volatility_lookback: int = 10  # Shorter lookback for scalping
    risk_per_trade_pct: float = 0.5  # Lower risk per trade
    scalping_timeout: int = 300  # 5 minutes max hold time
    min_volume: float = 1000.0  # Minimum volume requirement

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'SELL' if self.direction == "BUY" else 'BUY'


@dataclass
class RiskManager:
    """Risk management optimized for scalping."""
    daily_pnl: float = 0.0
    total_position_size: float = 0.0
    max_position_size: float = 0.5
    max_daily_loss: float = 50.0
    trades_today: int = 0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    recent_prices: deque = field(default_factory=lambda: deque(maxlen=50))
    emergency_stop: bool = False
    consecutive_losses: int = 0
    max_consecutive_losses: int = 3  # Lower for scalping
    scalping_losses: int = 0
    max_scalping_losses: int = 5

    def reset_daily_stats(self):
        """Reset daily statistics."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_reset_date = today
            self.consecutive_losses = 0
            self.scalping_losses = 0

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk parameters."""
        self.reset_daily_stats()

        if self.emergency_stop:
            return False, "Emergency stop activated"

        if self.daily_pnl <= -abs(self.max_daily_loss):
            return False, f"Daily loss limit reached: {self.daily_pnl}"

        if self.total_position_size >= self.max_position_size:
            return False, f"Maximum position size reached: {self.total_position_size}"

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"

        if self.scalping_losses >= self.max_scalping_losses:
            return False, f"Too many scalping losses: {self.scalping_losses}"

        return True, "OK"

    def calculate_volatility(self) -> float:
        """Calculate recent price volatility for scalping."""
        if len(self.recent_prices) < 5:
            return 0.005  # Lower default for scalping

        prices = list(self.recent_prices)
        returns = [(prices[i] / prices[i - 1] - 1) for i in range(1, len(prices))]

        if not returns:
            return 0.005

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5


@dataclass
class OrderMonitor:
    """Order monitoring optimized for scalping."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    timestamp: Optional[datetime] = None
    entry_time: Optional[datetime] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self):
        """Reset the monitor state."""
        with self._lock:
            self.order_id = None
            self.filled = False
            self.filled_price = None
            self.filled_qty = 0.0
            self.timestamp = None
            self.entry_time = None

    def update_fill(self, price: float, qty: float):
        """Update fill information."""
        with self._lock:
            self.filled = True
            self.filled_price = price
            self.filled_qty = qty
            self.timestamp = datetime.now()
            if not self.entry_time:
                self.entry_time = datetime.now()


@dataclass
class MarketData:
    """Market data optimized for scalping."""
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_price: float = 0.0
    volume_24h: float = 0.0
    timestamp: Optional[datetime] = None

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        if self.spread > 0 and self.last_price > 0:
            return (self.spread / self.last_price) * 100
        return 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.last_price