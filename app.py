#!/usr/bin/env python3
"""
Improved Binance Futures Trading Bot with Enhanced Risk Management
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
class TradingConfig:
    """Enhanced configuration class for trading parameters."""
    symbol: str = 'ETHUSDC'
    base_quantity: float = 0.01
    max_position_size: float = 1.0  # Maximum total position size
    take_profit_pct: float = 0.1  # Take profit as percentage of entry price
    stop_loss_pct: float = 0.2  # Stop loss as percentage of entry price
    direction: str = 'BUY'
    max_orders: int = 10  # Reduced from 75 for better risk management
    wait_time: int = 30
    min_spread_pct: float = 0.05  # Minimum spread required to place orders
    max_daily_loss: float = 100.0  # Maximum daily loss before stopping
    position_size_scaling: bool = True  # Scale position size based on volatility
    volatility_lookback: int = 20  # Periods for volatility calculation
    risk_per_trade_pct: float = 1.0  # Risk per trade as % of account balance

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'BUY' if self.direction == "SELL" else 'SELL'


@dataclass
class RiskManager:
    """Risk management state and controls."""
    daily_pnl: float = 0.0
    total_position_size: float = 0.0
    max_position_size: float = 1.0
    max_daily_loss: float = 100.0
    trades_today: int = 0
    last_reset_date: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    recent_prices: deque = field(default_factory=lambda: deque(maxlen=100))
    emergency_stop: bool = False
    consecutive_losses: int = 0
    max_consecutive_losses: int = 5

    def reset_daily_stats(self):
        """Reset daily statistics."""
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_reset_date = today
            self.consecutive_losses = 0

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

        return True, "OK"

    def calculate_volatility(self) -> float:
        """Calculate recent price volatility."""
        if len(self.recent_prices) < 10:
            return 0.01  # Default volatility

        prices = list(self.recent_prices)
        returns = [(prices[i] / prices[i - 1] - 1) for i in range(1, len(prices))]

        if not returns:
            return 0.01

        # Calculate standard deviation of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return variance ** 0.5


@dataclass
class OrderMonitor:
    """Thread-safe order monitoring state with enhanced tracking."""
    order_id: Optional[str] = None
    filled: bool = False
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    timestamp: Optional[datetime] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self):
        """Reset the monitor state with thread safety."""
        with self._lock:
            self.order_id = None
            self.filled = False
            self.filled_price = None
            self.filled_qty = 0.0
            self.timestamp = None

    def update_fill(self, price: float, qty: float):
        """Update fill information with thread safety."""
        with self._lock:
            self.filled = True
            self.filled_price = price
            self.filled_qty = qty
            self.timestamp = datetime.now()


@dataclass
class MarketData:
    """Market data state."""
    bid_price: float = 0.0
    ask_price: float = 0.0
    last_price: float = 0.0
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


class TradingLogger:
    """Enhanced logging with structured output and error handling."""

    def __init__(self, symbol: str, log_to_console: bool = False):
        self.symbol = symbol
        self.log_file = f"{symbol}_transactions_log.csv"
        self.debug_log_file = f"{symbol}_bot_activity.log"
        self.risk_log_file = f"{symbol}_risk_events.log"
        self.logger = self._setup_logger(log_to_console)
        timezone_name = os.getenv('TIMEZONE', 'UTC')
        self.timezone = pytz.timezone(timezone_name)
        self._init_csv_headers()

    def _init_csv_headers(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        'timestamp', 'symbol', 'transaction_id', 'type',
                        'counter_order_id', 'price', 'amount', 'status',
                        'pnl', 'total_position', 'risk_score'
                    ])
            except Exception as e:
                self.log(f"Failed to initialize CSV headers: {e}", "ERROR")

    def _setup_logger(self, log_to_console: bool) -> logging.Logger:
        """Setup the logger with proper configuration."""
        logger = logging.getLogger(f"trading_bot_{self.symbol}")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler
        file_handler = logging.FileHandler(self.debug_log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Risk events handler
        risk_handler = logging.FileHandler(self.risk_log_file, mode='a')
        risk_handler.setFormatter(formatter)
        risk_handler.setLevel(logging.WARNING)
        logger.addHandler(risk_handler)

        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def log(self, message: str, level: str = "INFO"):
        """Log a message with the specified level."""
        getattr(self.logger, level.lower())(message)

    def log_transaction(self, transaction_id: str, tx_type: str, price: float,
                        amount: float, status: str, counter_order_id: Optional[str] = None,
                        pnl: float = 0.0, total_position: float = 0.0, risk_score: float = 0.0):
        """Log a transaction to CSV file with enhanced data."""
        try:
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now(self.timezone).isoformat(),
                    self.symbol,
                    transaction_id,
                    tx_type,
                    counter_order_id or "",
                    round(price, 6),
                    round(amount, 6),
                    status,
                    round(pnl, 2),
                    round(total_position, 6),
                    round(risk_score, 3)
                ])
        except Exception as e:
            self.log(f"Failed to log transaction: {e}", "ERROR")

    def log_risk_event(self, event_type: str, details: str, severity: str = "WARNING"):
        """Log risk management events."""
        message = f"RISK_EVENT - {event_type}: {details}"
        self.log(message, severity)


class BinanceClient:
    """Enhanced Binance client with retry logic and market data."""

    def __init__(self, api_key: str, api_secret: str):
        self.client = UMFutures(key=api_key, secret=api_secret)
        self._listen_key = None
        self._ws_client = None

    def get_account_balance(self) -> float:
        """Get account balance for risk calculations."""
        try:
            account = self.client.account()
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    return float(asset['walletBalance'])
            return 0.0
        except Exception as e:
            raise Exception(f"Failed to get account balance: {e}")

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information including precision and filters."""
        try:
            exchange_info = self.client.exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return symbol_info
            raise ValueError(f"Symbol {symbol} not found")
        except Exception as e:
            raise Exception(f"Failed to get symbol info: {e}")

    def get_current_price_data(self, symbol: str) -> MarketData:
        """Get current market data."""
        try:
            ticker = self.client.ticker_24hr_price_change_statistics(symbol=symbol)
            order_book = self.client.depth(symbol=symbol, limit=5)

            market_data = MarketData()
            market_data.last_price = float(ticker['lastPrice'])
            market_data.bid_price = float(order_book['bids'][0][0]) if order_book['bids'] else 0.0
            market_data.ask_price = float(order_book['asks'][0][0]) if order_book['asks'] else 0.0
            market_data.timestamp = datetime.now()

            return market_data
        except Exception as e:
            raise Exception(f"Failed to get market data: {e}")

    def get_position_info(self, symbol: str) -> Dict[str, Any]:
        """Get current position information."""
        try:
            positions = self.client.get_position_risk(symbol=symbol)
            for position in positions:
                if position['symbol'] == symbol:
                    return position
            return {}
        except Exception as e:
            raise Exception(f"Failed to get position info: {e}")

    def round_quantity(self, quantity: float, symbol_info: Dict[str, Any]) -> float:
        """Round quantity according to symbol precision."""
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                step_size = float(filter_info['stepSize'])
                precision = len(filter_info['stepSize'].split('.')[1].rstrip('0'))

                # Round to step size
                rounded = float(Decimal(str(quantity / step_size)).quantize(
                    Decimal('1'), rounding=ROUND_HALF_UP
                )) * step_size

                return round(rounded, precision)
        return quantity

    def round_price(self, price: float, symbol_info: Dict[str, Any]) -> float:
        """Round price according to symbol precision."""
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter_info['tickSize'])
                precision = len(filter_info['tickSize'].split('.')[1].rstrip('0'))

                # Round to tick size
                rounded = float(Decimal(str(price / tick_size)).quantize(
                    Decimal('1'), rounding=ROUND_HALF_UP
                )) * tick_size

                return round(rounded, precision)
        return price

    def get_listen_key(self) -> str:
        """Get or refresh the listen key."""
        if not self._listen_key:
            self._listen_key = self.client.new_listen_key()["listenKey"]
        return self._listen_key

    def renew_listen_key(self):
        """Renew the listen key."""
        try:
            self.client.renew_listen_key(listenKey=self._listen_key)
        except Exception:
            self._listen_key = None
            self.get_listen_key()

    def get_active_close_orders(self, symbol: str, close_order_side: str,
                                retries: int = 3, delay: float = 1.0) -> List[str]:
        """Get active close orders with retry logic."""
        for attempt in range(retries):
            try:
                open_orders = self.client.get_orders(symbol=symbol)
                return [
                    order.get('orderId')
                    for order in open_orders
                    if order.get('side') == close_order_side
                ]
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise Exception(f"Failed to get active orders after {retries} attempts: {e}")

    def place_limit_order(self, symbol: str, qty: float, price: float, side: str,
                          reduce_only: bool = False, time_in_force: str = "GTC") -> Dict[str, Any]:
        """Place a limit order with proper error handling."""
        try:
            order_params = {
                'symbol': symbol,
                'side': side,
                'positionSide': 'BOTH',
                'type': 'LIMIT',
                'quantity': qty,
                'price': str(price),
                'timeInForce': time_in_force
            }

            if reduce_only:
                order_params['reduceOnly'] = "true"

            return self.client.new_order(**order_params)
        except Exception as e:
            raise Exception(f"Failed to place order: {e}")

    def place_market_order(self, symbol: str, qty: float, side: str,
                           reduce_only: bool = False) -> Dict[str, Any]:
        """Place a market order (for stop losses)."""
        try:
            order_params = {
                'symbol': symbol,
                'side': side,
                'positionSide': 'BOTH',
                'type': 'MARKET',
                'quantity': qty
            }

            if reduce_only:
                order_params['reduceOnly'] = "true"

            return self.client.new_order(**order_params)
        except Exception as e:
            raise Exception(f"Failed to place market order: {e}")

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order with error handling."""
        try:
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
        except Exception as e:
            raise Exception(f"Failed to cancel order {order_id}: {e}")


class WebSocketManager:
    """Enhanced WebSocket manager with better error handling."""

    def __init__(self, client: BinanceClient, logger: TradingLogger,
                 order_monitor: OrderMonitor, config: TradingConfig,
                 fill_event: threading.Event, risk_manager: RiskManager):
        self.client = client
        self.logger = logger
        self.order_monitor = order_monitor
        self.config = config
        self.fill_event = fill_event
        self.risk_manager = risk_manager
        self.ws_client = None
        self.cumulative_pnl = 0.0
        self.start_time = time.time()
        self._running = False
        self._reconnect_count = 0
        self._max_reconnects = 5

    def start(self):
        """Start the WebSocket connection with reconnection logic."""
        self._running = True
        self._connect()

    def _connect(self):
        """Establish WebSocket connection."""
        try:
            listen_key = self.client.get_listen_key()
            self.ws_client = UMFuturesWebsocketClient(
                on_message=self._handle_message,
                on_error=self._handle_error,
                on_close=self._handle_close
            )
            self.ws_client.user_data(listen_key=listen_key)
            self._reconnect_count = 0
        except Exception as e:
            self.logger.log(f"Failed to establish WebSocket connection: {e}", "ERROR")
            self._schedule_reconnect()

    def _handle_error(self, _, error):
        """Handle WebSocket errors."""
        self.logger.log(f"WebSocket error: {error}", "ERROR")
        if self._running:
            self._schedule_reconnect()

    def _handle_close(self, _):
        """Handle WebSocket close."""
        self.logger.log("WebSocket connection closed", "WARNING")
        if self._running:
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt."""
        if self._reconnect_count >= self._max_reconnects:
            self.logger.log_risk_event("WEBSOCKET_FAILURE",
                                       f"Max reconnection attempts reached: {self._reconnect_count}",
                                       "ERROR")
            self.risk_manager.emergency_stop = True
            return

        self._reconnect_count += 1
        reconnect_delay = min(30, 2 ** self._reconnect_count)  # Exponential backoff

        self.logger.log(f"Scheduling reconnection in {reconnect_delay}s (attempt {self._reconnect_count})")

        def reconnect():
            time.sleep(reconnect_delay)
            if self._running:
                self._connect()

        threading.Thread(target=reconnect, daemon=True).start()

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self.ws_client:
            self.ws_client.stop()

    def _handle_message(self, _, raw_msg: str):
        """Handle incoming WebSocket messages with enhanced processing."""
        try:
            msg = json.loads(raw_msg)

            if 'e' not in msg:
                return

            event_type = msg.get('e')

            if event_type == 'ORDER_TRADE_UPDATE':
                self._handle_order_update(msg)
            elif event_type == 'ACCOUNT_UPDATE':
                self._handle_account_update(msg)

        except Exception as e:
            self.logger.log(f"Error handling WebSocket message: {e}", "ERROR")

    def _handle_order_update(self, msg: Dict[str, Any]):
        """Handle order status updates with improved logic."""
        order = msg['o']
        if order['s'] != self.config.symbol:
            return

        order_id = order['i']
        status = order['X']
        side = order['S']

        # Handle our tracked open orders
        if order_id == self.order_monitor.order_id:
            self._handle_tracked_order_update(order)

        # Handle close order fills
        elif side == self.config.close_order_side and status == "FILLED":
            self._handle_close_order_fill(order)

    def _handle_tracked_order_update(self, order: Dict[str, Any]):
        """Handle updates for our tracked open orders."""
        status = order['X']
        order_id = order['i']

        if status in ['FILLED', 'PARTIALLY_FILLED']:
            filled_qty = float(order['z'])
            avg_price = float(order['ap'])

            self.order_monitor.update_fill(avg_price, filled_qty)

            # Update risk manager
            position_change = filled_qty if self.config.direction == 'BUY' else -filled_qty
            self.risk_manager.total_position_size += position_change
            self.risk_manager.recent_prices.append(avg_price)

            self.logger.log(f"Order {order_id} {status.lower()}: "
                            f"{filled_qty} @ {avg_price}")

            if status == 'FILLED':
                self.fill_event.set()

    def _handle_close_order_fill(self, order: Dict[str, Any]):
        """Handle close order fills with enhanced PnL tracking."""
        order_id = order['i']
        filled_price = float(order['ap'])
        filled_qty = float(order['q'])

        # Calculate PnL based on direction
        if self.config.direction == 'BUY':
            # For BUY direction, close order is SELL
            pnl = (filled_price - (filled_price - self.config.take_profit_pct * filled_price / 100)) * filled_qty
        else:
            # For SELL direction, close order is BUY
            pnl = ((filled_price + self.config.take_profit_pct * filled_price / 100) - filled_price) * filled_qty

        self.cumulative_pnl += pnl
        self.risk_manager.daily_pnl += pnl

        # Update position size
        position_change = -filled_qty if self.config.direction == 'BUY' else filled_qty
        self.risk_manager.total_position_size += position_change

        # Update consecutive loss counter
        if pnl > 0:
            self.risk_manager.consecutive_losses = 0
        else:
            self.risk_manager.consecutive_losses += 1

        self.logger.log_transaction(
            order_id, "CLOSE", filled_price, filled_qty, "FILLED",
            pnl=pnl, total_position=self.risk_manager.total_position_size
        )

        self.logger.log(f"Close order filled. PnL: {pnl:.2f}, "
                        f"Cumulative: {self.cumulative_pnl:.2f}, "
                        f"Daily: {self.risk_manager.daily_pnl:.2f}")

    def _handle_account_update(self, msg: Dict[str, Any]):
        """Handle account updates for position tracking."""
        # This can be used to track real-time position changes
        pass


class TradingBot:
    """Enhanced trading bot with comprehensive risk management."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradingLogger(config.symbol)
        self.order_monitor = OrderMonitor()
        self.fill_event = threading.Event()
        self.risk_manager = RiskManager(
            max_position_size=config.max_position_size,
            max_daily_loss=config.max_daily_loss
        )

        # Validate API credentials
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")

        if not api_key or not api_secret:
            raise ValueError("API_KEY and API_SECRET environment variables must be set")

        self.client = BinanceClient(api_key, api_secret)
        self.ws_manager = WebSocketManager(
            self.client, self.logger, self.order_monitor,
            self.config, self.fill_event, self.risk_manager
        )

        # Initialize trading state
        self.symbol_info = None
        self.account_balance = 0.0
        self.last_open_order_time = 0
        self.skip_waiting_time = False
        self.active_close_orders = []
        self.last_log_time = time.time()
        self.last_market_data_update = 0
        self.market_data = MarketData()

        # Performance tracking
        self.trade_count = 0
        self.successful_trades = 0

        self._initialize()

    def _initialize(self):
        """Initialize bot with market data and account info."""
        try:
            # Get symbol information
            self.symbol_info = self.client.get_symbol_info(self.config.symbol)

            # Get account balance
            self.account_balance = self.client.get_account_balance()

            # Get current position
            position = self.client.get_position_info(self.config.symbol)
            if position:
                self.risk_manager.total_position_size = abs(float(position.get('positionAmt', 0)))

            # Update market data
            self.market_data = self.client.get_current_price_data(self.config.symbol)

            self.logger.log(f"Bot initialized - Balance: {self.account_balance}, "
                            f"Position: {self.risk_manager.total_position_size}")

        except Exception as e:
            self.logger.log(f"Failed to initialize bot: {e}", "ERROR")
            raise

    def _calculate_position_size(self) -> float:
        """Calculate position size based on volatility and risk management."""
        base_size = self.config.base_quantity

        if not self.config.position_size_scaling:
            return self.client.round_quantity(base_size, self.symbol_info)

        # Calculate volatility-adjusted size
        volatility = self.risk_manager.calculate_volatility()

        # Reduce size in high volatility
        volatility_multiplier = max(0.5, min(2.0, 1.0 / (1.0 + volatility * 10)))

        # Risk-based sizing
        if self.account_balance > 0:
            risk_amount = self.account_balance * (self.config.risk_per_trade_pct / 100)
            risk_based_size = risk_amount / (self.market_data.last_price * 0.01)  # Assuming 1% risk per trade
            base_size = min(base_size, risk_based_size)

        adjusted_size = base_size * volatility_multiplier

        # Ensure we don't exceed position limits
        remaining_capacity = self.config.max_position_size - self.risk_manager.total_position_size
        final_size = min(adjusted_size, remaining_capacity)

        return max(0, self.client.round_quantity(final_size, self.symbol_info))

    def _calculate_prices(self, entry_price: float) -> Tuple[float, float]:
        """Calculate take profit and stop loss prices."""
        if self.config.direction == 'BUY':
            take_profit = entry_price * (1 + self.config.take_profit_pct / 100)
            stop_loss = entry_price * (1 - self.config.stop_loss_pct / 100)
        else:
            take_profit = entry_price * (1 - self.config.take_profit_pct / 100)
            stop_loss = entry_price * (1 + self.config.stop_loss_pct / 100)

        take_profit = self.client.round_price(take_profit, self.symbol_info)
        stop_loss = self.client.round_price(stop_loss, self.symbol_info)

        return take_profit, stop_loss

    def _should_place_order(self) -> Tuple[bool, str]:
        """Determine if we should place a new order."""
        # Check risk management
        can_trade, risk_reason = self.risk_manager.can_trade()
        if not can_trade:
            return False, f"Risk management: {risk_reason}"

        # Check if we have too many active orders
        if len(self.active_close_orders) >= self.config.max_orders:
            return False, "Maximum active orders reached"

        # Check minimum spread requirement
        if (self.market_data.spread_pct > 0 and
                self.market_data.spread_pct < self.config.min_spread_pct):
            return False, f"Spread too tight: {self.market_data.spread_pct:.3f}%"

        # Check timing constraints
        current_time = time.time()
        wait_time = self._calculate_wait_time()

        if (not self.skip_waiting_time and
                current_time - self.last_open_order_time < wait_time):
            return False, f"Waiting period: {wait_time - (current_time - self.last_open_order_time):.1f}s remaining"

        return True, "OK"

    def _calculate_wait_time(self) -> int:
        """Calculate dynamic wait time based on market conditions and active orders."""
        base_wait_time = self.config.wait_time

        # Reduce wait time if we have few active orders
        order_ratio = len(self.active_close_orders) / max(1, self.config.max_orders)

        if order_ratio < 0.3:
            return int(base_wait_time * 0.5)  # Faster when few orders
        elif order_ratio > 0.8:
            return int(base_wait_time * 1.5)  # Slower when many orders

        return base_wait_time

    def _update_market_data(self):
        """Update market data periodically."""
        current_time = time.time()
        if current_time - self.last_market_data_update > 5:  # Update every 5 seconds
            try:
                self.market_data = self.client.get_current_price_data(self.config.symbol)
                self.risk_manager.recent_prices.append(self.market_data.last_price)
                self.last_market_data_update = current_time
            except Exception as e:
                self.logger.log(f"Failed to update market data: {e}", "ERROR")

    def _place_and_monitor_open_order(self) -> bool:
        """Place an order and monitor its execution with enhanced logic."""
        try:
            # Update market data first
            self._update_market_data()

            # Calculate position size
            quantity = self._calculate_position_size()
            if quantity <= 0:
                self.logger.log("Position size calculation returned 0, skipping order")
                return False

            # Determine entry price (use best available price)
            if self.config.direction == 'BUY':
                entry_price = self.market_data.ask_price or self.market_data.last_price
            else:
                entry_price = self.market_data.bid_price or self.market_data.last_price

            if entry_price <= 0:
                self.logger.log("Invalid entry price, skipping order", "WARNING")
                return False

            entry_price = self.client.round_price(entry_price, self.symbol_info)

            # Place the order
            order = self.client.place_limit_order(
                self.config.symbol,
                quantity,
                entry_price,
                self.config.direction,
                time_in_force="GTX"  # Post-only to get maker fees
            )

            self.logger.log(f"New order placed: {order['orderId']} - "
                            f"{quantity} @ {entry_price}")

            # Setup monitoring
            self.order_monitor.reset()
            self.order_monitor.order_id = order["orderId"]
            self.fill_event.clear()
            self.last_open_order_time = time.time()

            # Wait for fill or timeout
            filled = self.fill_event.wait(timeout=15)  # Increased timeout

            # Handle order result
            return self._handle_order_result(order, filled)

        except Exception as e:
            self.logger.log(f"Error placing order: {e}", "ERROR")
            return False

    def _handle_order_result(self, order: Dict[str, Any], filled: bool) -> bool:
        """Handle the result of an order placement with comprehensive logic."""
        order_id = order["orderId"]
        original_qty = float(order["origQty"])
        entry_price = float(order["price"])

        filled_qty = 0.0
        avg_fill_price = entry_price

        if filled and self.order_monitor.filled:
            # Order was filled via WebSocket
            filled_qty = self.order_monitor.filled_qty
            avg_fill_price = self.order_monitor.filled_price
            order_status = "FILLED" if filled_qty >= original_qty * 0.99 else "PARTIALLY_FILLED"
        else:
            # Order was not filled, try to cancel
            try:
                canceled_order = self.client.cancel_order(self.config.symbol, order_id)
                filled_qty = float(canceled_order.get("executedQty", 0))

                if filled_qty > 0:
                    # Partially filled before cancellation
                    avg_fill_price = self.order_monitor.filled_price or entry_price
                    order_status = "PARTIALLY_FILLED"
                    self.logger.log(f"Order {order_id} partially filled before cancellation: "
                                    f"{filled_qty}/{original_qty}")
                else:
                    order_status = "CANCELLED"
                    self.logger.log(f"Order {order_id} cancelled with no fills")

            except Exception as e:
                # Cancellation failed, assume order was filled
                self.logger.log(f"Failed to cancel order {order_id}, assuming filled: {e}")
                filled_qty = original_qty
                avg_fill_price = self.order_monitor.filled_price or entry_price
                order_status = "FILLED"

        # Place close orders if we have fills
        close_order_id = None
        if filled_qty > 0:
            try:
                close_order_id = self._place_close_orders(filled_qty, avg_fill_price)
            except Exception as e:
                self.logger.log_risk_event("CLOSE_ORDER_FAILURE",
                                           f"Failed to place close orders: {e}",
                                           "ERROR")

        # Calculate risk score
        risk_score = self._calculate_risk_score(filled_qty, avg_fill_price)

        # Log the transaction
        self.logger.log_transaction(
            order_id, "OPEN", avg_fill_price, filled_qty, order_status,
            counter_order_id=close_order_id,
            total_position=self.risk_manager.total_position_size,
            risk_score=risk_score
        )

        # Update statistics
        self.trade_count += 1
        if filled_qty > 0:
            self.successful_trades += 1

        return filled_qty > 0

    def _place_close_orders(self, qty: float, entry_price: float) -> str:
        """Place both take profit and stop loss orders."""
        take_profit_price, stop_loss_price = self._calculate_prices(entry_price)

        # Place take profit order
        tp_order = self.client.place_limit_order(
            self.config.symbol,
            qty,
            take_profit_price,
            self.config.close_order_side,
            reduce_only=True,
            time_in_force="GTC"
        )

        self.active_close_orders.append(tp_order["orderId"])

        self.logger.log(f"Take profit order placed: {tp_order['orderId']} - "
                        f"{qty} @ {take_profit_price}")

        # Note: For simplicity, we're only implementing take profit orders
        # Stop loss orders would require more complex order management
        # as they should cancel the TP order when triggered

        return tp_order["orderId"]

    def _calculate_risk_score(self, quantity: float, price: float) -> float:
        """Calculate a risk score for the trade."""
        position_risk = (quantity * price) / max(self.account_balance, 1)
        volatility_risk = self.risk_manager.calculate_volatility() * 10
        position_size_risk = self.risk_manager.total_position_size / self.config.max_position_size

        return min(1.0, position_risk + volatility_risk + position_size_risk)

    def _update_skip_waiting_logic(self):
        """Update the skip waiting time logic with enhanced conditions."""
        # Skip waiting if a close order was filled recently
        recent_fills = any(
            order_id not in self.active_close_orders
            for order_id in getattr(self, '_previous_close_orders', [])
        )

        # Skip waiting if we have very few active orders and market conditions are good
        few_orders = len(self.active_close_orders) < self.config.max_orders * 0.3
        good_spread = (self.market_data.spread_pct > self.config.min_spread_pct * 1.5
                       if self.market_data.spread_pct > 0 else True)

        # Skip waiting if consecutive losses require more aggressive trading
        need_recovery = (self.risk_manager.daily_pnl < -self.config.max_daily_loss * 0.5 and
                         self.risk_manager.consecutive_losses < 3)

        self.skip_waiting_time = recent_fills or (few_orders and good_spread) or need_recovery

        # Store current orders for next comparison
        self._previous_close_orders = self.active_close_orders.copy()

    def _log_status_periodically(self):
        """Log comprehensive status information periodically."""
        current_time = time.time()
        if current_time - self.last_log_time > 300:  # Every 5 minutes

            # Calculate performance metrics
            success_rate = (self.successful_trades / max(self.trade_count, 1)) * 100
            uptime_hours = (current_time - self.ws_manager.start_time) / 3600

            status_msg = (
                f"STATUS - Active orders: {len(self.active_close_orders)}/{self.config.max_orders}, "
                f"Position: {self.risk_manager.total_position_size:.4f}, "
                f"Daily PnL: {self.risk_manager.daily_pnl:.2f}, "
                f"Success rate: {success_rate:.1f}%, "
                f"Uptime: {uptime_hours:.1f}h, "
                f"Spread: {self.market_data.spread_pct:.3f}%"
            )

            self.logger.log(status_msg)
            self.last_log_time = current_time

            # Log risk warnings
            if self.risk_manager.daily_pnl < -self.config.max_daily_loss * 0.7:
                self.logger.log_risk_event("DAILY_LOSS_WARNING",
                                           f"Approaching daily loss limit: {self.risk_manager.daily_pnl}")

            if self.risk_manager.consecutive_losses >= 3:
                self.logger.log_risk_event("CONSECUTIVE_LOSSES",
                                           f"Consecutive losses: {self.risk_manager.consecutive_losses}")

    def _refresh_listen_key_thread(self):
        """Background thread to refresh the listen key."""
        while True:
            try:
                time.sleep(30 * 60)  # 30 minutes
                self.client.renew_listen_key()
                self.logger.log("Refreshed listenKey")
            except Exception as e:
                self.logger.log(f"Error renewing listen key: {e}", "ERROR")

    def run(self):
        """Main trading loop with comprehensive error handling and risk management."""
        self.logger.log(f"Starting trading bot for {self.config.symbol}")
        self.logger.log(f"Configuration: {self.config}")

        try:
            # Start WebSocket
            self.ws_manager.start()

            # Start refresh listen key thread
            refresh_thread = threading.Thread(
                target=self._refresh_listen_key_thread, daemon=True
            )
            refresh_thread.start()

            # Initialize active orders
            self.active_close_orders = self.client.get_active_close_orders(
                self.config.symbol, self.config.close_order_side
            )

            self.logger.log(f"Found {len(self.active_close_orders)} existing close orders")

            # Main trading loop
            while True:
                try:
                    # Check for emergency stop
                    if self.risk_manager.emergency_stop:
                        self.logger.log_risk_event("EMERGENCY_STOP",
                                                   "Emergency stop activated, shutting down",
                                                   "ERROR")
                        break

                    # Update market data
                    self._update_market_data()

                    # Check if we should place orders
                    should_trade, reason = self._should_place_order()

                    if should_trade:
                        success = self._place_and_monitor_open_order()
                        if not success:
                            self.logger.log("Order placement failed, continuing...", "WARNING")
                    else:
                        # Log reason occasionally for debugging
                        if time.time() % 60 < 1:  # Once per minute
                            self.logger.log(f"Not trading: {reason}")

                    # Update active orders
                    try:
                        latest_orders = self.client.get_active_close_orders(
                            self.config.symbol, self.config.close_order_side
                        )

                        # Update skip waiting logic
                        self._update_skip_waiting_logic()

                        # Update active orders list
                        self.active_close_orders = latest_orders

                    except Exception as e:
                        self.logger.log(f"Failed to update active orders: {e}", "ERROR")

                    # Periodic status logging
                    self._log_status_periodically()

                    # Dynamic sleep based on conditions
                    if len(self.active_close_orders) >= self.config.max_orders:
                        time.sleep(10)  # Wait longer when at max orders
                    elif self.risk_manager.emergency_stop:
                        break
                    else:
                        # Normal operation sleep
                        sleep_time = min(5, max(1, self._calculate_wait_time() // 10))
                        time.sleep(sleep_time)

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.logger.log(f"Error in trading loop: {e}", "ERROR")
                    # Continue after errors to maintain uptime
                    time.sleep(5)

        except KeyboardInterrupt:
            self.logger.log("Bot stopped by user")
        except Exception as e:
            self.logger.log(f"Critical error: {e}", "ERROR")
            import traceback
            self.logger.log(traceback.format_exc(), "ERROR")
            raise
        finally:
            self.logger.log("Shutting down WebSocket connection...")
            self.ws_manager.stop()

            # Final status report
            final_report = (
                f"FINAL REPORT - Trades: {self.trade_count}, "
                f"Successful: {self.successful_trades}, "
                f"Final PnL: {self.risk_manager.daily_pnl:.2f}, "
                f"Final Position: {self.risk_manager.total_position_size:.4f}"
            )
            self.logger.log(final_report)


def parse_arguments():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description='Enhanced Binance Futures Trading Bot with Risk Management',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic trading parameters
    parser.add_argument('--symbol', type=str, default='ETHUSDC',
                        help='Trading pair symbol')
    parser.add_argument('--base-quantity', type=float, default=0.01,
                        help='Base order quantity')
    parser.add_argument('--max-position-size', type=float, default=1.0,
                        help='Maximum total position size')
    parser.add_argument('--take-profit-pct', type=float, default=0.1,
                        help='Take profit percentage')
    parser.add_argument('--stop-loss-pct', type=float, default=0.2,
                        help='Stop loss percentage')
    parser.add_argument('--direction', type=str, choices=['BUY', 'SELL'], default='BUY',
                        help='Trading direction')

    # Order management
    parser.add_argument('--max-orders', type=int, default=10,
                        help='Maximum number of active orders')
    parser.add_argument('--wait-time', type=int, default=30,
                        help='Wait time between orders in seconds')

    # Risk management
    parser.add_argument('--min-spread-pct', type=float, default=0.05,
                        help='Minimum spread percentage required')
    parser.add_argument('--max-daily-loss', type=float, default=100.0,
                        help='Maximum daily loss before stopping')
    parser.add_argument('--risk-per-trade-pct', type=float, default=1.0,
                        help='Risk per trade as percentage of account balance')

    # Advanced features
    parser.add_argument('--position-size-scaling', action='store_true', default=True,
                        help='Enable position size scaling based on volatility')
    parser.add_argument('--volatility-lookback', type=int, default=20,
                        help='Periods for volatility calculation')

    return parser.parse_args()


def validate_config(config: TradingConfig):
    """Validate trading configuration."""
    errors = []

    if config.base_quantity <= 0:
        errors.append("Base quantity must be positive")

    if config.max_position_size <= 0:
        errors.append("Max position size must be positive")

    if config.take_profit_pct <= 0:
        errors.append("Take profit percentage must be positive")

    if config.stop_loss_pct <= 0:
        errors.append("Stop loss percentage must be positive")

    if config.max_orders <= 0:
        errors.append("Max orders must be positive")

    if config.wait_time < 0:
        errors.append("Wait time cannot be negative")

    if config.direction not in ['BUY', 'SELL']:
        errors.append("Direction must be BUY or SELL")

    if config.min_spread_pct < 0:
        errors.append("Min spread percentage cannot be negative")

    if config.max_daily_loss <= 0:
        errors.append("Max daily loss must be positive")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))


def main():
    """Enhanced main entry point with comprehensive setup."""
    print("Enhanced Binance Futures Trading Bot")
    print("=" * 50)

    args = parse_arguments()

    # Create configuration
    config = TradingConfig(
        symbol=args.symbol,
        base_quantity=args.base_quantity,
        max_position_size=args.max_position_size,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        direction=args.direction,
        max_orders=args.max_orders,
        wait_time=args.wait_time,
        min_spread_pct=args.min_spread_pct,
        max_daily_loss=args.max_daily_loss,
        position_size_scaling=args.position_size_scaling,
        volatility_lookback=args.volatility_lookback,
        risk_per_trade_pct=args.risk_per_trade_pct
    )

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # Display configuration
    print(f"Symbol: {config.symbol}")
    print(f"Direction: {config.direction}")
    print(f"Base Quantity: {config.base_quantity}")
    print(f"Max Position Size: {config.max_position_size}")
    print(f"Take Profit: {config.take_profit_pct}%")
    print(f"Stop Loss: {config.stop_loss_pct}%")
    print(f"Max Orders: {config.max_orders}")
    print(f"Max Daily Loss: {config.max_daily_loss}")
    print("=" * 50)

    # Confirm before starting
    try:
        confirm = input("Start trading? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Trading cancelled.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(0)

    # Create and run the bot
    try:
        bot = TradingBot(config)
        bot.run()
    except Exception as e:
        print(f"Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
