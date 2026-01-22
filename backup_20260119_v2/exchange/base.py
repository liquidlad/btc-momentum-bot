"""
Base exchange client interface.
All exchange implementations should inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    client_id: str
    market: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float]
    status: OrderStatus
    filled_size: float = 0
    avg_fill_price: float = 0
    timestamp: int = 0


@dataclass
class Position:
    """Represents an open position."""
    market: str
    side: str  # "LONG" or "SHORT"
    size: float
    entry_price: float
    unrealized_pnl: float
    liquidation_price: Optional[float] = None


@dataclass
class Balance:
    """Represents account balance."""
    currency: str
    available: float
    total: float
    in_orders: float = 0


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BBO:
    """Best bid/offer data."""
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: int


class ExchangeClient(ABC):
    """Abstract base class for exchange clients."""

    def __init__(self, market: str = "BTC-USD-PERP", paper_mode: bool = True):
        self.market = market
        self.paper_mode = paper_mode
        self.connected = False

    # Connection methods
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    # Market data methods
    @abstractmethod
    async def get_bbo(self) -> BBO:
        """Get best bid/offer."""
        pass

    @abstractmethod
    async def get_candles(self, resolution: str, start_time: int, end_time: int) -> List[Candle]:
        """Get historical candles."""
        pass

    @abstractmethod
    async def subscribe_bbo(self, callback) -> None:
        """Subscribe to real-time BBO updates."""
        pass

    @abstractmethod
    async def subscribe_trades(self, callback) -> None:
        """Subscribe to real-time trade updates."""
        pass

    # Account methods
    @abstractmethod
    async def get_balance(self) -> Balance:
        """Get account balance."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get open positions."""
        pass

    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """Get open orders."""
        pass

    # Order methods
    @abstractmethod
    async def place_order(
        self,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> Order:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        pass

    # Utility methods
    def get_mid_price(self, bbo: BBO) -> float:
        """Calculate mid price from BBO."""
        return (bbo.bid_price + bbo.ask_price) / 2
