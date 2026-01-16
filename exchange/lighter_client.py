"""
Lighter exchange client implementation.
Connects to Lighter DEX for trading perpetuals.

Lighter has 0% fees for standard accounts (both maker and taker).
Trade-off: 200-300ms latency (acceptable for 1-minute candle strategies).
"""

import os
import asyncio
import logging
from typing import Optional, List, Callable
from decimal import Decimal
import time

from .base import (
    ExchangeClient, Order, Position, Balance, Candle, BBO,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class LighterClient(ExchangeClient):
    """
    Lighter exchange client.

    Uses the lighter-sdk for API interactions.
    Standard accounts have 0% maker and 0% taker fees.
    """

    def __init__(
        self,
        market: str = "BTC-USD-PERP",
        paper_mode: bool = True,
    ):
        super().__init__(market, paper_mode)
        self.signer_client = None
        self.order_api = None
        self.account_api = None
        self.tx_api = None
        self.account_index = None
        self._orderbook_id = None  # Lighter uses orderbook IDs

        # Paper trading state
        self._paper_balance = 50.0
        self._paper_position: Optional[Position] = None
        self._paper_orders: List[Order] = []
        self._last_bbo: Optional[BBO] = None

        # Market to orderbook mapping (will be fetched on connect)
        self._market_to_orderbook = {}

    async def connect(self) -> bool:
        """Connect to Lighter."""
        try:
            if self.paper_mode:
                logger.info("Lighter: Running in PAPER mode (no real connection)")
                self.connected = True
                return True

            # Import Lighter SDK
            try:
                from lighter import SignerClient, OrderApi, AccountApi, TransactionApi, Configuration
            except ImportError:
                logger.error("lighter-sdk not installed. Run: pip install lighter-sdk")
                return False

            # Get credentials from environment
            account_index = os.environ.get("LIGHTER_ACCOUNT_INDEX")
            api_key_index = os.environ.get("LIGHTER_API_KEY_INDEX")
            private_key = os.environ.get("LIGHTER_PRIVATE_KEY")

            if not all([account_index, private_key]):
                logger.error("Lighter: Missing credentials. Set LIGHTER_ACCOUNT_INDEX and LIGHTER_PRIVATE_KEY")
                return False

            self.account_index = int(account_index)
            api_key_idx = int(api_key_index) if api_key_index else 0

            # Get the base URL from configuration
            config = Configuration()
            base_url = config.host

            logger.info(f"Lighter: Connecting to {base_url}")

            # Initialize signer client
            self.signer_client = SignerClient(
                url=base_url,
                account_index=self.account_index,
                api_private_keys={api_key_idx: private_key}
            )

            # Initialize API clients
            self.order_api = OrderApi()
            self.account_api = AccountApi()
            self.tx_api = TransactionApi()

            # Get orderbook info to find the right orderbook ID for our market
            await self._fetch_orderbook_mapping()

            self.connected = True
            logger.info(f"Lighter: Connected successfully (account {self.account_index})")
            return True

        except Exception as e:
            logger.error(f"Lighter connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _fetch_orderbook_mapping(self):
        """Fetch orderbook IDs for markets."""
        try:
            orderbooks = await self.order_api.order_books()
            for ob in orderbooks.order_books:
                if ob.market_type == 'perp':
                    # Map symbol to market_id (e.g., "BTC" -> 1)
                    # Also map our market format (e.g., "BTC-USD-PERP" -> 1)
                    self._market_to_orderbook[ob.symbol] = ob.market_id
                    self._market_to_orderbook[f"{ob.symbol}-USD-PERP"] = ob.market_id

            # Extract symbol from our market format (e.g., "BTC-USD-PERP" -> "BTC")
            symbol = self.market.split("-")[0] if "-" in self.market else self.market

            if self.market in self._market_to_orderbook:
                self._orderbook_id = self._market_to_orderbook[self.market]
                logger.info(f"Lighter: Using market_id {self._orderbook_id} for {self.market}")
            elif symbol in self._market_to_orderbook:
                self._orderbook_id = self._market_to_orderbook[symbol]
                logger.info(f"Lighter: Using market_id {self._orderbook_id} for {symbol}")
            else:
                logger.warning(f"Lighter: Market {self.market} not found. Available: BTC, ETH, SOL, etc.")
        except Exception as e:
            logger.error(f"Error fetching orderbook mapping: {e}")
            import traceback
            traceback.print_exc()

    async def disconnect(self) -> None:
        """Disconnect from Lighter."""
        if self.signer_client and not self.paper_mode:
            try:
                await self.signer_client.close()
            except Exception as e:
                logger.warning(f"Error closing Lighter connection: {e}")
        self.connected = False
        logger.info("Lighter: Disconnected")

    async def get_bbo(self) -> BBO:
        """Get best bid/offer."""
        if self.paper_mode:
            if self._last_bbo:
                return self._last_bbo
            return BBO(
                bid_price=94990.0,
                bid_size=1.0,
                ask_price=95010.0,
                ask_size=1.0,
                timestamp=int(time.time() * 1000)
            )

        try:
            if not self._orderbook_id:
                raise ValueError(f"No orderbook ID for market {self.market}")

            # Get top of book from order_book_orders
            orders = await self.order_api.order_book_orders(market_id=self._orderbook_id, limit=1)

            bid_price = 0
            bid_size = 0
            ask_price = 0
            ask_size = 0

            if orders.bids and len(orders.bids) > 0:
                bid_price = float(orders.bids[0].price)
                bid_size = float(orders.bids[0].remaining_base_amount)

            if orders.asks and len(orders.asks) > 0:
                ask_price = float(orders.asks[0].price)
                ask_size = float(orders.asks[0].remaining_base_amount)

            return BBO(
                bid_price=bid_price,
                bid_size=bid_size,
                ask_price=ask_price,
                ask_size=ask_size,
                timestamp=int(time.time() * 1000)
            )
        except Exception as e:
            logger.error(f"Error fetching BBO: {e}")
            raise

    async def get_candles(self, resolution: str, start_time: int, end_time: int) -> List[Candle]:
        """Get historical candles."""
        if self.paper_mode:
            return []

        try:
            from lighter import CandlestickApi
            candle_api = CandlestickApi()

            candles = await candle_api.candlesticks(
                market_id=self._orderbook_id,
                resolution=resolution,
                start_time=start_time,
                end_time=end_time
            )

            return [
                Candle(
                    timestamp=int(c.open_timestamp),
                    open=float(c.open),
                    high=float(c.high),
                    low=float(c.low),
                    close=float(c.close),
                    volume=float(c.volume)
                )
                for c in candles.candlesticks
            ]
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return []

    async def subscribe_bbo(self, callback: Callable) -> None:
        """Subscribe to real-time BBO updates."""
        if self.paper_mode:
            logger.info("Lighter: BBO subscription registered (paper mode)")
            return

        try:
            from lighter import WsClient
            # WebSocket subscription - implementation depends on Lighter's WS API
            logger.info(f"Lighter: BBO subscription for {self.market}")
        except Exception as e:
            logger.error(f"Error subscribing to BBO: {e}")

    async def subscribe_trades(self, callback: Callable) -> None:
        """Subscribe to real-time trade updates."""
        if self.paper_mode:
            logger.info("Lighter: Trades subscription registered (paper mode)")
            return

    async def get_balance(self) -> Balance:
        """Get account balance."""
        if self.paper_mode:
            return Balance(
                currency="USDC",
                available=self._paper_balance,
                total=self._paper_balance,
                in_orders=0
            )

        try:
            account = await self.account_api.account(by='index', value=str(self.account_index))
            # Account info is in accounts list
            if hasattr(account, 'accounts') and account.accounts:
                acc = account.accounts[0]
                collateral = float(acc.collateral) if acc.collateral else 0
                available = float(acc.available_balance) if acc.available_balance else collateral
                return Balance(
                    currency="USDC",
                    available=available,
                    total=collateral,
                    in_orders=0
                )
            return Balance(currency="USDC", available=0, total=0, in_orders=0)
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def get_positions(self) -> List[Position]:
        """Get open positions."""
        if self.paper_mode:
            if self._paper_position:
                return [self._paper_position]
            return []

        try:
            account = await self.account_api.account(by='index', value=str(self.account_index))
            positions = []

            # Positions are in accounts list
            if hasattr(account, 'accounts') and account.accounts:
                acc = account.accounts[0]
                if hasattr(acc, 'positions'):
                    for pos in (acc.positions or []):
                        pos_market_id = getattr(pos, 'market_id', None) or getattr(pos, 'order_book_id', None)
                        if pos_market_id == self._orderbook_id and float(pos.size or 0) != 0:
                            size = float(pos.size)
                            positions.append(Position(
                                market=self.market,
                                side="LONG" if size > 0 else "SHORT",
                                size=abs(size),
                                entry_price=float(pos.entry_price or 0),
                                unrealized_pnl=float(pos.unrealized_pnl or 0),
                            ))

            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    async def get_open_orders(self) -> List[Order]:
        """Get open orders."""
        if self.paper_mode:
            return [o for o in self._paper_orders if o.status == OrderStatus.OPEN]

        try:
            orders = await self.order_api.account_active_orders(account_index=self.account_index)
            return [
                Order(
                    id=str(o.order_id),
                    client_id="",
                    market=self.market,
                    side=OrderSide.BUY if o.is_bid else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    size=float(o.size or 0),
                    price=float(o.price or 0),
                    status=OrderStatus.OPEN,
                    filled_size=float(o.filled_size or 0),
                    avg_fill_price=0,
                    timestamp=int(time.time() * 1000)
                )
                for o in (orders.orders or [])
                if o.market_id == self._orderbook_id
            ]
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            raise

    async def place_order(
        self,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        client_id: Optional[str] = None,
    ) -> Order:
        """Place an order."""
        client_id = client_id or f"bot_{int(time.time() * 1000)}"

        if self.paper_mode:
            # Simulate order execution
            bbo = await self.get_bbo()
            fill_price = bbo.ask_price if side == OrderSide.BUY else bbo.bid_price

            order = Order(
                id=f"paper_{int(time.time() * 1000)}",
                client_id=client_id,
                market=self.market,
                side=side,
                order_type=order_type,
                size=size,
                price=price,
                status=OrderStatus.FILLED,
                filled_size=size,
                avg_fill_price=fill_price,
                timestamp=int(time.time() * 1000)
            )

            # Update paper position
            if self._paper_position is None:
                self._paper_position = Position(
                    market=self.market,
                    side="LONG" if side == OrderSide.BUY else "SHORT",
                    size=size,
                    entry_price=fill_price,
                    unrealized_pnl=0
                )
            else:
                current_side = self._paper_position.side
                if (current_side == "LONG" and side == OrderSide.SELL) or \
                   (current_side == "SHORT" and side == OrderSide.BUY):
                    if size >= self._paper_position.size:
                        pnl = self._calculate_pnl(self._paper_position, fill_price)
                        self._paper_balance += pnl
                        self._paper_position = None
                    else:
                        self._paper_position.size -= size
                else:
                    self._paper_position.size += size
                    self._paper_position.entry_price = (
                        (self._paper_position.entry_price * (self._paper_position.size - size) +
                         fill_price * size) / self._paper_position.size
                    )

            logger.info(f"PAPER ORDER: {side.value} {size} @ {fill_price:.2f}")
            return order

        try:
            if not self._orderbook_id:
                raise ValueError(f"No orderbook ID for market {self.market}")

            is_bid = side == OrderSide.BUY

            # Get current price for market orders
            bbo = await self.get_bbo()
            current_price = bbo.ask_price if side == OrderSide.BUY else bbo.bid_price

            # Generate unique client order index
            client_order_idx = int(time.time() * 1000) % (2**32)

            # is_ask = True for SELL, False for BUY
            is_ask = side == OrderSide.SELL

            # Convert to integer format (Lighter uses smallest units)
            # BTC: size_decimals=5, price_decimals=1
            size_int = int(size * 100000)  # 5 decimals
            price_int = int(current_price * 10)  # 1 decimal

            if order_type == OrderType.MARKET:
                # Use create_market_order for market orders
                # Parameters: market_index, client_order_index, base_amount, avg_execution_price, is_ask
                logger.info(f"Placing market order: market={self._orderbook_id}, size={size_int}, price={price_int}, is_ask={is_ask}")
                result = await self.signer_client.create_market_order(
                    market_index=self._orderbook_id,
                    client_order_index=client_order_idx,
                    base_amount=size_int,
                    avg_execution_price=price_int,
                    is_ask=is_ask
                )
                # Result is tuple: (tx, response, error)
                tx, response, error = result
                if error:
                    logger.error(f"Order failed: {error}")
                    raise ValueError(f"Order failed: {error}")
                logger.info(f"Order response: {response}")
            else:
                # Use create_order for limit orders
                if price is None:
                    raise ValueError("Price required for limit orders")

                limit_price_int = int(price * 10)  # 1 decimal
                logger.info(f"Placing limit order: market={self._orderbook_id}, size={size_int}, price={limit_price_int}, is_ask={is_ask}")
                result = await self.signer_client.create_order(
                    market_index=self._orderbook_id,
                    client_order_index=client_order_idx,
                    base_amount=size_int,
                    price=limit_price_int,
                    is_ask=is_ask,
                    order_type=self.signer_client.ORDER_TYPE_LIMIT,
                    time_in_force=self.signer_client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME
                )
                # Result is tuple: (tx, response, error)
                tx, response, error = result
                if error:
                    logger.error(f"Order failed: {error}")
                    raise ValueError(f"Order failed: {error}")
                logger.info(f"Order response: {response}")

            logger.info(f"Lighter order placed: {side.value} {size} @ {price or 'MARKET'}")

            return Order(
                id=str(result.tx_hash if hasattr(result, 'tx_hash') else int(time.time() * 1000)),
                client_id=client_id,
                market=self.market,
                side=side,
                order_type=order_type,
                size=size,
                price=price,
                status=OrderStatus.OPEN,
                filled_size=0,
                avg_fill_price=0,
                timestamp=int(time.time() * 1000)
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.paper_mode:
            for order in self._paper_orders:
                if order.id == order_id:
                    order.status = OrderStatus.CANCELLED
                    return True
            return False

        try:
            self.signer_client.cancel_order(
                market_index=self._orderbook_id,
                order_index=int(order_id)
            )
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        if self.paper_mode:
            count = 0
            for order in self._paper_orders:
                if order.status == OrderStatus.OPEN:
                    order.status = OrderStatus.CANCELLED
                    count += 1
            return count

        try:
            # cancel_all_orders takes time_in_force and timestamp_ms
            self.signer_client.cancel_all_orders(
                time_in_force=0,  # Cancel all immediately
                timestamp_ms=int(time.time() * 1000)
            )
            return -1  # Unknown count
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0

    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate P&L for a position."""
        if position.side == "LONG":
            return (exit_price - position.entry_price) * position.size
        else:
            return (position.entry_price - exit_price) * position.size

    def update_paper_bbo(self, bbo: BBO) -> None:
        """Update BBO for paper trading."""
        self._last_bbo = bbo
        if self._paper_position:
            mid_price = (bbo.bid_price + bbo.ask_price) / 2
            self._paper_position.unrealized_pnl = self._calculate_pnl(
                self._paper_position, mid_price
            )
