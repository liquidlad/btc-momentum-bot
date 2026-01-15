"""
Lighter exchange client implementation.
Connects to Lighter DEX for trading BTC perpetuals.
"""

import os
import asyncio
import logging
from typing import Optional, List, Callable
import time

from .base import (
    ExchangeClient, Order, Position, Balance, Candle, BBO,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class LighterClient(ExchangeClient):
    """
    Lighter exchange client.

    Uses the lighter-v2-python SDK for API interactions.
    """

    BASE_URL = "https://api.lighter.xyz"

    def __init__(
        self,
        market: str = "BTC-USD",
        paper_mode: bool = True,
    ):
        super().__init__(market, paper_mode)
        self.client = None
        self.order_api = None
        self.tx_api = None
        self.account_api = None
        self.account_index = None

        # Paper trading state
        self._paper_balance = 200.0
        self._paper_position: Optional[Position] = None
        self._paper_orders: List[Order] = []
        self._last_bbo: Optional[BBO] = None

    async def connect(self) -> bool:
        """Connect to Lighter."""
        try:
            if self.paper_mode:
                logger.info("Lighter: Running in PAPER mode (no real connection)")
                self.connected = True
                return True

            # Import Lighter SDK
            try:
                import lighter
                from lighter.api import AccountApi, OrderApi, TransactionApi
            except ImportError:
                logger.error("lighter-v2-python not installed. Run: pip install lighter-v2-python")
                return False

            # Get credentials
            account_index = os.environ.get("LIGHTER_ACCOUNT_INDEX")
            api_key_index = os.environ.get("LIGHTER_API_KEY_INDEX")
            private_key = os.environ.get("API_KEY_PRIVATE_KEY")

            if not all([account_index, api_key_index, private_key]):
                logger.error("Lighter: Missing credentials. Set LIGHTER_ACCOUNT_INDEX, LIGHTER_API_KEY_INDEX, API_KEY_PRIVATE_KEY")
                return False

            self.account_index = int(account_index)
            api_key_idx = int(api_key_index)

            # Initialize client
            self.client = lighter.SignerClient(
                url=self.BASE_URL,
                api_private_keys={api_key_idx: private_key},
                account_index=self.account_index
            )

            self.order_api = OrderApi(self.BASE_URL)
            self.tx_api = TransactionApi(self.BASE_URL, self.client)
            self.account_api = AccountApi(self.BASE_URL)

            self.connected = True
            logger.info("Lighter: Connected successfully")
            return True

        except Exception as e:
            logger.error(f"Lighter connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Lighter."""
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
            book = self.order_api.get_orderbook(self.market)
            bids = book.get("bids", [])
            asks = book.get("asks", [])

            bid_price = float(bids[0]["price"]) if bids else 0
            bid_size = float(bids[0]["size"]) if bids else 0
            ask_price = float(asks[0]["price"]) if asks else 0
            ask_size = float(asks[0]["size"]) if asks else 0

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
        # Lighter API candle fetching - implementation depends on their API
        return []

    async def subscribe_bbo(self, callback: Callable) -> None:
        """Subscribe to real-time BBO updates."""
        if self.paper_mode:
            logger.info("Lighter: BBO subscription registered (paper mode)")
            return

        try:
            from lighter.ws import WsClient
            ws_client = WsClient(self.BASE_URL)
            ws_client.subscribe_orderbook(self.market)
            logger.info(f"Lighter: Subscribed to BBO for {self.market}")
        except Exception as e:
            logger.error(f"Error subscribing to BBO: {e}")

    async def subscribe_trades(self, callback: Callable) -> None:
        """Subscribe to real-time trade updates."""
        if self.paper_mode:
            logger.info("Lighter: Trades subscription registered (paper mode)")
            return
        # Implement based on Lighter WebSocket API
        pass

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
            account = self.account_api.get_account(account_index=self.account_index)
            return Balance(
                currency="USDC",
                available=float(account.get("available_balance", 0)),
                total=float(account.get("total_balance", 0)),
                in_orders=0
            )
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
            account = self.account_api.get_account(account_index=self.account_index)
            positions = account.get("positions", [])
            return [
                Position(
                    market=p.get("market"),
                    side="LONG" if float(p.get("size", 0)) > 0 else "SHORT",
                    size=abs(float(p.get("size", 0))),
                    entry_price=float(p.get("entry_price", 0)),
                    unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                )
                for p in positions
                if p.get("market") == self.market and float(p.get("size", 0)) != 0
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    async def get_open_orders(self) -> List[Order]:
        """Get open orders."""
        if self.paper_mode:
            return [o for o in self._paper_orders if o.status == OrderStatus.OPEN]
        # Implement based on Lighter API
        return []

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

            logger.info(f"PAPER ORDER: {side.value} {size} @ {fill_price:.2f}")
            return order

        try:
            nonce = self.tx_api.next_nonce()

            order_data = {
                "market": self.market,
                "side": side.value,
                "type": order_type.value,
                "base_amount": int(size * 1e6),
                "price": int(price * 1e6) if price else 0,
                "time_in_force": "IMMEDIATE_OR_CANCEL" if order_type == OrderType.MARKET else "GOOD_TILL_TIME",
                "nonce": nonce,
                "client_order_index": int(time.time() * 1000) % 1000000
            }

            result = self.tx_api.send_tx(order_data)

            return Order(
                id=str(result.get("order_id", "")),
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
        # Implement based on Lighter API
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
        # Implement based on Lighter API
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
