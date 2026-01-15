"""
Paradex exchange client implementation.
Connects to Paradex DEX for trading BTC perpetuals.
"""

import os
import asyncio
import logging
from typing import Optional, List, Callable
from datetime import datetime
import time

from .base import (
    ExchangeClient, Order, Position, Balance, Candle, BBO,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class ParadexClient(ExchangeClient):
    """
    Paradex exchange client.

    Uses the paradex-py SDK for API interactions.
    Supports both L1+L2 and L2-only authentication.
    """

    def __init__(
        self,
        market: str = "BTC-USD-PERP",
        paper_mode: bool = True,
        use_testnet: bool = False,
    ):
        super().__init__(market, paper_mode)
        self.use_testnet = use_testnet
        self.paradex = None
        self.ws_connected = False
        self._bbo_callbacks: List[Callable] = []
        self._trade_callbacks: List[Callable] = []
        self._position_callbacks: List[Callable] = []
        self._order_callbacks: List[Callable] = []

        # Paper trading state
        self._paper_balance = 200.0
        self._paper_position: Optional[Position] = None
        self._paper_orders: List[Order] = []
        self._last_bbo: Optional[BBO] = None

    async def connect(self) -> bool:
        """Connect to Paradex."""
        try:
            if self.paper_mode:
                logger.info("Paradex: Running in PAPER mode (no real connection)")
                self.connected = True
                return True

            # Import Paradex SDK
            try:
                from paradex_py import Paradex, ParadexSubkey
                from paradex_py.environment import Environment
            except ImportError:
                logger.error("paradex-py not installed. Run: pip install paradex-py")
                return False

            env = Environment.TESTNET if self.use_testnet else Environment.PROD

            # Try L2-only auth first (preferred for bots)
            l2_private_key = os.environ.get("PARADEX_L2_PRIVATE_KEY")
            l2_address = os.environ.get("PARADEX_L2_ADDRESS")

            if l2_private_key and l2_address:
                logger.info("Paradex: Using L2-only authentication")
                self.paradex = ParadexSubkey(
                    env=env,
                    l2_private_key=l2_private_key,
                    l2_address=l2_address
                )
            else:
                # Fall back to L1+L2 auth
                l1_address = os.environ.get("PARADEX_L1_ADDRESS")
                l1_private_key = os.environ.get("PARADEX_L1_PRIVATE_KEY")

                if not l1_address or not l1_private_key:
                    logger.error("Paradex: Missing credentials. Set PARADEX_L2_PRIVATE_KEY/ADDRESS or L1 credentials")
                    return False

                logger.info("Paradex: Using L1+L2 authentication")
                self.paradex = Paradex(
                    env=env,
                    l1_address=l1_address,
                    l1_private_key=l1_private_key
                )

            self.connected = True
            logger.info(f"Paradex: Connected to {'TESTNET' if self.use_testnet else 'PROD'}")
            return True

        except Exception as e:
            logger.error(f"Paradex connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Paradex."""
        if self.paradex and not self.paper_mode:
            try:
                await self.paradex.close()
            except Exception as e:
                logger.warning(f"Error closing Paradex connection: {e}")
        self.connected = False
        logger.info("Paradex: Disconnected")

    async def get_bbo(self) -> BBO:
        """Get best bid/offer."""
        if self.paper_mode:
            # Return simulated BBO for paper trading
            # In real paper mode, we'd fetch from public API
            if self._last_bbo:
                return self._last_bbo
            # Default simulated price
            return BBO(
                bid_price=94990.0,
                bid_size=1.0,
                ask_price=95010.0,
                ask_size=1.0,
                timestamp=int(time.time() * 1000)
            )

        try:
            bbo_data = self.paradex.api_client.fetch_bbo(self.market)
            return BBO(
                bid_price=float(bbo_data.get("bid_price", 0)),
                bid_size=float(bbo_data.get("bid_size", 0)),
                ask_price=float(bbo_data.get("ask_price", 0)),
                ask_size=float(bbo_data.get("ask_size", 0)),
                timestamp=int(time.time() * 1000)
            )
        except Exception as e:
            logger.error(f"Error fetching BBO: {e}")
            raise

    async def get_candles(self, resolution: str, start_time: int, end_time: int) -> List[Candle]:
        """Get historical candles."""
        if self.paper_mode:
            return []  # Would need to fetch from public API

        try:
            klines = self.paradex.api_client.fetch_klines(
                symbol=self.market,
                resolution=resolution,
                start_at=start_time,
                end_at=end_time
            )
            return [
                Candle(
                    timestamp=int(k["timestamp"]),
                    open=float(k["open"]),
                    high=float(k["high"]),
                    low=float(k["low"]),
                    close=float(k["close"]),
                    volume=float(k["volume"])
                )
                for k in klines
            ]
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            raise

    async def subscribe_bbo(self, callback: Callable) -> None:
        """Subscribe to real-time BBO updates."""
        self._bbo_callbacks.append(callback)

        if self.paper_mode:
            logger.info("Paradex: BBO subscription registered (paper mode)")
            return

        try:
            from paradex_py.message.websocket import ParadexWebsocketChannel

            async def on_bbo(channel, message):
                bbo = BBO(
                    bid_price=float(message.get("bid_price", 0)),
                    bid_size=float(message.get("bid_size", 0)),
                    ask_price=float(message.get("ask_price", 0)),
                    ask_size=float(message.get("ask_size", 0)),
                    timestamp=int(time.time() * 1000)
                )
                self._last_bbo = bbo
                for cb in self._bbo_callbacks:
                    await cb(bbo)

            await self.paradex.ws_client.connect()
            await self.paradex.ws_client.subscribe(
                ParadexWebsocketChannel.BBO,
                callback=on_bbo,
                params={"market": self.market}
            )
            self.ws_connected = True
            logger.info(f"Paradex: Subscribed to BBO for {self.market}")
        except Exception as e:
            logger.error(f"Error subscribing to BBO: {e}")
            raise

    async def subscribe_trades(self, callback: Callable) -> None:
        """Subscribe to real-time trade updates."""
        self._trade_callbacks.append(callback)

        if self.paper_mode:
            logger.info("Paradex: Trades subscription registered (paper mode)")
            return

        try:
            from paradex_py.message.websocket import ParadexWebsocketChannel

            async def on_trade(channel, message):
                for cb in self._trade_callbacks:
                    await cb(message)

            await self.paradex.ws_client.subscribe(
                ParadexWebsocketChannel.TRADES,
                callback=on_trade,
                params={"market": self.market}
            )
            logger.info(f"Paradex: Subscribed to trades for {self.market}")
        except Exception as e:
            logger.error(f"Error subscribing to trades: {e}")
            raise

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
            balances = self.paradex.api_client.fetch_balances()
            usdc_balance = next((b for b in balances if b.get("currency") == "USDC"), {})
            return Balance(
                currency="USDC",
                available=float(usdc_balance.get("available", 0)),
                total=float(usdc_balance.get("total", 0)),
                in_orders=float(usdc_balance.get("in_orders", 0))
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
            positions = self.paradex.api_client.fetch_positions()
            return [
                Position(
                    market=p.get("market"),
                    side="LONG" if float(p.get("size", 0)) > 0 else "SHORT",
                    size=abs(float(p.get("size", 0))),
                    entry_price=float(p.get("entry_price", 0)),
                    unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                    liquidation_price=float(p.get("liquidation_price")) if p.get("liquidation_price") else None
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

        try:
            orders = self.paradex.api_client.fetch_orders({"market": self.market})
            return [
                Order(
                    id=o.get("id"),
                    client_id=o.get("client_id", ""),
                    market=o.get("market"),
                    side=OrderSide.BUY if o.get("side") == "BUY" else OrderSide.SELL,
                    order_type=OrderType.LIMIT if o.get("type") == "LIMIT" else OrderType.MARKET,
                    size=float(o.get("size", 0)),
                    price=float(o.get("price")) if o.get("price") else None,
                    status=OrderStatus.OPEN,
                    filled_size=float(o.get("filled_size", 0)),
                    avg_fill_price=float(o.get("avg_fill_price", 0)),
                    timestamp=int(o.get("timestamp", 0))
                )
                for o in orders
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
                # New position
                self._paper_position = Position(
                    market=self.market,
                    side="LONG" if side == OrderSide.BUY else "SHORT",
                    size=size,
                    entry_price=fill_price,
                    unrealized_pnl=0
                )
            else:
                # Modify existing position
                current_side = self._paper_position.side
                if (current_side == "LONG" and side == OrderSide.SELL) or \
                   (current_side == "SHORT" and side == OrderSide.BUY):
                    # Closing position
                    if size >= self._paper_position.size:
                        # Close completely
                        pnl = self._calculate_pnl(self._paper_position, fill_price)
                        self._paper_balance += pnl
                        self._paper_position = None
                    else:
                        # Partial close
                        self._paper_position.size -= size
                else:
                    # Adding to position
                    self._paper_position.size += size
                    # Average in
                    self._paper_position.entry_price = (
                        (self._paper_position.entry_price * (self._paper_position.size - size) +
                         fill_price * size) / self._paper_position.size
                    )

            logger.info(f"PAPER ORDER: {side.value} {size} @ {fill_price:.2f}")
            return order

        try:
            from paradex_py.common.order import Order as ParadexOrder, OrderSide as PSide, OrderType as PType

            order_obj = ParadexOrder(
                market=self.market,
                side=PSide.BUY if side == OrderSide.BUY else PSide.SELL,
                type=PType.LIMIT if order_type == OrderType.LIMIT else PType.MARKET,
                size=str(size),
                price=str(price) if price else None,
                client_id=client_id
            )

            result = self.paradex.api_client.submit_order(order_obj)

            return Order(
                id=result.get("id", ""),
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
            self.paradex.api_client.cancel_order(order_id=order_id)
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
            self.paradex.api_client.cancel_all_orders({"market": self.market})
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
        """Update BBO for paper trading simulations."""
        self._last_bbo = bbo

        # Update unrealized PnL for paper position
        if self._paper_position:
            mid_price = (bbo.bid_price + bbo.ask_price) / 2
            self._paper_position.unrealized_pnl = self._calculate_pnl(
                self._paper_position, mid_price
            )
