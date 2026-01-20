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

# Global nonce lock shared across ALL LighterClient instances
# This prevents nonce collisions when multiple clients (BTC, ETH, SOL)
# share the same Lighter account and try to place orders simultaneously
_global_order_lock = asyncio.Lock()

# Track the last order time to add minimum spacing between orders
_last_order_time = 0
_MIN_ORDER_SPACING_MS = 200  # Minimum 200ms between orders

# CRITICAL: Share a single SignerClient across ALL LighterClient instances
# Each SignerClient has its own internal nonce counter. If we create separate
# SignerClients for BTC, ETH, SOL, they will each track nonces independently,
# causing nonce collisions when orders are placed on different markets.
_shared_signer_client = None
_shared_order_api = None
_shared_account_api = None
_shared_tx_api = None
_shared_account_index = None


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
        global _shared_signer_client, _shared_order_api, _shared_account_api, _shared_tx_api, _shared_account_index

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

            # Use shared signer client to prevent nonce collisions
            # All markets (BTC, ETH, SOL) must share the same SignerClient
            # because they share the same account and nonce sequence
            if _shared_signer_client is None:
                # First client to connect initializes the shared resources
                config = Configuration()
                base_url = config.host

                logger.info(f"Lighter: Connecting to {base_url}")

                _shared_signer_client = SignerClient(
                    url=base_url,
                    account_index=self.account_index,
                    api_private_keys={api_key_idx: private_key}
                )
                _shared_order_api = OrderApi()
                _shared_account_api = AccountApi()
                _shared_tx_api = TransactionApi()
                _shared_account_index = self.account_index

                logger.info(f"Lighter: Created shared SignerClient for account {self.account_index}")
            else:
                logger.info(f"Lighter: Reusing shared SignerClient for {self.market}")

            # Use the shared clients
            self.signer_client = _shared_signer_client
            self.order_api = _shared_order_api
            self.account_api = _shared_account_api
            self.tx_api = _shared_tx_api

            # Get orderbook info to find the right orderbook ID for our market
            await self._fetch_orderbook_mapping()

            # Only sync nonces on first connect (shared client handles nonce tracking)
            if _shared_account_index == self.account_index:
                await self._sync_nonces()

            self.connected = True
            logger.info(f"Lighter: Connected successfully (account {self.account_index})")
            return True

        except Exception as e:
            logger.error(f"Lighter connection failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _sync_nonces(self, force_refresh: bool = False):
        """
        Sync nonces from the server to avoid invalid nonce errors.

        Args:
            force_refresh: If True, always fetch fresh nonce from server
        """
        try:
            # Fetch account info to get current nonce state
            account = await self.account_api.account(by='index', value=str(self.account_index))
            if hasattr(account, 'accounts') and account.accounts:
                acc = account.accounts[0]

                # Try multiple possible attribute names for nonce
                nonce = None
                for attr in ['nonce', 'order_nonce', 'tx_nonce', 'sequence']:
                    nonce = getattr(acc, attr, None)
                    if nonce is not None:
                        break

                if nonce is not None:
                    logger.debug(f"Lighter: Account nonce from server: {nonce}")

                # If the signer_client has a way to set nonce, use it
                if nonce is not None:
                    if hasattr(self.signer_client, 'set_nonce'):
                        self.signer_client.set_nonce(int(nonce))
                        logger.info(f"Lighter: Set signer nonce via set_nonce() to {nonce}")
                    elif hasattr(self.signer_client, '_nonce'):
                        self.signer_client._nonce = int(nonce)
                        logger.info(f"Lighter: Set signer _nonce to {nonce}")
                    elif hasattr(self.signer_client, 'nonce'):
                        self.signer_client.nonce = int(nonce)
                        logger.info(f"Lighter: Set signer nonce to {nonce}")
                    else:
                        logger.warning(f"Lighter: Could not find way to set nonce on signer_client")
        except Exception as e:
            logger.warning(f"Lighter: Could not sync nonce: {e}")
            import traceback
            traceback.print_exc()

    async def place_order_until_filled(
        self,
        side: 'OrderSide',
        size: float,
        order_type: 'OrderType' = None,
        max_attempts: int = 5,
        verify_timeout: float = 5.0,
        closing_side: str = None,  # "SHORT" if closing a short, "LONG" if closing a long
    ) -> 'Order':
        """
        Place an order and retry with increasing slippage until it fills.

        This is critical for exit orders that MUST fill to close positions.
        Uses slippage ramp: 0.2% -> 0.5% -> 1% -> 1.5% -> 2%.

        Args:
            side: BUY or SELL
            size: Order size
            order_type: Order type (defaults to MARKET)
            max_attempts: Maximum number of attempts
            verify_timeout: Seconds to wait for fill verification
            closing_side: "SHORT" if closing a short, "LONG" if closing a long, None if opening

        Returns:
            Order object if successful

        Raises:
            ValueError if order fails after all attempts
        """
        if order_type is None:
            from .base import OrderType
            order_type = OrderType.MARKET

        last_error = None
        last_order = None
        # Slippage ramp for exits: 0.2% -> 0.5% -> 1% -> 1.5% -> 2%
        slippage_schedule = [0.2, 0.5, 1.0, 1.5, 2.0]

        for attempt in range(max_attempts):
            slippage = slippage_schedule[min(attempt, len(slippage_schedule) - 1)]

            # CRITICAL: Before retry, check if position still needs closing
            if attempt > 0 and closing_side is not None:
                try:
                    current_positions = await self.get_positions()
                    has_target_position = False
                    has_opposite_position = False
                    opposite_side = "LONG" if closing_side == "SHORT" else "SHORT"

                    for pos in current_positions:
                        if pos.side == closing_side and pos.size > 0.0001:
                            has_target_position = True
                        elif pos.side == opposite_side and pos.size > 0.0001:
                            has_opposite_position = True

                    if has_opposite_position:
                        logger.warning(f"Lighter: Position FLIPPED to {opposite_side} - {closing_side} is closed, aborting retries")
                        return last_order  # Don't send more orders

                    if not has_target_position:
                        logger.info(f"Lighter: No {closing_side} position remains - aborting retries")
                        return last_order  # Position already closed

                except Exception as e:
                    logger.warning(f"Lighter: Could not check position before retry: {e}")
                    # Continue with retry if we can't check

            try:
                # Place order with explicit slippage for exits
                logger.info(f"Exit attempt {attempt + 1}/{max_attempts} with {slippage}% slippage")
                order = await self.place_order(
                    side=side,
                    size=size,
                    order_type=order_type,
                    slippage_pct=slippage,  # Aggressive slippage for exits
                )
                last_order = order

                # Verify the order filled
                if not self.paper_mode:
                    filled = await self.verify_order_filled(side, size, timeout_seconds=verify_timeout, closing_side=closing_side)
                    if filled:
                        logger.info(f"Lighter: Order confirmed filled after {attempt + 1} attempt(s)")
                        return order
                    else:
                        # Order placed but not filled - try again with more slippage
                        logger.warning(f"Lighter: Order not filled, retrying with more slippage (attempt {attempt + 1}/{max_attempts})")
                        continue
                else:
                    return order

            except Exception as e:
                last_error = e
                logger.warning(f"Lighter: Order attempt {attempt + 1}/{max_attempts} failed: {e}")
                await asyncio.sleep(0.5)
                continue

        # All attempts failed
        error_msg = f"Order failed after {max_attempts} attempts. Last error: {last_error}"
        logger.error(f"Lighter: {error_msg}")
        raise ValueError(error_msg)

    async def verify_order_filled(self, side: 'OrderSide', expected_size: float, timeout_seconds: float = 10.0, closing_side: str = None) -> bool:
        """
        Verify an order was filled by checking position changes.

        For closing positions (BUY to close short, SELL to close long),
        we verify the position size decreased or was eliminated.

        Args:
            side: Order side (BUY or SELL)
            expected_size: Expected order size
            timeout_seconds: How long to wait for verification
            closing_side: "SHORT" if closing a short, "LONG" if closing a long, None if opening

        Returns True if order appears to have filled, False otherwise.
        """
        if self.paper_mode:
            return True  # Paper mode always "fills"

        try:
            import time
            start_time = time.time()
            check_interval = 0.5
            opposite_side = None
            if closing_side:
                opposite_side = "LONG" if closing_side == "SHORT" else "SHORT"

            while time.time() - start_time < timeout_seconds:
                positions = await self.get_positions()

                # If no position exists, the close order worked
                if not positions:
                    logger.info(f"Lighter: Order verified - no open position")
                    return True

                # Check position state
                for pos in positions:
                    if pos.market == self.market:
                        logger.info(f"Lighter: Position check - size={pos.size}, side={pos.side}")

                        # CRITICAL: Side-aware verification
                        if closing_side is not None:
                            # If position FLIPPED to opposite side, the close order worked
                            # (and possibly overfilled, creating opposite position)
                            if pos.side == opposite_side:
                                logger.warning(f"Lighter: Position flipped from {closing_side} to {opposite_side} - close order filled (with flip)")
                                return True  # Don't retry - would make opposite position bigger

                            # If position is still the side we're closing but very small, consider it closed
                            if pos.side == closing_side and pos.size < expected_size * 0.1:
                                logger.info(f"Lighter: Position reduced to dust ({pos.size}) - considering closed")
                                return True

                            # If still same side with significant size, keep waiting
                            if pos.side == closing_side:
                                # Position still exists on same side, continue waiting
                                pass
                        break

                await asyncio.sleep(check_interval)

            logger.warning(f"Lighter: Order fill verification timed out after {timeout_seconds}s")
            return False

        except Exception as e:
            logger.warning(f"Lighter: Could not verify order fill: {e}")
            return False  # Assume not filled if we can't verify

    async def _fetch_order_book_orders(self):
        """Fetch order book orders, handling both sync and async SDK methods."""
        import inspect
        method = self.order_api.order_book_orders

        # Check if the method is async
        if inspect.iscoroutinefunction(method):
            # Truly async method
            return await method(market_id=self._orderbook_id, limit=1)
        else:
            # Sync method - run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: method(market_id=self._orderbook_id, limit=1)
            )

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
        # Don't close the shared signer_client - it's used by all markets
        # The shared client will be closed when the process exits
        self.connected = False
        logger.info(f"Lighter: Disconnected {self.market}")

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
            if self._orderbook_id is None:
                raise ValueError(f"No orderbook ID for market {self.market}")

            # Get top of book from order_book_orders with timeout
            logger.debug(f"Fetching BBO for market_id={self._orderbook_id}...")
            try:
                # Run the API call with a timeout
                # The SDK may use async or sync - handle both cases
                orders = await asyncio.wait_for(
                    self._fetch_order_book_orders(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.error(f"BBO fetch timed out for market {self.market}")
                raise ValueError(f"BBO fetch timed out for market {self.market}")

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
                        # Try multiple attribute names for market ID
                        pos_market_id = None
                        for attr in ['market_id', 'order_book_id', 'orderbook_id', 'market_index', 'orderBookId', 'marketId']:
                            pos_market_id = getattr(pos, attr, None)
                            if pos_market_id is not None:
                                break

                        # Get position size - Lighter uses 'position' field (always positive)
                        # and 'sign' field to indicate direction (1=LONG, -1=SHORT)
                        pos_size_str = getattr(pos, 'position', None)
                        pos_sign = getattr(pos, 'sign', None)

                        if pos_size_str is None:
                            continue

                        pos_size = float(pos_size_str)

                        # If size is 0, skip (no position)
                        if pos_size == 0:
                            continue

                        # CRITICAL: Apply sign to get actual direction
                        # sign=1 means LONG, sign=-1 means SHORT
                        # The position value is always positive, sign determines direction
                        if pos_sign is not None:
                            raw_size = pos_size * int(pos_sign)
                        else:
                            raw_size = pos_size  # Fallback if no sign field

                        # Check if this position belongs to our market
                        # Handle type mismatches (string vs int) by comparing both ways
                        market_matches = False
                        if pos_market_id is not None and self._orderbook_id is not None:
                            try:
                                market_matches = (int(pos_market_id) == int(self._orderbook_id))
                            except (ValueError, TypeError):
                                market_matches = (str(pos_market_id) == str(self._orderbook_id))
                        elif pos_market_id is None:
                            # If we can't find market_id, log warning but still include position
                            # This is safer than missing positions
                            logger.warning(f"Lighter: Position has no market_id attribute, assuming it matches {self.market}")
                            market_matches = True

                        if not market_matches:
                            logger.debug(f"Lighter: Skipping position - market_id {pos_market_id} doesn't match our {self._orderbook_id} ({self.market})")
                            continue

                        if raw_size != 0:
                            # Determine side from the sign of raw_size
                            # raw_size is already signed (position * sign)
                            side = "LONG" if raw_size > 0 else "SHORT"
                            size = abs(raw_size)

                            logger.debug(f"Lighter: Position detected - {self.market} {side} {size:.6f}")

                            # Get entry price with fallback attribute names
                            entry_price = None
                            for attr in ['entry_price', 'avg_entry_price', 'average_price', 'price']:
                                entry_price = getattr(pos, attr, None)
                                if entry_price is not None:
                                    break
                            entry_price = float(entry_price or 0)

                            # Get unrealized PnL with fallback attribute names
                            unrealized_pnl = None
                            for attr in ['unrealized_pnl', 'unrealizedPnl', 'pnl', 'unrealized_profit']:
                                unrealized_pnl = getattr(pos, attr, None)
                                if unrealized_pnl is not None:
                                    break
                            unrealized_pnl = float(unrealized_pnl or 0)

                            positions.append(Position(
                                market=self.market,
                                side=side,
                                size=size,  # Already absolute value from abs(raw_size)
                                entry_price=entry_price,
                                unrealized_pnl=unrealized_pnl,
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
        _retry_count: int = 0,
        slippage_pct: Optional[float] = None,  # Override slippage (for exits)
    ) -> Order:
        """Place an order with automatic retry on nonce errors."""
        global _last_order_time

        MAX_RETRIES = 3  # Increased retries for nonce errors
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

        # Use global lock to prevent nonce collisions across all clients
        async with _global_order_lock:
            try:
                # Enforce minimum spacing between orders
                current_time_ms = int(time.time() * 1000)
                time_since_last = current_time_ms - _last_order_time
                if time_since_last < _MIN_ORDER_SPACING_MS:
                    wait_time = (_MIN_ORDER_SPACING_MS - time_since_last) / 1000.0
                    logger.debug(f"Waiting {wait_time:.2f}s for order spacing")
                    await asyncio.sleep(wait_time)

                # Sync nonce before placing order (fresh fetch)
                if _retry_count == 0:
                    await self._sync_nonces(force_refresh=True)

                if self._orderbook_id is None:
                    raise ValueError(f"No orderbook ID for market {self.market}")

                is_bid = side == OrderSide.BUY

                # Get current price for market orders
                bbo = await self.get_bbo()
                current_price = bbo.ask_price if side == OrderSide.BUY else bbo.bid_price

                # Generate unique client order index using timestamp + random component
                client_order_idx = (int(time.time() * 1000) + hash(self.market)) % (2**32)

                # is_ask = True for SELL, False for BUY
                is_ask = side == OrderSide.SELL

                # Apply slippage tolerance to avg_execution_price
                # For BUY: allow paying up to X% more than current ask
                # For SELL: allow receiving up to X% less than current bid
                # Use provided slippage_pct (for exits) or default for entries (0.2% -> 0.4% on retry)
                if slippage_pct is not None:
                    slippage_tolerance = slippage_pct / 100  # Convert percentage to decimal
                else:
                    # Entries: 0.2% first try, 0.4% on retry
                    slippage_tolerance = 0.002 if _retry_count == 0 else 0.004
                if side == OrderSide.BUY:
                    price_with_slippage = current_price * (1 + slippage_tolerance)
                else:
                    price_with_slippage = current_price * (1 - slippage_tolerance)

                # Convert to integer format (Lighter uses smallest units)
                # Market-specific decimal precision - MUST match Lighter's expected precision
                # If wrong, orders will be 10x too large/small!
                symbol = self.market.split("-")[0] if "-" in self.market else self.market
                if symbol == "BTC":
                    size_decimals = 5  # 0.00001 BTC precision
                    price_decimals = 1
                elif symbol == "ETH":
                    size_decimals = 4  # 0.0001 ETH precision
                    price_decimals = 2
                elif symbol == "SOL":
                    size_decimals = 3  # 0.001 SOL precision (NOT 4!)
                    price_decimals = 3
                else:
                    size_decimals = 4  # Default for other markets
                    price_decimals = 2

                size_int = int(size * (10 ** size_decimals))
                price_int = int(price_with_slippage * (10 ** price_decimals))

                # Log the conversion for verification
                logger.debug(f"Order size conversion: {size:.6f} {symbol} -> {size_int} (decimals={size_decimals})")

                if order_type == OrderType.MARKET:
                    # Use create_market_order for market orders
                    # Parameters: market_index, client_order_index, base_amount, avg_execution_price, is_ask
                    logger.debug(f"Placing market order: market={self._orderbook_id}, size={size_int}, bbo_price={current_price:.2f}, max_price={price_with_slippage:.2f} ({slippage_tolerance*100:.1f}% slippage), is_ask={is_ask}")
                    result = await self.signer_client.create_market_order(
                        market_index=self._orderbook_id,
                        client_order_index=client_order_idx,
                        base_amount=size_int,
                        avg_execution_price=price_int,
                        is_ask=is_ask
                    )
                    # Result is tuple: (tx, response, error)
                    tx, response, error = result

                    # Update last order time for spacing
                    _last_order_time = int(time.time() * 1000)

                    if error:
                        error_str = str(error).lower()
                        # For nonce errors, sync and retry
                        if "21104" in error_str or "invalid nonce" in error_str:
                            if _retry_count < MAX_RETRIES:
                                retry_delay = 0.5 * (2 ** _retry_count)
                                logger.warning(f"Nonce error on {self.market}, syncing and retrying in {retry_delay}s (attempt {_retry_count + 1}/{MAX_RETRIES})...")
                                await self._sync_nonces(force_refresh=True)
                                await asyncio.sleep(retry_delay)
                                return await self.place_order(side, size, order_type, price, client_id, _retry_count + 1, slippage_pct)
                        # For other errors on entries, retry with more slippage
                        elif slippage_pct is None and _retry_count < MAX_RETRIES:
                            retry_delay = 0.5
                            logger.warning(f"Order error: {error}, retrying with more slippage...")
                            await asyncio.sleep(retry_delay)
                            return await self.place_order(side, size, order_type, price, client_id, _retry_count + 1, slippage_pct)
                        logger.error(f"Order failed: {error}")
                        raise ValueError(f"Order failed: {error}")
                    logger.debug(f"Order response: {response}")
                else:
                    # Use create_order for limit orders
                    if price is None:
                        raise ValueError("Price required for limit orders")

                    limit_price_int = int(price * (10 ** price_decimals))
                    logger.debug(f"Placing limit order: market={self._orderbook_id}, size={size_int}, price={limit_price_int}, is_ask={is_ask}")
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

                    # Update last order time for spacing
                    _last_order_time = int(time.time() * 1000)

                    if error:
                        error_str = str(error)
                        # Check for invalid nonce error (code 21104)
                        if "21104" in error_str or "invalid nonce" in error_str.lower():
                            if _retry_count < MAX_RETRIES:
                                retry_delay = 0.5 * (2 ** _retry_count)
                                logger.warning(f"Invalid nonce error, retrying in {retry_delay}s (attempt {_retry_count + 1}/{MAX_RETRIES})...")
                                await self._sync_nonces(force_refresh=True)
                                await asyncio.sleep(retry_delay)
                                return await self.place_order(side, size, order_type, price, client_id, _retry_count + 1)
                        logger.error(f"Order failed: {error}")
                        raise ValueError(f"Order failed: {error}")
                    logger.debug(f"Order response: {response}")

                logger.debug(f"Lighter order placed: {side.value} {size} @ {price or 'MARKET'}")

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
            except ValueError:
                # Re-raise ValueError (our own errors) without wrapping
                raise
            except Exception as e:
                error_str = str(e)
                # Check for invalid nonce error in unexpected exceptions too
                if ("21104" in error_str or "invalid nonce" in error_str.lower()) and _retry_count < MAX_RETRIES:
                    retry_delay = 0.5 * (2 ** _retry_count)
                    logger.warning(f"Invalid nonce error (exception), retrying in {retry_delay}s (attempt {_retry_count + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(retry_delay)
                    await self._sync_nonces(force_refresh=True)
                    return await self.place_order(side, size, order_type, price, client_id, _retry_count + 1)
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
