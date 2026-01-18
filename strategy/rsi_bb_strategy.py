"""
RSI + Bollinger Band Strategy with Trailing Stop

Entry: SHORT when price > upper BB AND RSI > threshold
Exit: Trailing stop (0.2% after 0.1% profit) OR lower BB OR stop loss

Optimized parameters from backtest:
- BTC: RSI > 75, SL 0.3%, Trail 0.2%@0.1%
- ETH: RSI > 60, SL 0.3%, Trail 0.2%@0.1%
- SOL: RSI > 65, SL 0.3%, Trail 0.2%@0.1%

90-day backtest results:
- BTC: $1,013 PnL, 56% WR, 41% MaxDD
- ETH: $1,371 PnL, 52% WR, 44% MaxDD
- SOL: $1,792 PnL, 52% WR, 51% MaxDD
"""

import asyncio
import logging
import csv
import os
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Trade history CSV path
TRADE_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(TRADE_HISTORY_DIR, exist_ok=True)


@dataclass
class RSIBBTrade:
    """Active short trade."""
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    best_price: float  # Lowest price seen (for trailing)
    trailing_active: bool = False


@dataclass
class RSIBBConfig:
    """Configuration for RSI+BB strategy."""
    # Bollinger Band settings
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI settings
    rsi_period: int = 7

    # Exit settings
    stop_loss_pct: float = 0.30
    trailing_stop_pct: float = 0.20
    trailing_activation_pct: float = 0.10  # Activate trailing after 0.1% profit

    # Position sizing
    margin_per_trade: float = 100.0
    leverage: float = 20.0


class RSIBBStrategy:
    """
    RSI + Bollinger Band Strategy with Trailing Stop.

    Entry: SHORT when price > upper BB AND RSI > threshold
    Exit: Trailing stop OR lower BB OR stop loss
    """

    def __init__(
        self,
        exchange_clients: Dict[str, any],
        config: RSIBBConfig = None,
        rsi_entry_overrides: Dict[str, int] = None,
    ):
        self.clients = exchange_clients
        self.config = config or RSIBBConfig()

        # Per-asset RSI entry thresholds (optimized)
        self.rsi_entry = rsi_entry_overrides or {
            "BTC": 75,
            "ETH": 60,
            "SOL": 65,
        }

        # Price history for indicators
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=max(self.config.bb_period, self.config.rsi_period) + 10)
            for asset in exchange_clients.keys()
        }

        # Active trades
        self.active_trades: Dict[str, Optional[RSIBBTrade]] = {
            asset: None for asset in exchange_clients.keys()
        }

        # Stats
        self.stats = {
            asset: {
                "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
                "trailing_exits": 0, "lower_bb_exits": 0, "sl_exits": 0
            }
            for asset in exchange_clients.keys()
        }

        self.running = False

        # Trade history CSV
        self.trade_history_file = os.path.join(TRADE_HISTORY_DIR, "rsi_bb_trades.csv")
        self._init_trade_history()

    def _init_trade_history(self):
        """Initialize trade history CSV."""
        if not os.path.exists(self.trade_history_file):
            with open(self.trade_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'entry_time', 'exit_time', 'entry_price',
                    'exit_price', 'best_price', 'size', 'pnl_pct', 'pnl_usd',
                    'exit_reason', 'sl_price', 'hold_seconds', 'entry_rsi'
                ])

    def _log_trade(self, asset: str, trade: RSIBBTrade, exit_price: float,
                   pnl_pct: float, pnl_usd: float, exit_reason: str, entry_rsi: float):
        """Log completed trade to CSV."""
        try:
            hold_seconds = (datetime.now() - trade.entry_time).total_seconds()
            with open(self.trade_history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    asset,
                    trade.entry_time.isoformat(),
                    datetime.now().isoformat(),
                    trade.entry_price,
                    exit_price,
                    trade.best_price,
                    trade.size,
                    round(pnl_pct, 4),
                    round(pnl_usd, 2),
                    exit_reason,
                    trade.stop_loss_price,
                    round(hold_seconds, 1),
                    round(entry_rsi, 1)
                ])
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def calculate_rsi(self, prices: list) -> float:
        """Calculate RSI."""
        if len(prices) < self.config.rsi_period + 1:
            return 50.0  # Neutral

        prices_arr = np.array(prices)
        deltas = np.diff(prices_arr)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        recent_gains = gains[-self.config.rsi_period:]
        recent_losses = losses[-self.config.rsi_period:]

        avg_gain = np.mean(recent_gains)
        avg_loss = np.mean(recent_losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bb(self, prices: list) -> tuple:
        """Calculate Bollinger Bands."""
        if len(prices) < self.config.bb_period:
            return None, None, None

        recent = list(prices)[-self.config.bb_period:]
        sma = np.mean(recent)
        std = np.std(recent)

        upper = sma + self.config.bb_std * std
        lower = sma - self.config.bb_std * std

        return lower, sma, upper

    async def check_entry(self, asset: str, current_price: float) -> Optional[float]:
        """Check if we should enter. Returns RSI if entry, None otherwise."""
        if self.active_trades[asset] is not None:
            return None

        prices = list(self.price_history[asset])
        if len(prices) < max(self.config.bb_period, self.config.rsi_period + 1):
            return None

        _, _, upper_bb = self.calculate_bb(prices)
        rsi = self.calculate_rsi(prices)

        rsi_threshold = self.rsi_entry.get(asset, 70)

        if upper_bb and current_price > upper_bb and rsi > rsi_threshold:
            logger.info(f"{asset}: Price {current_price:.2f} > Upper BB {upper_bb:.2f}, RSI {rsi:.1f} > {rsi_threshold} - SHORT SIGNAL")
            return rsi

        return None

    async def check_exit(self, asset: str, current_price: float) -> Optional[str]:
        """Check exit conditions. Returns exit reason or None."""
        trade = self.active_trades[asset]
        if trade is None:
            return None

        # Update best price (lowest for short = most profit)
        if current_price < trade.best_price:
            trade.best_price = current_price

        # Calculate current PnL
        pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100

        # Check if trailing should activate
        if not trade.trailing_active and pnl_pct >= self.config.trailing_activation_pct:
            trade.trailing_active = True
            logger.info(f"{asset}: Trailing stop activated at {pnl_pct:.2f}% profit")

        # 1. Stop Loss
        if current_price >= trade.stop_loss_price:
            return "stop_loss"

        # 2. Trailing Stop (if active)
        if trade.trailing_active:
            trail_price = trade.best_price * (1 + self.config.trailing_stop_pct / 100)
            if current_price >= trail_price:
                return "trailing_stop"

        # 3. Lower BB exit
        prices = list(self.price_history[asset])
        if len(prices) >= self.config.bb_period:
            lower_bb, _, _ = self.calculate_bb(prices)
            if lower_bb and current_price < lower_bb:
                return "lower_bb"

        return None

    async def enter_short(self, asset: str, current_price: float, entry_rsi: float):
        """Enter a short position."""
        client = self.clients[asset]

        notional = self.config.margin_per_trade * self.config.leverage
        size = notional / current_price

        sl_price = current_price * (1 + self.config.stop_loss_pct / 100)

        try:
            from exchange.base import OrderSide, OrderType

            order = await client.place_order(
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            self.active_trades[asset] = RSIBBTrade(
                asset=asset,
                entry_time=datetime.now(),
                entry_price=fill_price,
                size=size,
                stop_loss_price=sl_price,
                best_price=fill_price,
                trailing_active=False,
            )

            # Store entry RSI for logging
            self.active_trades[asset]._entry_rsi = entry_rsi

            logger.info(
                f"{asset}: ENTERED SHORT @ {fill_price:.2f}, RSI: {entry_rsi:.1f}, "
                f"Size: ${notional:.0f} ({size:.4f}), SL: {sl_price:.2f}, "
                f"Trail: {self.config.trailing_stop_pct}%@{self.config.trailing_activation_pct}%"
            )

        except Exception as e:
            logger.error(f"{asset}: Failed to enter short: {e}")

    async def exit_short(self, asset: str, current_price: float, reason: str):
        """Exit a short position."""
        trade = self.active_trades[asset]
        if trade is None:
            return

        client = self.clients[asset]

        try:
            from exchange.base import OrderSide, OrderType

            # Use place_order_until_filled for critical exit orders
            # This retries with increasing slippage until the order fills
            if hasattr(client, 'place_order_until_filled'):
                order = await client.place_order_until_filled(
                    side=OrderSide.BUY,
                    size=trade.size,
                    max_attempts=5,
                )
            else:
                # Fallback to regular place_order for clients without the method
                order = await client.place_order(
                    side=OrderSide.BUY,
                    size=trade.size,
                    order_type=OrderType.MARKET,
                )

            fill_price = order.avg_fill_price or current_price

            pnl_pct = (trade.entry_price - fill_price) / trade.entry_price * 100
            notional = self.config.margin_per_trade * self.config.leverage
            pnl_usd = pnl_pct / 100 * notional

            # Update stats
            self.stats[asset]["trades"] += 1
            self.stats[asset]["pnl"] += pnl_usd
            if pnl_pct > 0:
                self.stats[asset]["wins"] += 1
            else:
                self.stats[asset]["losses"] += 1

            if reason == "trailing_stop":
                self.stats[asset]["trailing_exits"] += 1
            elif reason == "lower_bb":
                self.stats[asset]["lower_bb_exits"] += 1
            elif reason == "stop_loss":
                self.stats[asset]["sl_exits"] += 1

            # Log trade
            entry_rsi = getattr(trade, '_entry_rsi', 0)
            self._log_trade(asset, trade, fill_price, pnl_pct, pnl_usd, reason, entry_rsi)

            win_rate = self.stats[asset]["wins"] / self.stats[asset]["trades"] * 100

            max_profit_pct = (trade.entry_price - trade.best_price) / trade.entry_price * 100

            logger.info(
                f"{asset}: EXITED SHORT @ {fill_price:.2f} ({reason}), "
                f"PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f}), MaxProfit: {max_profit_pct:.2f}%"
            )
            logger.info(
                f"{asset}: Stats - Trades: {self.stats[asset]['trades']}, WR: {win_rate:.1f}%, "
                f"Trail: {self.stats[asset]['trailing_exits']}, BB: {self.stats[asset]['lower_bb_exits']}, "
                f"SL: {self.stats[asset]['sl_exits']}, PnL: ${self.stats[asset]['pnl']:.2f}"
            )

            self.active_trades[asset] = None

        except Exception as e:
            logger.error(f"{asset}: Failed to exit short: {e}")

    async def update(self, asset: str, current_price: float):
        """Update strategy with new price."""
        self.price_history[asset].append(current_price)

        # Check exit first
        exit_reason = await self.check_exit(asset, current_price)
        if exit_reason:
            await self.exit_short(asset, current_price, exit_reason)
            return

        # Check entry
        entry_rsi = await self.check_entry(asset, current_price)
        if entry_rsi is not None:
            await self.enter_short(asset, current_price, entry_rsi)

    async def run_asset(self, asset: str):
        """Run strategy loop for a single asset."""
        client = self.clients[asset]

        logger.info(f"{asset}: Starting RSI+BB strategy (RSI entry > {self.rsi_entry.get(asset, 70)})")
        log_counter = 0
        first_bbo = True

        while self.running:
            try:
                bbo = await client.get_bbo()
                mid_price = (bbo.bid_price + bbo.ask_price) / 2

                if first_bbo:
                    logger.info(f"{asset}: First BBO received - Bid: {bbo.bid_price:.2f}, Ask: {bbo.ask_price:.2f}")
                    first_bbo = False

                trade = self.active_trades[asset]
                if trade:
                    pass  # Position logging handled below
                else:
                    # Log status periodically when not in a trade
                    log_counter += 1
                    if log_counter >= 4:  # Log every ~2 minutes (4 * 30s)
                        prices = list(self.price_history[asset])
                        price_count = len(prices)
                        if price_count >= max(self.config.bb_period, self.config.rsi_period + 1):
                            _, _, upper_bb = self.calculate_bb(prices)
                            rsi = self.calculate_rsi(prices)
                            rsi_thresh = self.rsi_entry.get(asset, 70)
                            logger.info(
                                f"{asset}: Watching - Price: {mid_price:.2f}, Upper BB: {upper_bb:.2f}, "
                                f"RSI: {rsi:.1f}/{rsi_thresh} (need price > BB AND RSI > {rsi_thresh})"
                            )
                        else:
                            logger.info(f"{asset}: Building history - {price_count}/{self.config.bb_period} prices collected")
                        log_counter = 0

                if trade:
                    log_counter += 1
                    if log_counter >= 12:  # Log every ~60s
                        pnl_pct = (trade.entry_price - bbo.bid_price) / trade.entry_price * 100
                        pnl_usd = pnl_pct / 100 * self.config.margin_per_trade * self.config.leverage
                        trail_status = "ACTIVE" if trade.trailing_active else "inactive"
                        trail_price = trade.best_price * (1 + self.config.trailing_stop_pct / 100) if trade.trailing_active else 0

                        prices = list(self.price_history[asset])
                        lower_bb, _, _ = self.calculate_bb(prices) if len(prices) >= self.config.bb_period else (None, None, None)
                        lower_str = f"{lower_bb:.2f}" if lower_bb else "N/A"

                        logger.info(
                            f"{asset}: Bid: {bbo.bid_price:.2f}, Best: {trade.best_price:.2f}, "
                            f"Trail({trail_status}): {trail_price:.2f}, LowerBB: {lower_str}, "
                            f"SL: {trade.stop_loss_price:.2f}, PnL: ${pnl_usd:+.2f}"
                        )
                        log_counter = 0
                else:
                    log_counter = 0

                # Use bid for exit checks when in trade
                await self.update(asset, bbo.bid_price if trade else mid_price)

                # Fast checks when in trade, slower for entries
                if self.active_trades[asset]:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"{asset}: Error in strategy loop: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Run strategy for all assets."""
        self.running = True

        for asset, client in self.clients.items():
            if not client.connected:
                await client.connect()

        logger.info("=" * 70)
        logger.info("RSI+BB STRATEGY - Starting")
        logger.info("=" * 70)
        logger.info(f"Assets: {list(self.clients.keys())}")
        logger.info(f"Entry: Price > Upper BB({self.config.bb_period}) AND RSI({self.config.rsi_period}) > threshold")
        logger.info(f"RSI thresholds: {self.rsi_entry}")
        logger.info(f"Exit: Trail {self.config.trailing_stop_pct}%@{self.config.trailing_activation_pct}% OR Lower BB OR SL {self.config.stop_loss_pct}%")
        logger.info(f"Position: ${self.config.margin_per_trade} x {self.config.leverage}x = ${self.config.margin_per_trade * self.config.leverage}")
        logger.info("=" * 70)

        tasks = [self.run_asset(asset) for asset in self.clients.keys()]
        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop strategy and close positions."""
        self.running = False

        for asset, trade in self.active_trades.items():
            if trade is not None:
                try:
                    bbo = await self.clients[asset].get_bbo()
                    mid_price = (bbo.bid_price + bbo.ask_price) / 2
                    await self.exit_short(asset, mid_price, "shutdown")
                except Exception as e:
                    logger.error(f"{asset}: Failed to close on shutdown: {e}")

        logger.info("=" * 70)
        logger.info("FINAL STATS")
        logger.info("=" * 70)
        total_trades = 0
        total_wins = 0
        total_pnl = 0

        for asset, stats in self.stats.items():
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                logger.info(
                    f"{asset}: {stats['trades']} trades, WR: {wr:.1f}%, "
                    f"Trail: {stats['trailing_exits']}, BB: {stats['lower_bb_exits']}, "
                    f"SL: {stats['sl_exits']}, PnL: ${stats['pnl']:.2f}"
                )
                total_trades += stats["trades"]
                total_wins += stats["wins"]
                total_pnl += stats["pnl"]

        if total_trades > 0:
            logger.info(f"TOTAL: {total_trades} trades, WR: {total_wins/total_trades*100:.1f}%, PnL: ${total_pnl:.2f}")
