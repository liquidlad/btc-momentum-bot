"""
Bollinger Band Band-to-Band Strategy (Mean Reversion)

Entry: SHORT when price closes above upper Bollinger Band
Exit: When price closes below lower Bollinger Band (or SL hit)

Higher risk/reward than fixed TP - holds until full mean reversion.
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
class BB2BTrade:
    """Active short trade."""
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    # No fixed TP - exit on lower band


@dataclass
class BB2BConfig:
    """Configuration for BB Band-to-Band strategy."""
    bb_period: int = 20
    bb_std: float = 2.0
    stop_loss_pct: float = 0.30  # Optimal from backtest: BTC 0.2%, ETH 0.4%, SOL 0.3%
    margin_per_trade: float = 25.0  # $25 margin per asset (25% of main bot)
    leverage: float = 20.0  # 20x leverage = $500 notional per trade


class BB2BStrategy:
    """
    Bollinger Band Band-to-Band Mean Reversion Strategy.

    Shorts when price is overbought (above upper BB).
    Exits when price reaches lower BB (full mean reversion) or SL.
    """

    def __init__(
        self,
        exchange_clients: Dict[str, any],
        config: BB2BConfig = None,
        asset_sl_overrides: Dict[str, float] = None,  # Per-asset SL optimization
    ):
        self.clients = exchange_clients
        self.config = config or BB2BConfig()

        # Per-asset SL overrides based on backtest optimization
        self.asset_sl = asset_sl_overrides or {
            "BTC": 0.20,  # Optimal for BTC
            "ETH": 0.40,  # Optimal for ETH
            "SOL": 0.30,  # Optimal for SOL
        }

        # Price history for BB calculation (need bb_period + buffer)
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=self.config.bb_period + 10)
            for asset in exchange_clients.keys()
        }

        # Active trades per asset
        self.active_trades: Dict[str, Optional[BB2BTrade]] = {
            asset: None for asset in exchange_clients.keys()
        }

        # Stats
        self.stats = {
            asset: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "band_exits": 0, "sl_exits": 0}
            for asset in exchange_clients.keys()
        }

        self.running = False

        # Trade history CSV file
        self.trade_history_file = os.path.join(TRADE_HISTORY_DIR, "bandtoband_trades.csv")
        self._init_trade_history()

    def _init_trade_history(self):
        """Initialize trade history CSV if it doesn't exist."""
        if not os.path.exists(self.trade_history_file):
            with open(self.trade_history_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'entry_time', 'exit_time', 'entry_price',
                    'exit_price', 'size', 'pnl_pct', 'pnl_usd', 'exit_reason',
                    'sl_price', 'hold_seconds'
                ])

    def _log_trade(self, asset: str, trade, exit_price: float, pnl_pct: float, pnl_usd: float, exit_reason: str):
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
                    trade.size,
                    round(pnl_pct, 4),
                    round(pnl_usd, 2),
                    exit_reason,
                    trade.stop_loss_price,
                    round(hold_seconds, 1)
                ])
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

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

    async def check_entry(self, asset: str, current_price: float) -> bool:
        """Check if we should enter a short position."""
        if self.active_trades[asset] is not None:
            return False  # Already in trade

        prices = list(self.price_history[asset])
        if len(prices) < self.config.bb_period:
            return False

        _, _, upper_bb = self.calculate_bb(prices)

        if upper_bb and current_price > upper_bb:
            logger.info(f"{asset}: Price {current_price:.2f} > Upper BB {upper_bb:.2f} - SHORT SIGNAL")
            return True

        return False

    async def check_exit(self, asset: str, current_price: float) -> Optional[str]:
        """Check if we should exit position. Returns exit reason or None."""
        trade = self.active_trades[asset]
        if trade is None:
            return None

        # Check SL first (price went up for short)
        if current_price >= trade.stop_loss_price:
            return "stop_loss"

        # Check if price is below lower BB (mean reversion complete)
        prices = list(self.price_history[asset])
        if len(prices) >= self.config.bb_period:
            lower_bb, _, _ = self.calculate_bb(prices)
            if lower_bb and current_price < lower_bb:
                return "lower_band"

        return None

    async def enter_short(self, asset: str, current_price: float):
        """Enter a short position."""
        client = self.clients[asset]

        # Calculate position size: margin × leverage = notional
        notional = self.config.margin_per_trade * self.config.leverage
        size = notional / current_price

        # Get asset-specific SL
        sl_pct = self.asset_sl.get(asset, self.config.stop_loss_pct)
        sl_price = current_price * (1 + sl_pct / 100)

        try:
            from exchange.base import OrderSide, OrderType

            order = await client.place_order(
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            self.active_trades[asset] = BB2BTrade(
                asset=asset,
                entry_time=datetime.now(),
                entry_price=fill_price,
                size=size,
                stop_loss_price=sl_price,
            )

            logger.info(f"{asset}: ENTERED SHORT @ {fill_price:.2f}, Size: ${notional:.0f} ({size:.4f}), SL: {sl_price:.2f} ({sl_pct}%), Exit: Lower BB")

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

            order = await client.place_order(
                side=OrderSide.BUY,  # Buy to close short
                size=trade.size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            # Calculate PnL (for short: profit = entry - exit)
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

            if reason == "lower_band":
                self.stats[asset]["band_exits"] += 1
            elif reason == "stop_loss":
                self.stats[asset]["sl_exits"] += 1

            # Log trade to CSV
            self._log_trade(asset, trade, fill_price, pnl_pct, pnl_usd, reason)

            win_rate = self.stats[asset]["wins"] / self.stats[asset]["trades"] * 100 if self.stats[asset]["trades"] > 0 else 0

            logger.info(f"{asset}: EXITED SHORT @ {fill_price:.2f} ({reason}), PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            logger.info(f"{asset}: Stats - Trades: {self.stats[asset]['trades']}, WR: {win_rate:.1f}%, Band: {self.stats[asset]['band_exits']}, SL: {self.stats[asset]['sl_exits']}, Total PnL: ${self.stats[asset]['pnl']:.2f}")

            self.active_trades[asset] = None

        except Exception as e:
            logger.error(f"{asset}: Failed to exit short: {e}")

    async def update(self, asset: str, current_price: float):
        """Update strategy with new price."""
        # Add to price history
        self.price_history[asset].append(current_price)

        # Check for exit first
        exit_reason = await self.check_exit(asset, current_price)
        if exit_reason:
            await self.exit_short(asset, current_price, exit_reason)
            return

        # Check for entry
        if await self.check_entry(asset, current_price):
            await self.enter_short(asset, current_price)

    async def run_asset(self, asset: str):
        """Run strategy loop for a single asset."""
        client = self.clients[asset]

        logger.info(f"{asset}: Starting BB Band-to-Band strategy")
        log_counter = 0

        while self.running:
            try:
                bbo = await client.get_bbo()
                mid_price = (bbo.bid_price + bbo.ask_price) / 2

                # Log price if we have an active trade
                trade = self.active_trades[asset]
                if trade:
                    log_counter += 1
                    if log_counter >= 12:  # Log every ~60 seconds
                        prices = list(self.price_history[asset])
                        lower_bb, _, _ = self.calculate_bb(prices) if len(prices) >= self.config.bb_period else (None, None, None)
                        pnl_pct = (trade.entry_price - bbo.bid_price) / trade.entry_price * 100
                        pnl_usd = pnl_pct / 100 * self.config.margin_per_trade * self.config.leverage
                        lower_str = f"{lower_bb:.2f}" if lower_bb else "N/A"
                        logger.info(f"{asset}: Bid: {bbo.bid_price:.2f}, LowerBB: {lower_str}, SL: {trade.stop_loss_price:.2f}, PnL: ${pnl_usd:+.2f}")
                        log_counter = 0
                else:
                    log_counter = 0

                # Use bid price for exit checks when in trade
                await self.update(asset, bbo.bid_price if self.active_trades[asset] else mid_price)

                # When in a trade: check every 5 seconds
                # When not in trade: check every 30 seconds for entry signals
                if self.active_trades[asset]:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"{asset}: Error in strategy loop: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Run strategy for all assets concurrently."""
        self.running = True

        # Connect all clients
        for asset, client in self.clients.items():
            if not client.connected:
                await client.connect()

        logger.info("="*60)
        logger.info("BB BAND-TO-BAND STRATEGY - Starting")
        logger.info(f"Assets: {list(self.clients.keys())}")
        logger.info(f"Config: BB({self.config.bb_period}), Exit: Lower BB")
        logger.info(f"Stop Loss by asset: {self.asset_sl}")
        logger.info(f"Position size: ${self.config.margin_per_trade} x {self.config.leverage}x = ${self.config.margin_per_trade * self.config.leverage}")
        logger.info("="*60)

        # Run all assets concurrently
        tasks = [self.run_asset(asset) for asset in self.clients.keys()]
        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop the strategy and close all positions."""
        self.running = False

        # Close any open positions
        for asset, trade in self.active_trades.items():
            if trade is not None:
                try:
                    bbo = await self.clients[asset].get_bbo()
                    mid_price = (bbo.bid_price + bbo.ask_price) / 2
                    await self.exit_short(asset, mid_price, "shutdown")
                except Exception as e:
                    logger.error(f"{asset}: Failed to close position on shutdown: {e}")

        # Print final stats
        logger.info("="*60)
        logger.info("FINAL STATS")
        logger.info("="*60)
        total_trades = 0
        total_wins = 0
        total_pnl = 0
        for asset, stats in self.stats.items():
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                logger.info(f"{asset}: {stats['trades']} trades, WR: {wr:.1f}%, Band: {stats['band_exits']}, SL: {stats['sl_exits']}, PnL: ${stats['pnl']:.2f}")
                total_trades += stats["trades"]
                total_wins += stats["wins"]
                total_pnl += stats["pnl"]

        if total_trades > 0:
            logger.info(f"TOTAL: {total_trades} trades, WR: {total_wins/total_trades*100:.1f}%, PnL: ${total_pnl:.2f}")
