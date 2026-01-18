#!/usr/bin/env python3
"""
Para BB RSI Bot - RSI + Bollinger Band Strategy on Paradex

Entry: SHORT when price > upper BB AND RSI > threshold
Exit: Trailing stop (0.2% after 0.1% profit) OR lower BB OR stop loss 0.3%

Optimized parameters:
- BTC: RSI > 75
- ETH: RSI > 60
- SOL: RSI > 65

90-day backtest results ($20 margin x 20x = $400 per trade):
- BTC: ~$203 PnL, 56% WR (scaled from $100 margin results)
- ETH: ~$274 PnL, 52% WR
- SOL: ~$358 PnL, 52% WR

Note: Paradex has 0.019% API taker fee (~$0.15 per $400 trade round-trip)
Expected fee drag: ~13% of profits

Usage:
    python para_bb_rsi.py              # Paper trading (all assets)
    python para_bb_rsi.py --live       # Live trading
    python para_bb_rsi.py --assets BTC # Single asset
"""

import argparse
import asyncio
import logging
import signal
import sys
import os
import csv
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass
from collections import deque
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Ensure logs directory exists
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, f'para_bb_rsi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

# Trade history CSV path
TRADE_HISTORY_FILE = os.path.join(LOGS_DIR, "para_bb_rsi_trades.csv")


@dataclass
class BBRSITrade:
    """Active short trade."""
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    best_price: float  # Lowest price seen (for trailing)
    trailing_active: bool = False
    entry_rsi: float = 0.0


@dataclass
class BBRSIConfig:
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
    margin_per_trade: float = 20.0  # $20 per trade
    leverage: float = 20.0          # 20x leverage = $400 notional


class ParaBBRSIBot:
    """
    RSI + Bollinger Band Strategy on Paradex.

    Entry: SHORT when price > upper BB AND RSI > threshold
    Exit: Trailing stop OR lower BB OR stop loss
    """

    def __init__(
        self,
        assets: list,
        config: BBRSIConfig,
        paper_mode: bool = True,
        rsi_entry_overrides: Dict[str, int] = None,
    ):
        self.assets = assets
        self.config = config
        self.paper_mode = paper_mode
        self.running = False

        # Per-asset RSI entry thresholds (optimized)
        self.rsi_entry = rsi_entry_overrides or {
            "BTC": 75,
            "ETH": 60,
            "SOL": 65,
        }

        # Exchange clients (one per asset)
        self.clients: Dict[str, any] = {}

        # Price history for indicators
        self.price_history: Dict[str, deque] = {
            asset: deque(maxlen=max(config.bb_period, config.rsi_period) + 10)
            for asset in assets
        }

        # Active trades
        self.active_trades: Dict[str, Optional[BBRSITrade]] = {
            asset: None for asset in assets
        }

        # Stats
        self.stats = {
            asset: {
                "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0,
                "fees": 0.0,
                "trailing_exits": 0, "lower_bb_exits": 0, "sl_exits": 0
            }
            for asset in assets
        }

        # Initialize trade history CSV
        self._init_trade_history()

    def _init_trade_history(self):
        """Initialize trade history CSV."""
        if not os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'asset', 'entry_time', 'exit_time', 'entry_price',
                    'exit_price', 'best_price', 'size', 'pnl_pct', 'pnl_usd', 'fees',
                    'net_pnl', 'exit_reason', 'sl_price', 'hold_seconds', 'entry_rsi'
                ])

    def _log_trade(self, asset: str, trade: BBRSITrade, exit_price: float,
                   pnl_pct: float, pnl_usd: float, fees: float, exit_reason: str):
        """Log completed trade to CSV."""
        try:
            hold_seconds = (datetime.now() - trade.entry_time).total_seconds()
            net_pnl = pnl_usd - fees
            with open(TRADE_HISTORY_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    asset,
                    trade.entry_time.isoformat(),
                    datetime.now().isoformat(),
                    round(trade.entry_price, 2),
                    round(exit_price, 2),
                    round(trade.best_price, 2),
                    round(trade.size, 6),
                    round(pnl_pct, 4),
                    round(pnl_usd, 2),
                    round(fees, 4),
                    round(net_pnl, 2),
                    exit_reason,
                    round(trade.stop_loss_price, 2),
                    round(hold_seconds, 1),
                    round(trade.entry_rsi, 1)
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

    async def setup(self) -> bool:
        """Initialize exchange clients."""
        try:
            from exchange.paradex_client import ParadexClient

            for asset in self.assets:
                market = f"{asset}-USD-PERP"
                client = ParadexClient(
                    market=market,
                    paper_mode=self.paper_mode,
                )
                # Set initial paper balance
                client._paper_balance = 100.0 / len(self.assets)  # Split $100 across assets

                connected = await client.connect()
                if not connected:
                    logger.error(f"Failed to connect client for {asset}")
                    return False

                self.clients[asset] = client
                logger.info(f"Connected to Paradex for {asset} ({market})")

            return True

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

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

            self.active_trades[asset] = BBRSITrade(
                asset=asset,
                entry_time=datetime.now(),
                entry_price=fill_price,
                size=size,
                stop_loss_price=sl_price,
                best_price=fill_price,
                trailing_active=False,
                entry_rsi=entry_rsi,
            )

            logger.info(
                f"{asset}: ENTERED SHORT @ {fill_price:.2f}, RSI: {entry_rsi:.1f}, "
                f"Size: ${notional:.0f} ({size:.6f}), SL: {sl_price:.2f}, "
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

            order = await client.place_order(
                side=OrderSide.BUY,
                size=trade.size,
                order_type=OrderType.MARKET,
            )

            fill_price = order.avg_fill_price or current_price

            # Calculate P&L
            pnl_pct = (trade.entry_price - fill_price) / trade.entry_price * 100
            notional = self.config.margin_per_trade * self.config.leverage
            pnl_usd = pnl_pct / 100 * notional

            # Calculate fees (0.019% per side = 0.038% round trip)
            fee_rate = 0.00019  # 0.019%
            entry_notional = trade.entry_price * trade.size
            exit_notional = fill_price * trade.size
            fees = (entry_notional + exit_notional) * fee_rate

            net_pnl = pnl_usd - fees

            # Update stats
            self.stats[asset]["trades"] += 1
            self.stats[asset]["pnl"] += net_pnl
            self.stats[asset]["fees"] += fees
            if net_pnl > 0:
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
            self._log_trade(asset, trade, fill_price, pnl_pct, pnl_usd, fees, reason)

            win_rate = self.stats[asset]["wins"] / self.stats[asset]["trades"] * 100 if self.stats[asset]["trades"] > 0 else 0

            max_profit_pct = (trade.entry_price - trade.best_price) / trade.entry_price * 100

            logger.info(
                f"{asset}: EXITED SHORT @ {fill_price:.2f} ({reason}), "
                f"Gross: {pnl_pct:+.2f}% (${pnl_usd:+.2f}), Fees: ${fees:.2f}, Net: ${net_pnl:+.2f}"
            )
            logger.info(
                f"{asset}: Stats - Trades: {self.stats[asset]['trades']}, WR: {win_rate:.1f}%, "
                f"Trail: {self.stats[asset]['trailing_exits']}, BB: {self.stats[asset]['lower_bb_exits']}, "
                f"SL: {self.stats[asset]['sl_exits']}, Net PnL: ${self.stats[asset]['pnl']:.2f}"
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

    async def bootstrap_prices(self, asset: str):
        """Fetch historical prices from Binance to bootstrap indicators."""
        import requests

        symbol_map = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
        symbol = symbol_map.get(asset, f"{asset}USDT")

        try:
            logger.info(f"{asset}: Fetching historical prices for indicator bootstrap...")
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=30"
            response = requests.get(url, timeout=10)
            data = response.json()

            if not data or not isinstance(data, list):
                logger.warning(f"{asset}: Could not fetch historical prices")
                return

            # Feed prices to history (skip last one as it's incomplete)
            for kline in data[:-1]:
                close_price = float(kline[4])
                self.price_history[asset].append(close_price)

            logger.info(f"{asset}: Bootstrapped with {len(data)-1} historical prices - indicators ready!")

        except Exception as e:
            logger.warning(f"{asset}: Could not bootstrap prices: {e}")

    async def run_asset(self, asset: str):
        """Run strategy loop for a single asset."""
        client = self.clients[asset]

        # Bootstrap prices first
        await self.bootstrap_prices(asset)

        logger.info(f"{asset}: Starting RSI+BB strategy (RSI entry > {self.rsi_entry.get(asset, 70)})")
        log_counter = 0

        while self.running:
            try:
                bbo = await client.get_bbo()
                mid_price = (bbo.bid_price + bbo.ask_price) / 2

                # Update paper BBO if in paper mode
                if self.paper_mode and hasattr(client, 'update_paper_bbo'):
                    client.update_paper_bbo(bbo)

                trade = self.active_trades[asset]
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

        notional = self.config.margin_per_trade * self.config.leverage
        fee_per_trade = notional * 0.00038  # 0.019% x 2 sides

        logger.info("=" * 70)
        logger.info("PARA BB RSI BOT - Paradex RSI+Bollinger Band Strategy")
        logger.info("=" * 70)
        logger.info(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"Exchange: Paradex (0.019% taker fee)")
        logger.info(f"Assets: {self.assets}")
        logger.info(f"Capital: $100 total")
        logger.info(f"Position: ${self.config.margin_per_trade} margin x {self.config.leverage}x = ${notional} notional")
        logger.info(f"Est. fee per trade: ${fee_per_trade:.2f}")
        logger.info("")
        logger.info("Strategy:")
        logger.info(f"  Entry: SHORT when price > Upper BB({self.config.bb_period}) AND RSI({self.config.rsi_period}) > threshold")
        logger.info(f"  RSI thresholds: {self.rsi_entry}")
        logger.info(f"  Exit: Trail {self.config.trailing_stop_pct}%@{self.config.trailing_activation_pct}% OR Lower BB OR SL {self.config.stop_loss_pct}%")
        logger.info("")
        logger.info("=" * 70)

        tasks = [self.run_asset(asset) for asset in self.assets]
        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop strategy and close positions."""
        self.running = False

        # Close any open positions
        for asset, trade in self.active_trades.items():
            if trade is not None:
                try:
                    bbo = await self.clients[asset].get_bbo()
                    mid_price = (bbo.bid_price + bbo.ask_price) / 2
                    await self.exit_short(asset, mid_price, "shutdown")
                except Exception as e:
                    logger.error(f"{asset}: Failed to close on shutdown: {e}")

        # Disconnect clients
        for asset, client in self.clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"{asset}: Error disconnecting: {e}")

        # Print final stats
        logger.info("=" * 70)
        logger.info("FINAL STATS")
        logger.info("=" * 70)
        total_trades = 0
        total_wins = 0
        total_pnl = 0
        total_fees = 0

        for asset, stats in self.stats.items():
            if stats["trades"] > 0:
                wr = stats["wins"] / stats["trades"] * 100
                logger.info(
                    f"{asset}: {stats['trades']} trades, WR: {wr:.1f}%, "
                    f"Trail: {stats['trailing_exits']}, BB: {stats['lower_bb_exits']}, "
                    f"SL: {stats['sl_exits']}, Fees: ${stats['fees']:.2f}, Net PnL: ${stats['pnl']:.2f}"
                )
                total_trades += stats["trades"]
                total_wins += stats["wins"]
                total_pnl += stats["pnl"]
                total_fees += stats["fees"]

        if total_trades > 0:
            logger.info(f"TOTAL: {total_trades} trades, WR: {total_wins/total_trades*100:.1f}%, Fees: ${total_fees:.2f}, Net PnL: ${total_pnl:.2f}")
        logger.info("=" * 70)


async def main(args):
    """Main entry point."""
    assets = [a.upper() for a in args.assets.split(',')]
    valid_assets = ['BTC', 'ETH', 'SOL']
    for asset in assets:
        if asset not in valid_assets:
            logger.error(f"Invalid asset: {asset}. Valid: {valid_assets}")
            return

    # Create config with $20 margin
    config = BBRSIConfig(
        bb_period=20,
        bb_std=2.0,
        rsi_period=7,
        stop_loss_pct=0.30,
        trailing_stop_pct=0.20,
        trailing_activation_pct=0.10,
        margin_per_trade=20.0,  # $20 per trade
        leverage=20.0,          # 20x = $400 notional
    )

    # Per-asset RSI entry thresholds
    rsi_entry_overrides = {
        "BTC": 75,
        "ETH": 60,
        "SOL": 65,
    }

    # Create bot
    bot = ParaBBRSIBot(
        assets=assets,
        config=config,
        paper_mode=not args.live,
        rsi_entry_overrides=rsi_entry_overrides,
    )

    # Setup
    if not await bot.setup():
        logger.error("Failed to setup bot")
        return

    # Handle shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received...")
        asyncio.create_task(bot.stop())

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
    finally:
        await bot.stop()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Para BB RSI Bot - RSI+BB Strategy on Paradex")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL", help="Comma-separated assets (default: BTC,ETH,SOL)")

    args = parser.parse_args()

    asyncio.run(main(args))
