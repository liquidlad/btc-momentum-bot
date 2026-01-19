#!/usr/bin/env python3
"""
BB RSI All Assets Bot - RSI + Bollinger Band Strategy on Paradex

Runs on BTC, ETH, and SOL simultaneously.

Entry: SHORT when price > upper BB AND RSI > threshold
Exit: Trailing stop (0.2% after 0.1% profit) OR lower BB OR stop loss 0.3%

RSI thresholds:
- BTC: 75
- ETH: 60
- SOL: 65

Usage:
    python run_bb_rsi_all.py              # Paper trading
    python run_bb_rsi_all.py --live       # Live trading
    python run_bb_rsi_all.py --assets BTC # Single asset
"""

import argparse
import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np
import requests

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "strategy"))

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Setup logging
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, f'bb_rsi_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)


# RSI entry thresholds per asset
RSI_THRESHOLDS = {
    "BTC": 75,
    "ETH": 60,
    "SOL": 65,
}


@dataclass
class BBRSIConfig:
    """Configuration for RSI+BB strategy."""
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 7
    stop_loss_pct: float = 0.30
    trailing_stop_pct: float = 0.20
    trailing_activation_pct: float = 0.10
    margin_per_trade: float = 20.0
    leverage: float = 20.0


@dataclass
class BBRSITrade:
    """Active short trade."""
    asset: str
    entry_time: datetime
    entry_price: float
    size: float
    stop_loss_price: float
    best_price: float
    trailing_active: bool = False
    entry_rsi: float = 0.0
    candles_held: int = 0  # Track hold time


class BBRSIAllBot:
    """RSI+BB strategy bot running on multiple assets."""

    def __init__(
        self,
        assets: list,
        config: BBRSIConfig,
        paper_mode: bool = True,
    ):
        self.assets = assets
        self.config = config
        self.paper_mode = paper_mode
        self.running = False

        # Per-asset state
        self.clients: Dict = {}
        self.trades: Dict[str, Optional[BBRSITrade]] = {a: None for a in assets}
        self.prices: Dict[str, deque] = {a: deque(maxlen=30) for a in assets}
        self.stats: Dict[str, dict] = {
            a: {"trades": 0, "wins": 0, "pnl": 0.0, "trailing_exits": 0, "lower_bb_exits": 0, "sl_exits": 0, "fees": 0.0}
            for a in assets
        }

    async def setup(self) -> bool:
        """Initialize exchange clients."""
        from exchange.paradex_client import ParadexClient

        for asset in self.assets:
            market = f"{asset}-USD-PERP"
            try:
                client = ParadexClient(market=market, paper_mode=self.paper_mode)
                connected = await client.connect()
                if not connected:
                    logger.error(f"Failed to connect client for {asset}")
                    return False
                self.clients[asset] = client
                logger.info(f"Connected to Paradex for {asset} ({market})")
            except Exception as e:
                logger.error(f"Failed to setup {asset}: {e}")
                return False

        return True

    def _calculate_bb_rsi(self, prices: list) -> tuple:
        """Calculate current BB and RSI values."""
        if len(prices) < self.config.bb_period:
            return None, None, None, None

        closes = np.array(prices)

        # Bollinger Bands
        sma = np.mean(closes[-self.config.bb_period:])
        std = np.std(closes[-self.config.bb_period:])
        bb_upper = sma + (self.config.bb_std * std)
        bb_lower = sma - (self.config.bb_std * std)

        # RSI
        if len(closes) < self.config.rsi_period + 1:
            return bb_upper, bb_lower, sma, None

        deltas = np.diff(closes[-(self.config.rsi_period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        return bb_upper, bb_lower, sma, rsi

    async def _check_entry(self, asset: str, price: float) -> bool:
        """Check if SHORT entry conditions are met."""
        prices = list(self.prices[asset])
        if len(prices) < self.config.bb_period:
            return False

        bb_upper, bb_lower, bb_mid, rsi = self._calculate_bb_rsi(prices)
        if bb_upper is None or rsi is None:
            return False

        rsi_threshold = RSI_THRESHOLDS.get(asset, 75)

        if price > bb_upper and rsi > rsi_threshold:
            logger.info(f"{asset}: Entry signal - price {price:.2f} > BB upper {bb_upper:.2f}, RSI {rsi:.1f} > {rsi_threshold}")
            return True
        return False

    async def _check_exit(self, asset: str, price: float) -> Optional[str]:
        """Check exit conditions for active trade."""
        trade = self.trades[asset]
        if not trade:
            return None

        # Increment hold counter
        trade.candles_held += 1

        # Stop loss - always check
        if price >= trade.stop_loss_price:
            return "stop_loss"

        # Trailing stop check
        profit_pct = (trade.entry_price - price) / trade.entry_price * 100

        if profit_pct >= self.config.trailing_activation_pct and not trade.trailing_active:
            trade.trailing_active = True
            logger.info(f"{asset}: Trailing stop activated at {profit_pct:.2f}% profit")

        if trade.trailing_active:
            if price < trade.best_price:
                trade.best_price = price
            trail_price = trade.best_price * (1 + self.config.trailing_stop_pct / 100)
            if price >= trail_price:
                return "trailing_stop"

        # Lower BB exit - only after minimum hold time (5 minutes)
        MIN_HOLD_SECONDS = 300  # 5 minutes
        hold_time = (datetime.now() - trade.entry_time).total_seconds()

        if hold_time >= MIN_HOLD_SECONDS:
            prices = list(self.prices[asset])
            if len(prices) >= self.config.bb_period:
                bb_upper, bb_lower, _, _ = self._calculate_bb_rsi(prices)
                if bb_lower and price <= bb_lower:
                    return "lower_bb"

        return None

    async def _open_trade(self, asset: str, price: float) -> None:
        """Open a SHORT position."""
        from exchange.base import OrderSide, OrderType

        notional = self.config.margin_per_trade * self.config.leverage
        size = notional / price

        # Round size
        if asset == "SOL":
            size = round(size, 2)
        else:
            size = round(size, 4)

        stop_loss_price = price * (1 + self.config.stop_loss_pct / 100)

        try:
            order = await self.clients[asset].place_order(
                side=OrderSide.SELL,
                size=size,
                order_type=OrderType.MARKET,
            )

            entry_price = order.avg_fill_price or price

            prices = list(self.prices[asset])
            _, _, _, rsi = self._calculate_bb_rsi(prices)

            self.trades[asset] = BBRSITrade(
                asset=asset,
                entry_time=datetime.now(),
                entry_price=entry_price,
                size=size,
                stop_loss_price=stop_loss_price,
                best_price=entry_price,
                entry_rsi=rsi or 0,
            )

            logger.info(f"{asset}: OPENED SHORT size={size:.4f} @ {entry_price:.2f}, SL={stop_loss_price:.2f}")

        except Exception as e:
            logger.error(f"{asset}: Failed to open trade: {e}")

    async def _close_trade(self, asset: str, price: float, reason: str) -> None:
        """Close a SHORT position."""
        from exchange.base import OrderSide, OrderType

        trade = self.trades[asset]
        if not trade:
            return

        try:
            order = await self.clients[asset].place_order(
                side=OrderSide.BUY,
                size=trade.size,
                order_type=OrderType.MARKET,
            )

            exit_price = order.avg_fill_price or price

            # Calculate P&L
            pnl = (trade.entry_price - exit_price) * trade.size
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price * 100

            # Fees (Paradex 0.019% per side)
            fee_rate = 0.019 / 100
            notional = trade.size * trade.entry_price
            fees = notional * fee_rate * 2
            net_pnl = pnl - fees

            # Update stats
            self.stats[asset]["trades"] += 1
            self.stats[asset]["pnl"] += net_pnl
            self.stats[asset]["fees"] += fees
            if net_pnl > 0:
                self.stats[asset]["wins"] += 1

            if reason == "trailing_stop":
                self.stats[asset]["trailing_exits"] += 1
            elif reason == "lower_bb":
                self.stats[asset]["lower_bb_exits"] += 1
            elif reason == "stop_loss":
                self.stats[asset]["sl_exits"] += 1

            logger.info(f"{asset}: CLOSED SHORT @ {exit_price:.2f}, P&L=${net_pnl:.2f} ({pnl_pct:.2f}%), reason={reason}")

            self.trades[asset] = None

        except Exception as e:
            logger.error(f"{asset}: Failed to close trade: {e}")

    async def _fetch_bootstrap_prices(self, asset: str) -> None:
        """Fetch historical prices from Binance for indicator bootstrap."""
        symbol = f"{asset}USDT"
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=30"
            response = requests.get(url, timeout=10)
            data = response.json()

            for kline in data[:-1]:  # Skip last incomplete candle
                self.prices[asset].append(float(kline[4]))  # Close price

            logger.info(f"{asset}: Bootstrapped with {len(self.prices[asset])} historical prices")
        except Exception as e:
            logger.warning(f"{asset}: Could not bootstrap: {e}")

    async def run(self) -> None:
        """Main trading loop."""
        self.running = True

        logger.info("=" * 70)
        logger.info("BB RSI ALL ASSETS BOT - Paradex RSI+Bollinger Band Strategy")
        logger.info("=" * 70)
        logger.info(f"Mode: {'PAPER' if self.paper_mode else 'LIVE'}")
        logger.info(f"Exchange: Paradex (0.019% taker fee)")
        logger.info(f"Assets: {self.assets}")
        logger.info(f"Position: ${self.config.margin_per_trade} margin x {self.config.leverage}x = ${self.config.margin_per_trade * self.config.leverage} notional")
        logger.info("")
        logger.info("Strategy:")
        logger.info(f"  Entry: SHORT when price > Upper BB({self.config.bb_period}) AND RSI({self.config.rsi_period}) > threshold")
        logger.info(f"  RSI thresholds: {RSI_THRESHOLDS}")
        logger.info(f"  Exit: Trail {self.config.trailing_stop_pct}%@{self.config.trailing_activation_pct}% OR Lower BB OR SL {self.config.stop_loss_pct}%")
        logger.info("=" * 70)

        # Bootstrap prices
        for asset in self.assets:
            await self._fetch_bootstrap_prices(asset)
            logger.info(f"{asset}: Starting RSI+BB strategy (RSI entry > {RSI_THRESHOLDS.get(asset, 75)})")

        last_log_time = datetime.now()

        try:
            while self.running:
                for asset in self.assets:
                    try:
                        # Get current price
                        bbo = await self.clients[asset].get_bbo()
                        price = (bbo.bid_price + bbo.ask_price) / 2
                        self.prices[asset].append(price)

                        # Check active trade
                        if self.trades[asset]:
                            exit_reason = await self._check_exit(asset, price)
                            if exit_reason:
                                await self._close_trade(asset, price, exit_reason)
                        else:
                            # Check entry
                            if await self._check_entry(asset, price):
                                await self._open_trade(asset, price)

                    except Exception as e:
                        logger.error(f"{asset}: Error in loop: {e}")

                # Log status every 60 seconds
                now = datetime.now()
                if (now - last_log_time).seconds >= 60:
                    for asset in self.assets:
                        prices = list(self.prices[asset])
                        if len(prices) >= self.config.bb_period:
                            bb_upper, bb_lower, _, rsi = self._calculate_bb_rsi(prices)
                            price = prices[-1] if prices else 0
                            pos = "YES" if self.trades[asset] else "NO"
                            logger.info(f"{asset}: ${price:.2f} | RSI={rsi:.1f} | BB=[{bb_lower:.2f}, {bb_upper:.2f}] | Pos={pos} | Trades={self.stats[asset]['trades']} | PnL=${self.stats[asset]['pnl']:.2f}")
                    last_log_time = now

                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Bot cancelled")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Graceful shutdown."""
        self.running = False
        logger.info("Shutting down...")

        # Close open positions
        for asset in self.assets:
            if self.trades[asset]:
                try:
                    bbo = await self.clients[asset].get_bbo()
                    price = (bbo.bid_price + bbo.ask_price) / 2
                    await self._close_trade(asset, price, "shutdown")
                except Exception as e:
                    logger.error(f"{asset}: Error closing position: {e}")

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

    config = BBRSIConfig(
        bb_period=20,
        bb_std=2.0,
        rsi_period=7,
        stop_loss_pct=0.30,
        trailing_stop_pct=0.20,
        trailing_activation_pct=0.10,
        margin_per_trade=20.0,
        leverage=20.0,
    )

    bot = BBRSIAllBot(
        assets=assets,
        config=config,
        paper_mode=not args.live,
    )

    if not await bot.setup():
        logger.error("Failed to setup bot")
        return

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
    parser = argparse.ArgumentParser(description="BB RSI All Assets Bot - RSI+BB Strategy on Paradex")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL", help="Comma-separated assets (default: BTC,ETH,SOL)")

    args = parser.parse_args()
    asyncio.run(main(args))
