#!/usr/bin/env python3
"""
BB Short Multi-Asset Bot

Runs the Bollinger Band Short (Mean Reversion) strategy on BTC, ETH, and SOL simultaneously.

Strategy:
- Entry: SHORT when price > upper Bollinger Band (overbought)
- Exit: SL = 0.3%, TP = 0.4%
- No time-based exits - pure SL/TP only

Verified metrics (90-day backtest):
- BTC: 46.1% WR, 14 trades/day, +32% return
- ETH: 46.8% WR, 19 trades/day, +60% return
- SOL: 46.1% WR, 23 trades/day, +56% return

Usage:
    # Paper trading (default)
    python run_bb_short.py

    # Live trading
    python run_bb_short.py --live

    # Single asset
    python run_bb_short.py --assets ETH

    # Custom position size
    python run_bb_short.py --size 100
"""

import argparse
import asyncio
import logging
import signal
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from .env
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
        logging.FileHandler(os.path.join(LOGS_DIR, f'bb_short_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

from strategy.bb_short_strategy import BBShortStrategy, BBShortConfig
from exchange.lighter_client import LighterClient


async def main(args):
    """Main entry point."""

    # Parse assets
    assets = [a.upper() for a in args.assets.split(',')]
    valid_assets = ['BTC', 'ETH', 'SOL']
    for asset in assets:
        if asset not in valid_assets:
            logger.error(f"Invalid asset: {asset}. Valid: {valid_assets}")
            return

    logger.info("="*60)
    logger.info("BB SHORT MULTI-ASSET BOT")
    logger.info("="*60)
    logger.info(f"Mode: {'LIVE' if args.live else 'PAPER'}")
    logger.info(f"Assets: {assets}")
    logger.info(f"Margin per trade: ${args.size}")
    logger.info(f"Leverage: 20x")
    logger.info(f"Position size: ${args.size * 20:.0f} notional per trade")
    logger.info(f"Exchange: Lighter (0% fees)")
    logger.info("="*60)
    logger.info("")
    logger.info("Strategy: Short when price > upper Bollinger Band")
    logger.info("Exit: SL = 0.3%, TP = 0.4%")
    logger.info("Expected WR: ~46% (breakeven: 42.9%)")
    logger.info("")
    logger.info("="*60)

    # Create exchange clients for each asset
    clients = {}
    for asset in assets:
        market = f"{asset}-USD-PERP"
        client = LighterClient(market=market, paper_mode=not args.live)
        clients[asset] = client
        logger.info(f"Created client for {asset} ({market})")

    # Create strategy config
    # $100 margin × 20x leverage = $2000 notional per trade
    config = BBShortConfig(
        bb_period=20,
        bb_std=2.0,
        stop_loss_pct=0.30,
        take_profit_pct=0.40,
        margin_per_trade=args.size,
        leverage=20.0,
    )

    # Create strategy
    strategy = BBShortStrategy(exchange_clients=clients, config=config)

    # Handle shutdown gracefully
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received...")
        asyncio.create_task(strategy.stop())

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        # Run the strategy
        await strategy.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
    finally:
        await strategy.stop()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BB Short Multi-Asset Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL", help="Comma-separated assets (default: BTC,ETH,SOL)")
    parser.add_argument("--size", type=float, default=100.0, help="Position size in USD per asset (default: 100)")

    args = parser.parse_args()

    asyncio.run(main(args))
