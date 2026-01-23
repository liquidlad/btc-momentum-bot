#!/usr/bin/env python3
"""
RSI+BB Bot

RSI + Bollinger Band Strategy

Entry: SHORT when price > upper BB AND RSI > 75
Exit: Lower BB OR max hold time (10 min) OR stop loss (0.3%)

RSI threshold: 75 for all assets (BTC, ETH, SOL)

Usage:
    python run_rsi_bb.py              # Paper trading
    python run_rsi_bb.py --live       # Live trading
    python run_rsi_bb.py --assets ETH # Single asset
    python run_rsi_bb.py --size 50    # Custom margin size
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
        logging.FileHandler(os.path.join(LOGS_DIR, f'rsi_bb_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

from strategy.rsi_bb_strategy import RSIBBStrategy, RSIBBConfig
from exchange.lighter_client import LighterClient


async def main(args):
    """Main entry point."""

    assets = [a.upper() for a in args.assets.split(',')]
    valid_assets = ['BTC', 'ETH', 'SOL']
    for asset in assets:
        if asset not in valid_assets:
            logger.error(f"Invalid asset: {asset}. Valid: {valid_assets}")
            return

    logger.info("=" * 70)
    logger.info("RSI+BB BOT")
    logger.info("=" * 70)
    logger.info(f"Mode: {'LIVE' if args.live else 'PAPER'}")
    logger.info(f"Assets: {assets}")
    logger.info(f"Margin per trade: ${args.size}")
    logger.info(f"Leverage: 20x")
    logger.info(f"Position size: ${args.size * 20:.0f} notional per trade")
    logger.info(f"Exchange: Lighter (0% fees)")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Strategy:")
    logger.info("  Entry: SHORT when price > upper BB(40) AND RSI(7) > 75")
    logger.info("  Exit: Lower BB OR Max Hold 10min OR SL 0.3%")
    logger.info("")
    logger.info("=" * 70)

    # Create exchange clients
    clients = {}
    for asset in assets:
        market = f"{asset}-USD-PERP"
        client = LighterClient(market=market, paper_mode=not args.live)
        clients[asset] = client
        logger.info(f"Created client for {asset} ({market})")

    # Create strategy config
    # BB period 40 with 10-sec polling = 400 sec = ~6.7 min of data
    config = RSIBBConfig(
        bb_period=40,
        bb_std=2.0,
        rsi_period=7,
        stop_loss_pct=0.30,
        max_hold_seconds=600,  # 10 minutes
        margin_per_trade=args.size,
        leverage=20.0,
    )

    # Per-asset RSI entry thresholds (all 75 based on performance analysis)
    rsi_entry_overrides = {
        "BTC": 75,
        "ETH": 75,
        "SOL": 75,
    }

    # Create strategy
    strategy = RSIBBStrategy(
        exchange_clients=clients,
        config=config,
        rsi_entry_overrides=rsi_entry_overrides,
    )

    # Handle shutdown
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received...")
        asyncio.create_task(strategy.stop())

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        await strategy.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received...")
    finally:
        await strategy.stop()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSI+BB Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL", help="Comma-separated assets (default: BTC,ETH,SOL)")
    parser.add_argument("--size", type=float, default=100.0, help="Margin size in USD per asset (default: 100)")

    args = parser.parse_args()

    asyncio.run(main(args))
