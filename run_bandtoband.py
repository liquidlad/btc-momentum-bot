#!/usr/bin/env python3
"""
Band-to-Band Bot

Runs the Bollinger Band Band-to-Band strategy on BTC, ETH, and SOL.

Strategy:
- Entry: SHORT when price > upper Bollinger Band (overbought)
- Exit: When price < lower Bollinger Band (full mean reversion) or SL hit
- SL optimized per asset: BTC 0.2%, ETH 0.4%, SOL 0.3%

Position size: 25% of main bot ($25 margin × 20x = $500 notional per trade)

Usage:
    # Paper trading (default)
    python run_bandtoband.py

    # Live trading
    python run_bandtoband.py --live

    # Single asset
    python run_bandtoband.py --assets ETH
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
        logging.FileHandler(os.path.join(LOGS_DIR, f'bandtoband_{__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
logger = logging.getLogger(__name__)

from strategy.bb_band2band_strategy import BB2BStrategy, BB2BConfig
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
    logger.info("BAND-TO-BAND BOT")
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
    logger.info("Exit: When price < lower Bollinger Band (full reversion)")
    logger.info("Stop Loss: BTC 0.2%, ETH 0.4%, SOL 0.3%")
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
    config = BB2BConfig(
        bb_period=20,
        bb_std=2.0,
        margin_per_trade=args.size,
        leverage=20.0,
    )

    # Per-asset optimized stop losses from backtest
    asset_sl_overrides = {
        "BTC": 0.20,
        "ETH": 0.40,
        "SOL": 0.30,
    }

    # Create strategy
    strategy = BB2BStrategy(
        exchange_clients=clients,
        config=config,
        asset_sl_overrides=asset_sl_overrides,
    )

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
    parser = argparse.ArgumentParser(description="Band-to-Band Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL", help="Comma-separated assets (default: BTC,ETH,SOL)")
    parser.add_argument("--size", type=float, default=25.0, help="Margin size in USD per asset (default: 25)")

    args = parser.parse_args()

    asyncio.run(main(args))
