#!/usr/bin/env python3
"""
BTC Momentum Trading Bot - Main Entry Point

This bot executes a momentum scalping strategy on Paradex (primary) or Lighter (secondary).
Designed for high-volume trading with tight stops and quick take-profits.

PHASE 1 (Complete): Strategy backtested on 90 days of 1m data
- 54.7 trades/day, 150.94% return, 25.3% max drawdown
- Optimized parameters in config/optimized_params.json

PHASE 2: Live/Paper trading with exchange integration
- Paradex: Zero-fee perp DEX on Starknet L2
- Lighter: Zero-fee perp DEX on zk-rollup

Usage:
    # Paper trading (simulated)
    python runbot.py --exchange paradex --ticker BTC --paper

    # Live trading (requires credentials)
    python runbot.py --exchange paradex --ticker BTC

    # With custom config
    python runbot.py --exchange paradex --ticker BTC --config config/optimized_params.json
"""

import argparse
import asyncio
import json
import os
import sys
import signal
import time
from datetime import datetime, timedelta
from typing import Optional
import logging

# Setup path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the bot."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/bot_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    return logger


class MomentumBot:
    """
    Main trading bot that integrates Phase 1 strategy with Phase 2 exchange APIs.
    """

    def __init__(
        self,
        exchange: str,
        ticker: str,
        strategy_params: dict,
        paper_mode: bool = True,
        use_testnet: bool = False,
    ):
        self.exchange_name = exchange
        self.ticker = ticker
        self.strategy_params = strategy_params
        self.paper_mode = paper_mode
        self.use_testnet = use_testnet

        self.logger = logging.getLogger(__name__)
        self.running = False
        self.exchange_client = None
        self.strategy = None

        # Load configs
        self.risk_config = load_config("config/risk_limits.json")

        # Determine market symbol
        if exchange == "paradex":
            self.market = f"{ticker}-USD-PERP"
        else:  # lighter
            self.market = f"{ticker}-USD"

        self.logger.info(f"Initialized MomentumBot")
        self.logger.info(f"Exchange: {exchange}")
        self.logger.info(f"Market: {self.market}")
        self.logger.info(f"Paper mode: {paper_mode}")

    async def setup(self) -> bool:
        """Initialize exchange client and strategy."""
        try:
            # Create exchange client
            if self.exchange_name == "paradex":
                from exchange.paradex_client import ParadexClient
                self.exchange_client = ParadexClient(
                    market=self.market,
                    paper_mode=self.paper_mode,
                    use_testnet=self.use_testnet,
                )
            else:  # lighter
                from exchange.lighter_client import LighterClient
                self.exchange_client = LighterClient(
                    market=self.market,
                    paper_mode=self.paper_mode,
                )

            # Connect to exchange
            connected = await self.exchange_client.connect()
            if not connected:
                self.logger.error("Failed to connect to exchange")
                return False

            # Create live strategy
            from strategy.live_strategy import LiveStrategy
            self.strategy = LiveStrategy(
                exchange_client=self.exchange_client,
                strategy_params=self.strategy_params,
                risk_params=self.risk_config,
            )

            # Add trade callback for logging
            self.strategy.add_trade_callback(self._on_trade_event)

            self.logger.info("Setup complete")
            return True

        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _on_trade_event(self, event_type: str, data) -> None:
        """Callback for trade events."""
        if event_type == "open":
            self.logger.info(f"Trade opened: {data.side} @ {data.entry_price:.2f}")
        elif event_type == "close":
            self.logger.info(f"Trade closed: P&L ${data.pnl:.2f} ({data.pnl_pct:.2f}%)")

    async def run(self) -> None:
        """Main bot loop."""
        self.running = True

        self.logger.info("=" * 60)
        if self.paper_mode:
            self.logger.info("PAPER TRADING MODE - No real orders will be placed")
        else:
            self.logger.info("LIVE TRADING MODE - Real orders will be placed!")
        self.logger.info("=" * 60)

        # Print strategy parameters
        self.logger.info("Strategy Parameters (from Phase 1 optimization):")
        self.logger.info(f"  min_conditions: {self.strategy_params.get('min_conditions', 3)}")
        self.logger.info(f"  ema_fast: {self.strategy_params.get('ema_fast', 3)}")
        self.logger.info(f"  ema_slow: {self.strategy_params.get('ema_slow', 15)}")
        self.logger.info(f"  roc_threshold: {self.strategy_params.get('roc_threshold', 0.08)}%")
        self.logger.info(f"  stop_loss: {self.strategy_params.get('stop_loss_pct', 0.10)}%")
        self.logger.info(f"  take_profit: {self.strategy_params.get('take_profit_1_pct', 0.12)}%")

        # Variables for candle aggregation
        candle_interval = 60  # 1 minute in seconds
        current_candle = None
        candle_start_time = None

        try:
            while self.running:
                # Get current BBO
                bbo = await self.exchange_client.get_bbo()
                mid_price = (bbo.bid_price + bbo.ask_price) / 2

                # Update strategy with price (for stop-loss monitoring)
                await self.strategy.on_price_update(mid_price)

                # Check if we need to start a new candle
                current_time = datetime.now()
                current_minute = current_time.replace(second=0, microsecond=0)

                if candle_start_time is None or current_minute > candle_start_time:
                    # Close previous candle if exists
                    if current_candle is not None:
                        signal = await self.strategy.on_candle_close(current_candle)
                        self.logger.debug(f"Candle closed: O={current_candle['open']:.2f} H={current_candle['high']:.2f} L={current_candle['low']:.2f} C={current_candle['close']:.2f} Signal={signal}")

                    # Start new candle
                    candle_start_time = current_minute
                    current_candle = {
                        "timestamp": int(candle_start_time.timestamp() * 1000),
                        "open": mid_price,
                        "high": mid_price,
                        "low": mid_price,
                        "close": mid_price,
                        "volume": 0,  # We don't have real volume in paper mode
                    }

                # Update current candle
                if current_candle:
                    current_candle["high"] = max(current_candle["high"], mid_price)
                    current_candle["low"] = min(current_candle["low"], mid_price)
                    current_candle["close"] = mid_price

                # Update paper trading BBO if applicable
                if self.paper_mode and hasattr(self.exchange_client, 'update_paper_bbo'):
                    self.exchange_client.update_paper_bbo(bbo)

                # Log status periodically
                if current_time.second == 0:
                    stats = self.strategy.get_stats()
                    self.logger.info(
                        f"Price: ${mid_price:.2f} | "
                        f"Equity: ${stats['equity']:.2f} | "
                        f"Daily P&L: ${stats['daily_pnl']:.2f} | "
                        f"Trades: {stats['total_trades']} | "
                        f"Win Rate: {stats['win_rate']:.1f}% | "
                        f"Position: {'YES' if stats['has_position'] else 'NO'}"
                    )

                # Sleep before next iteration
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("Bot cancelled")
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self.running = False
        self.logger.info("Shutting down...")

        # Close any open positions
        if self.strategy and self.strategy.state.active_trade:
            self.logger.info("Closing open position...")
            try:
                bbo = await self.exchange_client.get_bbo()
                mid_price = (bbo.bid_price + bbo.ask_price) / 2
                await self.strategy._close_position("shutdown", mid_price)
            except Exception as e:
                self.logger.error(f"Error closing position: {e}")

        # Disconnect from exchange
        if self.exchange_client:
            await self.exchange_client.disconnect()

        # Print final stats
        if self.strategy:
            stats = self.strategy.get_stats()
            self.logger.info("=" * 60)
            self.logger.info("FINAL STATISTICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Final Equity: ${stats['equity']:.2f}")
            self.logger.info(f"Total P&L: ${stats['equity'] - 200:.2f}")
            self.logger.info(f"Total Trades: {stats['total_trades']}")
            self.logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
            self.logger.info(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
            self.logger.info("=" * 60)


async def main_async(args) -> None:
    """Async main function."""
    # Load strategy config
    try:
        config = load_config(args.config)
        # Handle both flat and nested config formats
        if "optimized_params" in config:
            strategy_params = config["optimized_params"]
        else:
            strategy_params = config.get("indicator_sets", {}).get("ema_rsi_volume", {}).get("params", config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Using default parameters...")
        strategy_params = {
            "min_conditions": 3,
            "ema_fast": 3,
            "ema_slow": 15,
            "roc_threshold": 0.08,
            "volume_multiplier": 1.0,
            "stop_loss_pct": 0.10,
            "take_profit_1_pct": 0.12,
        }

    # Create bot
    bot = MomentumBot(
        exchange=args.exchange,
        ticker=args.ticker,
        strategy_params=strategy_params,
        paper_mode=args.paper,
        use_testnet=args.testnet,
    )

    # Setup
    if not await bot.setup():
        print("Failed to setup bot")
        return

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        bot.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Run bot
    await bot.run()


def main():
    parser = argparse.ArgumentParser(
        description="BTC Momentum Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Paper trading on Paradex
    python runbot.py --exchange paradex --ticker BTC --paper

    # Live trading on Lighter
    python runbot.py --exchange lighter --ticker BTC

    # Use optimized parameters from Phase 1
    python runbot.py --exchange paradex --ticker BTC --paper --config config/optimized_params.json

Environment Variables (for live trading):
    Paradex:
        PARADEX_L2_PRIVATE_KEY - L2 private key
        PARADEX_L2_ADDRESS - L2 address
        or
        PARADEX_L1_ADDRESS - L1 Ethereum address
        PARADEX_L1_PRIVATE_KEY - L1 private key

    Lighter:
        LIGHTER_ACCOUNT_INDEX - Account index
        LIGHTER_API_KEY_INDEX - API key index
        API_KEY_PRIVATE_KEY - API private key
        """
    )

    parser.add_argument(
        "--exchange",
        type=str,
        required=True,
        choices=["paradex", "lighter"],
        help="Exchange to trade on"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="BTC",
        help="Ticker to trade (default: BTC)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/optimized_params.json",
        help="Path to strategy config file (default: config/optimized_params.json)"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (no real orders)"
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use testnet instead of mainnet"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validate environment variables for live trading
    if not args.paper:
        if args.exchange == "paradex":
            # Check for L2-only OR L1+L2 credentials
            has_l2 = os.environ.get("PARADEX_L2_PRIVATE_KEY") and os.environ.get("PARADEX_L2_ADDRESS")
            has_l1 = os.environ.get("PARADEX_L1_ADDRESS") and os.environ.get("PARADEX_L1_PRIVATE_KEY")

            if not (has_l2 or has_l1):
                print("Error: Missing Paradex credentials for live trading.")
                print("Set either:")
                print("  - PARADEX_L2_PRIVATE_KEY and PARADEX_L2_ADDRESS (recommended)")
                print("  - PARADEX_L1_ADDRESS and PARADEX_L1_PRIVATE_KEY")
                print("\nOr use --paper for paper trading.")
                sys.exit(1)
        else:  # lighter
            required_vars = ["LIGHTER_ACCOUNT_INDEX", "LIGHTER_API_KEY_INDEX", "API_KEY_PRIVATE_KEY"]
            missing = [var for var in required_vars if not os.environ.get(var)]
            if missing:
                print(f"Error: Missing Lighter credentials for live trading:")
                for var in missing:
                    print(f"  - {var}")
                print("\nOr use --paper for paper trading.")
                sys.exit(1)

    # Run async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nShutdown requested...")


if __name__ == "__main__":
    main()
