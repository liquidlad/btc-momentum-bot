#!/usr/bin/env python3
"""
BTC Momentum Trading Bot - Main Entry Point

This bot executes a momentum scalping strategy on Paradex (primary) or Lighter (secondary).
Designed for high-volume trading with tight stops and quick take-profits.

Usage:
    # Paper trading
    python runbot.py --exchange paradex --ticker BTC --paper

    # Live trading
    python runbot.py --exchange paradex --ticker BTC --quantity 1000

    # With custom config
    python runbot.py --exchange paradex --ticker BTC --config config/optimized_params.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

# Add strategy module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'strategy'))


def load_config(config_path: str) -> dict:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_logging():
    """Configure logging for the bot."""
    import logging

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/bot_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


class TradingBot:
    """Main trading bot class."""

    def __init__(
        self,
        exchange: str,
        ticker: str,
        config: dict,
        paper_mode: bool = True,
        quantity: float = 1000,
    ):
        self.exchange = exchange
        self.ticker = ticker
        self.config = config
        self.paper_mode = paper_mode
        self.base_quantity = quantity

        self.logger = setup_logging()
        self.running = False

        # Load exchange and risk configs
        self.exchange_config = load_config("config/exchange_config.json")
        self.risk_limits = load_config("config/risk_limits.json")

        # State
        self.position = None
        self.equity = self.risk_limits["capital"]["initial"]
        self.daily_pnl = 0
        self.daily_trades = 0
        self.consecutive_losses = 0

        self.logger.info(f"Initialized bot for {exchange} - {ticker}")
        self.logger.info(f"Paper mode: {paper_mode}")
        self.logger.info(f"Starting equity: ${self.equity}")

    def check_circuit_breakers(self) -> bool:
        """Check if any circuit breakers are triggered."""
        limits = self.risk_limits["circuit_breakers"]

        # Check consecutive losses
        if self.consecutive_losses >= limits["max_consecutive_losses"]:
            self.logger.warning(f"Circuit breaker: {self.consecutive_losses} consecutive losses")
            return False

        # Check daily drawdown
        daily_dd_pct = abs(self.daily_pnl / self.equity * 100) if self.daily_pnl < 0 else 0
        if daily_dd_pct >= limits["max_daily_drawdown_pct"]:
            self.logger.warning(f"Circuit breaker: Daily drawdown {daily_dd_pct:.1f}%")
            return False

        # Check daily trade limit
        if self.daily_trades >= limits["max_daily_trades"]:
            self.logger.warning(f"Circuit breaker: Max daily trades reached ({self.daily_trades})")
            return False

        return True

    def calculate_position_size(self) -> float:
        """Calculate position size based on risk parameters."""
        risk_per_trade = self.equity * (self.risk_limits["position_limits"]["risk_per_trade_pct"] / 100)
        stop_loss_pct = self.risk_limits["stop_loss"]["default_pct"] / 100

        # Base size from risk
        size = risk_per_trade / stop_loss_pct if stop_loss_pct > 0 else 0

        # Apply drawdown scaling
        if self.risk_limits["drawdown_scaling"]["enabled"]:
            current_dd = abs(self.daily_pnl / self.equity * 100) if self.daily_pnl < 0 else 0
            for threshold in self.risk_limits["drawdown_scaling"]["thresholds"]:
                if current_dd >= threshold["drawdown_pct"]:
                    size *= threshold["position_multiplier"]
                    break

        # Cap at max position
        max_position = self.risk_limits["position_limits"]["max_position_notional"]
        return min(size, max_position)

    def run(self):
        """Main bot loop."""
        self.running = True
        self.logger.info("Starting trading bot...")

        if self.paper_mode:
            self.logger.info("=" * 50)
            self.logger.info("PAPER TRADING MODE - No real orders will be placed")
            self.logger.info("=" * 50)

        try:
            while self.running:
                # Check circuit breakers
                if not self.check_circuit_breakers():
                    self.logger.info("Circuit breaker triggered - pausing...")
                    time.sleep(self.risk_limits["circuit_breakers"]["cooldown_after_losses_seconds"])
                    self.consecutive_losses = 0  # Reset after cooldown
                    continue

                # Main trading logic would go here
                # This is a placeholder for the actual implementation
                # which would:
                # 1. Fetch current market data
                # 2. Calculate indicators
                # 3. Check for entry/exit signals
                # 4. Execute trades via exchange API

                self.logger.info(f"[{datetime.now()}] Bot running - Equity: ${self.equity:.2f}, Daily P&L: ${self.daily_pnl:.2f}")

                # Sleep between iterations
                time.sleep(1)  # 1 second for 1m candles

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested...")
        finally:
            self.stop()

    def stop(self):
        """Stop the bot gracefully."""
        self.running = False
        self.logger.info("Bot stopped")
        self.logger.info(f"Final equity: ${self.equity:.2f}")
        self.logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
        self.logger.info(f"Total trades: {self.daily_trades}")


def main():
    parser = argparse.ArgumentParser(
        description="BTC Momentum Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Paper trading on Paradex
    python runbot.py --exchange paradex --ticker BTC --paper

    # Live trading on Lighter
    python runbot.py --exchange lighter --ticker BTC --quantity 2000

    # Use optimized parameters
    python runbot.py --exchange paradex --ticker BTC --config config/optimized_params.json
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
        default="config/strategy_params.json",
        help="Path to strategy config file"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (no real orders)"
    )
    parser.add_argument(
        "--quantity",
        type=float,
        default=1000,
        help="Base notional quantity per trade in USDC (default: 1000)"
    )

    args = parser.parse_args()

    # Load strategy config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print("Run the optimizer first or use default config.")
        sys.exit(1)

    # Validate environment variables for live trading
    if not args.paper:
        if args.exchange == "paradex":
            required_vars = ["PARADEX_L1_ADDRESS", "PARADEX_L2_PRIVATE_KEY"]
        else:  # lighter
            required_vars = ["LIGHTER_ACCOUNT_INDEX", "LIGHTER_API_KEY_INDEX", "API_KEY_PRIVATE_KEY"]

        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            print(f"Error: Missing environment variables for {args.exchange}:")
            for var in missing:
                print(f"  - {var}")
            print("\nSet these variables or use --paper for paper trading.")
            sys.exit(1)

    # Create and run bot
    bot = TradingBot(
        exchange=args.exchange,
        ticker=args.ticker,
        config=config,
        paper_mode=args.paper,
        quantity=args.quantity,
    )

    bot.run()


if __name__ == "__main__":
    main()
