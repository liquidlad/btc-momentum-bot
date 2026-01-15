#!/usr/bin/env python3
"""
Integration test for BTC Momentum Bot.
Tests that Phase 1 strategy connects properly with Phase 2 exchange clients.
"""

import asyncio
import sys
import json
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exchange.paradex_client import ParadexClient
from exchange.lighter_client import LighterClient
from exchange.base import OrderSide, OrderType, BBO
from strategy.live_strategy import LiveStrategy


async def test_paper_trading():
    """Test paper trading order flow."""
    print("=" * 60)
    print("TEST 1: Paper Trading Order Flow")
    print("=" * 60)

    client = ParadexClient(market="BTC-USD-PERP", paper_mode=True)
    connected = await client.connect()
    assert connected, "Failed to connect"
    print("Connected to Paradex (paper mode)")

    # Test BBO
    bbo = await client.get_bbo()
    assert bbo.bid_price > 0, "Invalid bid price"
    assert bbo.ask_price > 0, "Invalid ask price"
    print(f"BBO: {bbo.bid_price} / {bbo.ask_price}")

    # Test balance
    balance = await client.get_balance()
    assert balance.available == 200.0, "Initial balance should be $200"
    print(f"Balance: ${balance.available}")

    # Test buy order
    order = await client.place_order(side=OrderSide.BUY, size=0.01, order_type=OrderType.MARKET)
    assert order.status.name == "FILLED", "Order should be filled"
    print(f"Buy order filled at {order.avg_fill_price}")

    # Check position
    positions = await client.get_positions()
    assert len(positions) == 1, "Should have 1 position"
    assert positions[0].side == "LONG", "Should be long"
    print(f"Position: {positions[0].side} {positions[0].size}")

    # Test sell order to close
    order = await client.place_order(side=OrderSide.SELL, size=0.01, order_type=OrderType.MARKET)
    print(f"Sell order filled at {order.avg_fill_price}")

    # Check position closed
    positions = await client.get_positions()
    assert len(positions) == 0, "Position should be closed"
    print("Position closed")

    await client.disconnect()
    print("TEST 1 PASSED\n")


async def test_strategy_signals():
    """Test strategy signal generation with simulated candles."""
    print("=" * 60)
    print("TEST 2: Strategy Signal Generation")
    print("=" * 60)

    # Load optimized params
    with open("config/optimized_params.json") as f:
        config = json.load(f)
    strategy_params = config["optimized_params"]
    print(f"Strategy params: {strategy_params}")

    # Load risk params
    with open("config/risk_limits.json") as f:
        risk_params = json.load(f)

    # Create client and strategy
    client = ParadexClient(market="BTC-USD-PERP", paper_mode=True)
    await client.connect()

    strategy = LiveStrategy(
        exchange_client=client,
        strategy_params=strategy_params,
        risk_params=risk_params,
        candle_window=50,
    )

    print("Generating test candles (uptrend scenario)...")

    # Simulate candles - need enough for indicator warmup
    import time
    base_price = 95000
    base_time = int(time.time() * 1000)
    signals_generated = []

    for i in range(60):
        # Uptrend with volatility
        trend = i * 5
        noise = (i % 3 - 1) * 10
        price = base_price + trend + noise

        candle = {
            "timestamp": base_time + i * 60000,
            "open": price - 5,
            "high": price + 10,
            "low": price - 10,
            "close": price,
            "volume": 100 + i * 2,
        }

        signal = await strategy.on_candle_close(candle)
        if signal != 0:
            signals_generated.append((i, signal, price))

    print(f"Signals generated: {len(signals_generated)}")
    for idx, sig, price in signals_generated[:5]:
        print(f"  Candle {idx}: signal={sig}, price={price}")

    stats = strategy.get_stats()
    print(f"Final stats: equity=${stats['equity']:.2f}, trades={stats['total_trades']}")

    await client.disconnect()
    print("TEST 2 PASSED\n")


async def test_risk_management():
    """Test circuit breakers and risk management."""
    print("=" * 60)
    print("TEST 3: Risk Management")
    print("=" * 60)

    with open("config/optimized_params.json") as f:
        strategy_params = json.load(f)["optimized_params"]

    with open("config/risk_limits.json") as f:
        risk_params = json.load(f)

    client = ParadexClient(market="BTC-USD-PERP", paper_mode=True)
    await client.connect()

    strategy = LiveStrategy(
        exchange_client=client,
        strategy_params=strategy_params,
        risk_params=risk_params,
    )

    # Test position sizing
    position_size = strategy._calculate_position_size()
    print(f"Calculated position size: ${position_size:.2f}")
    assert position_size > 0, "Position size should be positive"
    assert position_size <= 10000, "Position size should not exceed max"

    # Test circuit breaker check
    can_trade = strategy._check_circuit_breakers()
    assert can_trade, "Should be able to trade initially"
    print("Circuit breakers: OK")

    # Simulate consecutive losses
    strategy.state.consecutive_losses = 5
    can_trade = strategy._check_circuit_breakers()
    assert not can_trade, "Should trigger circuit breaker after 5 losses"
    print("Consecutive loss circuit breaker: TRIGGERED")

    await client.disconnect()
    print("TEST 3 PASSED\n")


async def test_lighter_client():
    """Test Lighter client in paper mode."""
    print("=" * 60)
    print("TEST 4: Lighter Client (Paper Mode)")
    print("=" * 60)

    client = LighterClient(market="BTC-USD", paper_mode=True)
    connected = await client.connect()
    assert connected, "Failed to connect"
    print("Connected to Lighter (paper mode)")

    bbo = await client.get_bbo()
    print(f"BBO: {bbo.bid_price} / {bbo.ask_price}")

    balance = await client.get_balance()
    print(f"Balance: ${balance.available}")

    await client.disconnect()
    print("TEST 4 PASSED\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("BTC MOMENTUM BOT - INTEGRATION TESTS")
    print("Phase 1 Strategy + Phase 2 Exchange Integration")
    print("=" * 60 + "\n")

    try:
        await test_paper_trading()
        await test_strategy_signals()
        await test_risk_management()
        await test_lighter_client()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe bot is ready for paper trading.")
        print("Run: python runbot.py --exchange paradex --ticker BTC --paper")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
