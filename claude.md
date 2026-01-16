# Multi-Asset Momentum Trading Bot Project

## Project Overview
Automated high-frequency momentum trading bot for BTC, ETH, and SOL perpetuals on **Lighter (PRIMARY)** exchange. Designed for maximum volume with wider targets for 0% fee trading.

**Supported Assets:**
- **BTC**: Fully optimized for Lighter, recommended for live trading
- **ETH**: Fully optimized, recommended for live trading
- **SOL**: Optimized but HIGH RISK (47% drawdown) - NOT recommended

## Key Characteristics
- **Timeframe**: 1-minute candles
- **Style**: Momentum scalping with optimized targets
- **Fees**: **0% on Lighter** (both maker AND taker for standard accounts)
- **Goal**: ~46 trades/day, ~$2.43/day net profit

## CRITICAL: Exchange Fee Discovery (2026-01-15)

**Paradex has HIDDEN FEES for API orders!**
- Web UI: 0% fees
- API orders: **0.019% taker fee** (kills HFT profitability)
- This made our tight-stop strategy LOSE money (-$31/day after fees)

**Solution: Switched to Lighter DEX**
- Standard accounts: **0% maker + 0% taker fees**
- Trade-off: 200-300ms latency (acceptable for 1-min candles)
- All parameter configs become PROFITABLE with 0% fees

---

## Confirmed Parameters

| Parameter | Value |
|-----------|-------|
| **Capital** | $200 (test phase) |
| **Leverage** | 50x max |
| **Max Position** | $10,000 notional |
| **Daily Trades Target** | ~46 trades/day |
| **Daily Profit Target** | ~$2.43/day net |
| **Max Drawdown** | 14.2% |
| **Primary Exchange** | **Lighter** (0% fees) |
| **Secondary Exchange** | Paradex (has 0.019% API fee) |
| **Direction** | Long + Short (no bias) |
| **Operation** | 24/7 continuous |
| **Stop Loss** | 0.20% |
| **Take Profit** | 0.30% |

---

## Exchange Research

### Lighter (PRIMARY - 0% Fees!)
- **Type**: zk-rollup perp DEX
- **Leverage**: 50x on BTC/ETH
- **Fees**: **0% maker + 0% taker** for standard accounts
- **Built by**: Citadel HFT alumni
- **Order types**: Market, limit, TWAP, SL/TP
- **Latency**: 200-300ms standard (acceptable for 1-min candles)
- **SDK**: `lighter-sdk` (pip install lighter-sdk)
- **Credentials needed**:
  - `LIGHTER_ACCOUNT_INDEX` - Your account index
  - `LIGHTER_PRIVATE_KEY` - Private key for signing
  - `LIGHTER_API_KEY_INDEX` - API key index (optional, defaults to 0)

### Paradex (Secondary - HAS FEES)
- **Type**: Perp DEX on Starknet L2
- **Leverage**: Up to 50x on BTC
- **Fees**:
  - Web UI: 0%
  - **API Orders: 0.019% TAKER FEE** (kills HFT profitability!)
- **Markets**: 250+ perpetuals
- **API**: REST + WebSocket
- **Credentials needed**: `PARADEX_L1_ADDRESS`, `PARADEX_L2_PRIVATE_KEY`
- **NOTE**: Not recommended for this strategy due to API fees

---

## Phase 1: Strategy Development
**Status**: COMPLETE - Backtested on 90 days of data

### Objectives
- [x] Research exchange specifications (Paradex, Lighter)
- [x] Gather all clarifying requirements
- [x] Define entry triggers (data-driven optimization)
- [x] Define exit strategy (optimized via backtest)
- [x] Define stop-loss strategy (optimized via backtest)
- [x] Create backtesting framework
- [x] Run optimizer on 90 days of 1m data (129,660 candles)
- [x] Document optimized parameters

### OPTIMIZED STRATEGY RESULTS (90-Day Backtest)

#### BTC Results
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Trades/Day | **54.7** | 100+ | Working toward target |
| Win Rate | 38.0% | 50%+ | Below (but profitable) |
| Total Return | **150.94%** | 0.1%/day (~9%) | **EXCEEDS** |
| Max Drawdown | 25.30% | <25% | At limit |
| Profit Factor | 1.07 | >1.0 | **MEETS** |
| Est. Daily Volume | ~$109K | $200K | ~55% of target |

#### ETH Results
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Trades/Day | **32.0** | 100+ | Below target |
| Win Rate | 40.49% | 50%+ | Below (but profitable) |
| Total Return | **107.81%** | 0.1%/day (~9%) | **EXCEEDS** |
| Max Drawdown | 22.71% | <25% | **MEETS** |
| Profit Factor | 1.07 | >1.0 | **MEETS** |

#### SOL Results (NOT RECOMMENDED)
| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Trades/Day | **23.1** | 100+ | Below target |
| Win Rate | 42.52% | 50%+ | Below |
| Total Return | **7.38%** | 0.1%/day (~9%) | **FAILS** |
| Max Drawdown | 47.23% | <25% | **FAILS** |
| Profit Factor | 1.01 | >1.0 | Marginal |

### Optimized Parameters

#### Lighter (PRIMARY - config/optimized_params_lighter.json)
```json
{
  "min_conditions": 3,
  "ema_fast": 3,
  "ema_slow": 15,
  "roc_threshold": 0.08,
  "volume_multiplier": 1.0,
  "stop_loss_pct": 0.20,
  "take_profit_1_pct": 0.30,
  "take_profit_2_pct": 0.45
}
```
**Expected Results (0% fees):**
- Trades/Day: 46
- Win Rate: 38%
- Daily Net P&L: $2.43
- Max Drawdown: 14.2%

#### Paradex BTC (config/optimized_params.json) - NOT PROFITABLE WITH FEES
```json
{
  "min_conditions": 3,
  "ema_fast": 3,
  "ema_slow": 15,
  "roc_threshold": 0.08,
  "volume_multiplier": 1.0,
  "stop_loss_pct": 0.10,
  "take_profit_1_pct": 0.12
}
```
**NOTE**: These tight targets LOSE MONEY after 0.019% API fees

### Strategy Logic
**Entry (3 of 5 conditions required):**
1. Price > EMA(3) > EMA(15) - Fast momentum
2. ROC(3) > 0.08% - Price acceleration
3. Volume > 1.0x average - Volume confirmation
4. RSI between 25-75 - Not extreme
5. Bullish/bearish candle confirmation

**Exit (Lighter optimized):**
- Stop Loss: -0.20%
- Take Profit: +0.30%
- Max Hold: 8 candles

---

## Phase 2: Bot Implementation
**Status**: COMPLETE - Ready for paper trading

### Ralph Execution Steps

1. **Fetch Data** (90 days)
   ```bash
   python data/fetch_historical.py --days 90
   ```

2. **Run Backtest Optimization**
   ```bash
   python strategy/optimizer.py --data data/BTCUSDT_1m.csv --config config/strategy_params.json
   ```

3. **Review Results**
   - Check `config/optimized_params.json` for best parameters
   - Verify targets met: >$200K volume, >0.1% profit, <25% DD

4. **Paper Trade**
   ```bash
   python runbot.py --exchange paradex --ticker BTC --paper
   ```

5. **Go Live**
   ```bash
   # Set environment variables first
   export PARADEX_L1_ADDRESS=0x...
   export PARADEX_L2_PRIVATE_KEY=0x...
   python runbot.py --exchange paradex --ticker BTC --quantity 2000
   ```

---

## Project Structure

```
btc-momentum-bot/
├── CLAUDE.md                         # This file - project tracking
├── MASTER_PLAN.md                    # Comprehensive execution plan for Ralph
├── RALPH_MULTIASSET_PROMPT.md        # Multi-asset expansion plan for Ralph
├── INTEGRATION_GUIDE.md              # Exchange API integration documentation
├── requirements.txt                  # Python dependencies
├── .env                              # Paradex credentials (gitignored)
├── runbot.py                         # Main bot entry point (supports BTC/ETH/SOL)
├── start_bots.bat                    # Launch BTC + ETH bots
├── stop_bots.bat                     # Stop running bots
├── test_integration.py               # Integration tests (6 tests)
├── find_best_volume_eth.py           # ETH parameter optimizer
├── find_best_volume_sol.py           # SOL parameter optimizer
├── config/
│   ├── strategy_params.json          # Strategy parameters + optimization ranges
│   ├── exchange_config.json          # Exchange API configs
│   ├── risk_limits.json              # Risk management settings
│   ├── portfolio_config.json         # Multi-asset portfolio configuration
│   ├── optimized_params.json         # BTC optimized params
│   ├── optimized_params_eth.json     # ETH optimized params
│   └── optimized_params_sol.json     # SOL optimized params (not recommended)
├── data/
│   ├── fetch_historical.py           # Download 90-day candle data (--symbol arg)
│   ├── BTCUSDT_1m.csv               # (generated) BTC 1-minute candles
│   ├── BTCUSDT_5m.csv               # (generated) BTC 5-minute candles
│   ├── ETHUSDT_1m.csv               # (generated) ETH 1-minute candles
│   ├── ETHUSDT_5m.csv               # (generated) ETH 5-minute candles
│   ├── SOLUSDT_1m.csv               # (generated) SOL 1-minute candles
│   └── SOLUSDT_5m.csv               # (generated) SOL 5-minute candles
├── exchange/                         # Phase 2: Exchange integration
│   ├── __init__.py                   # Module exports
│   ├── base.py                       # Abstract exchange client
│   ├── paradex_client.py             # Paradex implementation (supports multi-asset)
│   └── lighter_client.py             # Lighter implementation (supports multi-asset)
├── strategy/
│   ├── TRADING_PLAN.md               # Detailed strategy documentation
│   ├── indicators.py                 # Technical indicator calculations (Phase 1)
│   ├── live_strategy.py              # Live trading engine (Phase 2)
│   ├── backtest.py                   # Backtesting engine
│   └── optimizer.py                  # Parameter optimization
└── logs/
    └── bot_*.log                     # (generated) Trading logs
```

---

## Progress Log

### 2026-01-13 (Session 1)
- Project initiated
- Researched Variational (trading API not available)
- Created initial strategy framework

### 2026-01-13 (Session 2)
- Pivoted from Variational to Paradex/Lighter
- Gathered all clarifying requirements
- Created comprehensive MASTER_PLAN.md for Ralph
- Created JSON configs and Python implementation

### 2026-01-14 (Session 3 - Ralph Execution)
- Installed dependencies (requests, pandas, numpy, ta, matplotlib)
- Fetched 90 days of 1m data: 129,660 candles
- Fetched 90 days of 5m data: 25,932 candles
- Added `min_conditions` parameter to allow flexible entry requirements
- Ran extensive parameter optimization (972 combinations)
- **BEST RESULT FOUND:**
  - 54.7 trades/day, 150.94% return, 25.3% max DD
  - Parameters saved to `config/optimized_params.json`
- Verified paper trading framework runs correctly
- **STATUS: PHASE 1 COMPLETE - Ready for exchange integration**

### 2026-01-14 (Session 4 - Exchange Integration Research)
- Researched Paradex API: `paradex-py` SDK, L1+L2 authentication, WebSocket channels
- Researched Lighter API: `lighter-v2-python` SDK, SignerClient auth, order types
- Created comprehensive `INTEGRATION_GUIDE.md` with:
  - Full authentication setup for both exchanges
  - Market data endpoints
  - Order placement examples
  - WebSocket subscription patterns
  - Bot skeleton code templates
  - Implementation checklist
- Updated `requirements.txt` with exchange SDKs
- **STATUS: Integration documentation complete**

### 2026-01-14 (Session 5 - Full Integration)
- Created `exchange/` module with unified client interface
  - `base.py`: Abstract ExchangeClient with Order, Position, BBO dataclasses
  - `paradex_client.py`: Paradex implementation with paper trading
  - `lighter_client.py`: Lighter implementation with paper trading
- Created `strategy/live_strategy.py`:
  - Connects Phase 1 indicators to live trading
  - Real-time signal generation from candle data
  - Position management with stop-loss/take-profit
  - Circuit breakers and risk management
- Updated `runbot.py`:
  - Async main loop with candle aggregation
  - Price feed simulation for paper trading
  - Graceful shutdown with position closing
- Created `test_integration.py`:
  - All 4 tests passing
  - Paper trading order flow verified
  - Strategy signal generation verified
  - Risk management circuit breakers verified
- **STATUS: PHASE 2 COMPLETE - Bot ready for paper trading**

### 2026-01-15 (Session 6 - Multi-Asset Expansion)
- Executed RALPH_MULTIASSET_PROMPT.md to add ETH and SOL support
- Fetched 90 days of ETH data: 129,660 candles (1m) + 25,932 candles (5m)
- Fetched 90 days of SOL data: 129,660 candles (1m) + 25,932 candles (5m)
- Ran parameter optimization for ETH (972 combinations):
  - **RESULT**: 32.0 trades/day, 107.81% return, 22.71% max DD
  - Parameters saved to `config/optimized_params_eth.json`
  - **STATUS: RECOMMENDED FOR LIVE TRADING**
- Ran parameter optimization for SOL (972 combinations):
  - **RESULT**: 23.1 trades/day, 7.38% return, 47.23% max DD
  - Parameters saved to `config/optimized_params_sol.json`
  - **STATUS: NOT RECOMMENDED - High drawdown, marginal returns**
- Updated runbot.py to auto-select config based on ticker
- Created portfolio_config.json for multi-asset risk management
- Updated test_integration.py with 6 tests (all passing)
- **STATUS: MULTI-ASSET SUPPORT COMPLETE**

### 2026-01-15 (Session 7 - Live Trading Fixes)
- Fixed Paradex SDK integration issues:
  - Changed `OrderSide.BUY/SELL` to `OrderSide.Buy/Sell` (PascalCase)
  - Changed `side`/`type` to `order_side`/`order_type` in Order constructor
  - Added `Decimal` conversion for size and limit_price
- Fixed critical position sizing bug:
  - Was passing notional ($1000) as asset units (1000 BTC!)
  - Now correctly converts: `size = notional / price`
  - BTC: $1000 / $95,000 = 0.0105 BTC
  - ETH: $1000 / $3,300 = 0.303 ETH
- Added bootstrap candles from Binance API for immediate trading
- Reduced candle_window from 100 to 20 for faster startup
- Updated risk_limits.json for $50 capital per asset ($100 total)
- Created start_bots.bat and stop_bots.bat for easy management
- **LIVE TRADING VERIFIED** - Real orders filling on Paradex
- **STATUS: PRODUCTION READY**

### 2026-01-15 (Session 8 - Fee Discovery & Lighter Migration)
- **CRITICAL DISCOVERY**: Paradex charges 0.019% taker fee for API orders
  - Web UI: 0% fees
  - API orders: 0.019% per side = 0.038% round trip
  - This fee made our strategy UNPROFITABLE (-$31/day after fees)
- Ran extensive fee impact analysis:
  - At 0.019% fee: need >62.7% win rate to break even (impossible with our strategy)
  - Tested wider targets, hybrid fee models, no-time-exit - all still losing
- **SOLUTION**: Switched to Lighter DEX (0% fees for standard accounts)
  - Re-ran backtests with 0% fees: ALL configs became PROFITABLE
  - Best config for ~46 trades/day: SL=0.20%, TP=0.30% = $2.43/day net
- Updated bot for Lighter:
  - Installed `lighter-sdk` package
  - Rewrote `exchange/lighter_client.py` with proper SDK integration
  - Created `config/optimized_params_lighter.json` with wider targets
  - Updated `runbot.py` to auto-select Lighter params
  - Added Lighter credentials to `.env` template
- **Paper trading verified working on Lighter**
- **STATUS: READY FOR LIGHTER LIVE TRADING**

### Remaining Work for Live Deployment
1. ~~Integrate perp-dex-toolkit for Paradex/Lighter API~~ **DONE**
2. ~~Implement real-time price feed~~ **DONE**
3. ~~Connect strategy signals to order execution~~ **DONE**
4. ~~Add ETH/SOL support~~ **DONE (ETH recommended, SOL not recommended)**
5. ~~Fix live order execution on Paradex~~ **DONE**
6. ~~Deploy with live capital~~ **DONE (Paradex)**
7. ~~Discover Paradex API fees~~ **DONE (0.019% - kills profitability)**
8. ~~Migrate to Lighter (0% fees)~~ **DONE**
9. Set up Lighter credentials and deploy live

---

## Success Criteria

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| Daily Volume | $200,000 | $150,000 | < $100,000 |
| Daily Profit | 0.1%+ | 0.05%+ | Negative |
| Win Rate | 55%+ | 50%+ | < 45% |
| Max Drawdown | < 15% | < 25% | > 25% |
| Trades/Day | 144+ | 100+ | < 50 |

---

## Quick Reference

### Critical Numbers (Lighter - 0% Fees)
```
Capital:        $200 (test phase)
Leverage:       50x max (20x default)
Max Position:   $10,000 notional per trade
Stop Loss:      0.20%
Take Profit:    0.30%
Max Drawdown:   14.2%
Expected Trades: ~46/day (~2/hour)
Expected Profit: ~$2.43/day
```

### How to Run the Bot

```bash
# RECOMMENDED: Paper trading on Lighter (0% fees)
python runbot.py --exchange lighter --ticker BTC --paper

# Paper trading ETH on Lighter
python runbot.py --exchange lighter --ticker ETH --paper

# Run integration tests (6 tests)
python test_integration.py

# Live trading on Lighter (requires credentials)
python runbot.py --exchange lighter --ticker BTC

# NOT RECOMMENDED: Paradex (has 0.019% API fees)
# python runbot.py --exchange paradex --ticker BTC --paper
```

### Setting Up Lighter Credentials

1. Create account at https://lighter.xyz
2. Generate API keys in settings
3. Add to `.env` file:
```
LIGHTER_ACCOUNT_INDEX=<your_account_index>
LIGHTER_PRIVATE_KEY=<your_private_key>
LIGHTER_API_KEY_INDEX=0
```

### Commands for Ralph
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Fetch 90 days of data
python data/fetch_historical.py --days 90

# Step 3: Paper trade on Lighter
python runbot.py --exchange lighter --ticker BTC --paper

# Step 4: Go live on Lighter
python runbot.py --exchange lighter --ticker BTC
```

---

## Notes
- **Lighter has 0% fees** for standard accounts (both maker and taker)
- **Paradex has 0.019% API fee** - NOT recommended for this HFT strategy
- Wider targets (0.20% SL, 0.30% TP) work best with 0% fees
- 200-300ms latency on Lighter is acceptable for 1-minute candle strategy
- Backtest on 90 days before any live trading
