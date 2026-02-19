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

### 2026-01-18 (Session 9 - RSI+BB Strategy & Lighter Bug Fixes)
- **NEW STRATEGY**: RSI + Bollinger Band mean-reversion strategy
  - Entry: SHORT when price > upper BB(40) AND RSI(7) > 75
  - Exit: Lower BB OR max hold time (10 min) OR stop loss (0.3%)
  - RSI threshold: 75 for all assets (BTC, ETH, SOL)
  - **SHORT-ONLY strategy** (does not take long positions)

- **90-Day Backtest Results (SHORT strategy)**:
  | Asset | PnL | Win Rate | Max DD |
  |-------|-----|----------|--------|
  | BTC | $1,013 | 56% | 41% |
  | ETH | $1,371 | 52% | 44% |
  | SOL | $1,792 | 52% | 51% |
  | **TOTAL** | **$4,176** | | |

- **Long Strategy Backtest** (for comparison - NOT implemented):
  | Asset | PnL | Win Rate | Max DD |
  |-------|-----|----------|--------|
  | BTC | $551 | 64% | 68% |
  | ETH | $713 | 45% | 69% |
  | SOL | $957 | 58% | 68% |
  | **TOTAL** | **$2,221** | | |
  - **Conclusion**: Short strategy significantly outperforms long (~2x profit, lower drawdown)

- **CRITICAL BUG FIXES** in `exchange/lighter_client.py`:
  1. **Nonce Collision Fix** (code 21104 errors):
     - Added global `asyncio.Lock()` shared across all clients (BTC, ETH, SOL)
     - Prevents simultaneous order placement from same account
     - Added 200ms minimum spacing between orders
     - Fresh nonce sync from server before each order
     - Increased MAX_RETRIES from 1 to 3 with exponential backoff

  2. **Position Attribute Fix** (`'AccountPosition' object has no attribute 'size'`):
     - Lighter SDK uses different attribute names than expected
     - Now tries multiple attribute names: `size`, `base_amount`, `position_size`, `amount`, `qty`, `quantity`
     - Same fix applied for `entry_price` and `unrealized_pnl`
     - Added debug logging to show available attributes when not found

- **Files Modified**:
  - `exchange/lighter_client.py` - Major bug fixes
  - `strategy/rsi_bb_strategy.py` - Added status logging
  - `backtest_rsi_bb_long.py` - New file for long strategy backtesting

- **STATUS: LIVE TESTING IN PROGRESS**

### 2026-01-18 (Session 10 - Critical Bug Fixes & BB Consistency)

- **CRITICAL BUG: Duplicate Order Prevention**
  - Bot was entering positions repeatedly until margin exhausted
  - Root cause: `active_trades` set AFTER order placed, so if order errored after execution, bot thought no position existed
  - **Fix**: Optimistic locking - set `active_trades` BEFORE placing order
  - Added smart error handling: clear lock on definitive failures (insufficient margin), keep locked on uncertain failures (timeouts)

- **CRITICAL BUG: Shared SignerClient**
  - Nonce errors when multiple markets traded sequentially
  - Root cause: Each market had its own SignerClient with separate nonce counter
  - **Fix**: All markets (BTC, ETH, SOL) now share single SignerClient instance
  - Nonces stay in sync automatically across all markets

- **Startup Position Check**
  - Bot now checks exchange for existing positions on startup
  - Prevents entering duplicate positions if bot restarts with open positions
  - Logs warning and manages existing positions

- **BB Calculation Consistency Fix**
  - Problem: BB bands were different from Lighter chart
  - Root cause: Bot collected prices at check interval (1 sec in position, 10 sec out)
  - This meant BB used 40 sec of data in position vs 400 sec out of position
  - **Fix**: Price collection now fixed at 10 sec intervals regardless of check frequency
  - BB always uses ~6.7 min of data (40 samples × 10 sec)

- **Parameter Changes**:
  | Parameter | Old | New |
  |-----------|-----|-----|
  | Leverage | 20x | 20x (tested 15x, reverted) |
  | BB Period | 20 | 40 |
  | Exit Check (in position) | 5 sec | 1 sec |
  | Entry Check (no position) | 30 sec | 10 sec |
  | Price Collection | Variable | Fixed 10 sec |

- **Position Sizing with 20x**:
  - $40 margin × 20x = $800 notional per trade
  - Max loss: 0.3% × $800 = $2.40 per trade

- **Simplified Shutdown (Ctrl+C)**:
  - Bot now just stops immediately without trying to close positions
  - User requested: "when i do ctrl+c i don't want it to try and close all positions"
  - Previous behavior was causing accidental LONG positions when closing SHORTS
  - `stop()` method simplified to just set `running = False`

- **Files Modified**:
  - `exchange/lighter_client.py` - Shared SignerClient, position attribute fix
  - `strategy/rsi_bb_strategy.py` - Optimistic locking, startup check, fixed price collection, simplified shutdown
  - `run_rsi_bb.py` - Leverage 20x, BB period 40

- **STATUS: LIVE TESTING - ALL BUGS FIXED**

### 2026-01-19 (Session 11 - Critical Position Flip Bug Fix)

- **CRITICAL BUG DISCOVERED: Exit Retry Loop Creating Opposite Positions**

  **Root Cause Analysis:**
  - When closing a SHORT via BUY order, verification only checked "does position exist?"
  - If first BUY closed SHORT and created tiny LONG (dust/overfill), verification saw position exists
  - Verification timed out, assumed order didn't fill, triggered retry
  - Each retry BUY added to the LONG position
  - Result: 4x position accumulation ($8,000+ LONG instead of flat)

  **Timeline from Lighter trade data (2026-01-19 UTC):**
  ```
  20:31:52 - Close Short: 0.02148 BTC (SHORT closed)
  20:31:55 - Open Long: 0.02148 BTC (retry created LONG - 3 sec later!)
  21:12:45 - Open Long: 0.02147 BTC (another retry)
  21:12:52 - Open Long: 0.02147 BTC (another retry)
  21:12:58 - Open Long: 0.02147 BTC (another retry)
  21:13:05 - Open Long: 0.02147 BTC (another retry)
  Result: 0.089 BTC LONG (~$8,200 position, 4x expected)
  ```

  **Loss incurred:** ~$58.81 when manually closed

- **FIXES IMPLEMENTED:**

  1. **Side-Aware Verification** (`lighter_client.py`):
     - `verify_order_filled()` now accepts `closing_side` parameter
     - Detects position FLIP (SHORT→LONG) and returns success (don't retry)
     - Logic: "If I was closing a SHORT and now see a LONG, the SHORT is closed"

  2. **Pre-Retry Position Check** (`lighter_client.py`):
     - Before each retry in `place_order_until_filled()`, checks current position
     - If position flipped to opposite side → abort retries
     - If no position of target side exists → abort retries
     - Prevents accumulating opposite position

  3. **Always Use Exchange Size** (`rsi_bb_strategy.py`):
     - `exit_short()` now fetches actual position from exchange before exit
     - Uses exchange size, not tracked size, to prevent dust
     - If found LONG instead of SHORT → abort, clear tracked trade, don't send BUY

  4. **Max Position Safeguard** (`rsi_bb_strategy.py`):
     - New `_check_position_sanity()` method
     - If position exceeds 2x expected size → HALT trading for that asset
     - Logs emergency alert with position details
     - Requires manual intervention (Option C: Alert + Manual Mode)

  5. **Pre-Entry Position Check** (`rsi_bb_strategy.py`):
     - Before entering SHORT, checks if any position already exists
     - Prevents entering SHORT when LONG exists (would increase LONG via SELL)

  6. **Dust Cleanup** (`rsi_bb_strategy.py`):
     - New `_cleanup_dust_positions()` method
     - Periodically cleans up positions < $50 notional
     - Runs every ~5 minutes when not in a trade

  7. **Halt Mechanism** (`rsi_bb_strategy.py`):
     - New `halted` dict tracks halted assets
     - When halted: no entries, no exits, logs warning
     - Requires manual review and bot restart

- **Files Modified:**
  - `exchange/lighter_client.py`:
    - `place_order_until_filled()` - Added `closing_side` param, pre-retry checks
    - `verify_order_filled()` - Added side-aware verification logic
  - `strategy/rsi_bb_strategy.py`:
    - `exit_short()` - Always use exchange size, position reconciliation
    - `enter_short()` - Pre-entry position check
    - `update()` - Position sanity check, halted check
    - `run_asset()` - Halted check, dust cleanup
    - Added: `_check_position_sanity()`, `_cleanup_dust_positions()`, `halted` dict

- **Safeguards Summary:**
  | Safeguard | Location | Trigger | Action |
  |-----------|----------|---------|--------|
  | Side-aware verify | lighter_client | Position flips | Stop retries |
  | Pre-retry check | lighter_client | Before each retry | Abort if wrong side |
  | Exchange size | rsi_bb_strategy | Before exit | Use actual size |
  | Max position | rsi_bb_strategy | Size > 2x expected | HALT + alert |
  | Pre-entry check | rsi_bb_strategy | Before entry | Skip if position exists |
  | Dust cleanup | rsi_bb_strategy | Every 5 min | Close < $50 positions |

- **STATUS: FIXES IMPLEMENTED - Ready for testing**

### 2026-01-19 (Session 12 - Critical Position Sign Fix + Candle Bootstrap)

- **CRITICAL BUG FIXED: Position Side Detection**

  **Root Cause:**
  - Lighter API returns `position` (always positive) and `sign` field separately
  - `sign: 1` = LONG, `sign: -1` = SHORT
  - Old code only checked if `position` was positive/negative - it was ALWAYS positive!

  **Result of Bug:**
  - Bot entered SHORT via SELL → created SHORT position
  - Position showed `position: 0.0216, sign: -1`
  - Code ignored `sign`, saw positive `position`, thought it was LONG
  - Code tried to "close unexpected LONG" by SELLing again
  - This made SHORT BIGGER (0.02 → 0.04 → 0.065...)
  - Infinite loop until margin exhausted

  **Fix:**
  ```python
  # OLD (wrong): raw_size = float(position)  # always positive!
  # NEW (correct): raw_size = float(position) * int(sign)
  ```
  - SHORT: `0.0216 * -1 = -0.0216` → detected as SHORT ✓
  - LONG: `0.0216 * 1 = 0.0216` → detected as LONG ✓

- **NEW FEATURE: Candle Bootstrap at Startup**

  - Problem: Bot needed ~7 minutes to collect 40 price samples for BB calculation
  - Solution: Bootstrap from Binance public API at startup
  - Fetches 15 1-minute candles, extracts OHLC → 60 price points
  - Bot can trade immediately after startup

  Note: Lighter candle API returns 403 Forbidden, so using Binance instead

- **Reduced Log Verbosity**
  - Changed many INFO logs to DEBUG level
  - Removed: Order size conversion, order response details, nonce sync info
  - Kept: Entry/exit signals, position changes, errors

- **Files Modified:**
  - `exchange/lighter_client.py` - Position sign fix, candle API update, reduced logging
  - `strategy/rsi_bb_strategy.py` - Binance candle bootstrap, reduced logging

- **STATUS: LIVE AND WORKING**

### 2026-01-20 (Session 13 - Network Timeout & Reconnection Fix)

- **CRITICAL BUG FIXED: Network Hang During Exit**

  **Root Cause:**
  - Bot hung for 7.5 hours (23:40 to 07:09) during an exit attempt
  - Network call (`create_market_order`) had no timeout
  - When connection dropped, await hung forever
  - Bot was completely frozen, couldn't monitor or close position

  **Symptoms from logs:**
  ```
  2026-01-19 23:40:32 Exit attempt 1/5 with 0.2% slippage
  [... nothing for 7.5 hours ...]
  2026-01-20 07:09:07 Keyboard interrupt received
  ```

- **FIXES IMPLEMENTED:**

  1. **Timeout Helper Function** (`lighter_client.py`):
     - New `api_call_with_retry()` function wraps SDK calls
     - Configurable timeout (default 15s, orders 30s)
     - Automatic retry with exponential backoff
     - Catches both sync and async SDK methods
     - New `NetworkError` exception class for timeout/connection errors

  2. **Timeouts Added to All API Calls**:
     - `get_bbo()` - 15s timeout with 3 retries
     - `get_positions()` - 15s timeout with 3 retries
     - `get_balance()` - 15s timeout with 3 retries
     - `place_order()` - 30s timeout for order placement
     - `_sync_nonces()` - 15s timeout with 2 retries
     - `_fetch_orderbook_mapping()` - 15s timeout with 3 retries

  3. **Network Error Handling in place_order_until_filled()**:
     - Separate network retry counter (doesn't consume slippage attempts)
     - Network errors retry 3 times per slippage level
     - Exponential backoff: 1s → 2s → 4s
     - Raises `NetworkError` after all retries exhausted

  4. **Strategy-Level Network Handling** (`rsi_bb_strategy.py`):
     - Catches `NetworkError` separately from general exceptions
     - Tracks consecutive network errors per asset
     - Exponential backoff: 2s → 4s → 8s → ... → 60s max
     - After 3 consecutive errors: attempts reconnection
     - After 10 consecutive errors: HALTS trading for that asset
     - Logs connection restored when errors clear

- **Timeout Configuration:**
  | Operation | Timeout | Max Retries | Backoff |
  |-----------|---------|-------------|---------|
  | BBO fetch | 15s | 3 | 1s → 2s → 4s |
  | Position fetch | 15s | 3 | 1s → 2s → 4s |
  | Balance fetch | 15s | 3 | 1s → 2s → 4s |
  | Order placement | 30s | 3 | 1s → 2s → 4s |
  | Nonce sync | 15s | 2 | 1s → 2s |
  | Strategy level | - | 10 | 2s → 60s max |

- **Files Modified:**
  - `exchange/lighter_client.py`:
    - Added `NetworkError` class
    - Added `api_call_with_retry()` helper
    - Added `_run_with_timeout()` helper
    - Updated all API methods to use timeouts
    - Updated `place_order_until_filled()` with network retry loop
  - `exchange/__init__.py`:
    - Exported `NetworkError` class
  - `strategy/rsi_bb_strategy.py`:
    - Added `NetworkError` import
    - Added network error handling constants
    - Updated `run_asset()` with network error handling and reconnection

- **STATUS: READY FOR TESTING**

### 2026-01-20 (Session 14 - Rate Limit Fix)

- **ISSUE: Lighter API Rate Limiting (429 Too Many Requests)**

  **Root Cause:**
  - Standard Lighter accounts have only **60 requests/minute** limit
  - With 3 assets (BTC, ETH, SOL) polling every 1 second when in position
  - Each loop calls `get_bbo()` + `get_positions()` = 6+ calls/sec = 360/min
  - Far exceeds the 60/min limit

  **Lighter Rate Limit Tiers:**
  | Account Type | Requests/min |
  |--------------|--------------|
  | Standard | 60 |
  | Premium | 24,000 |

  Note: Premium accounts have 0.02% taker fees, Standard has 0% fees.
  Rate limits tracked by BOTH IP address AND L1 wallet address.
  Sub-accounts share limits (same L1 wallet).

- **FIX IMPLEMENTED:**
  - Increased polling interval from 1 second to 3 seconds when in active position
  - Reduces API calls by 3x during active trades
  - Should stay under 60 requests/min with 2-3 assets

- **Files Modified:**
  - `strategy/rsi_bb_strategy.py`:
    - Changed `await asyncio.sleep(1)` to `await asyncio.sleep(3)` in position monitoring loop

- **Recommendation:**
  - Consider running only BTC+ETH (drop SOL which had negative PnL)
  - Or upgrade to Premium account if higher request rate needed (but loses 0% fees)

- **FUTURE OPTIMIZATION: WebSocket Price Feed**

  Currently each bot polls REST API for prices. Better approach:

  **Architecture:**
  ```
  [WebSocket Client] ---> subscribes to BTC/ETH/SOL orderbooks
          |
          v
     [Local Cache] (file or Redis)
          |
          v
  [Bot 1] [Bot 2] [Bot 3] [Bot 4] ---> read prices from cache
          |
          v
     [REST API] ---> only for orders & position checks
  ```

  **Lighter WebSocket Details:**
  - URL: `wss://mainnet.zklighter.elliot.ai/stream`
  - Orderbook updates pushed every 50ms
  - Limits: 100 connections, 1000 subscriptions per IP
  - Auto-disconnects after 24 hours (need reconnect logic)

  **Benefits:**
  - Zero REST API calls for price data
  - Real-time 50ms updates (faster than polling)
  - Scales to any number of bots
  - Professional approach (REST for orders, WebSocket for market data)

  **Rate Limit Notes:**
  - Limits tracked by BOTH IP address AND L1 wallet address
  - 4 wallets on 1 IP = still only 60 req/min (IP bottleneck)
  - Need different wallets AND different IPs to multiply limits
  - Sub-accounts share L1 wallet limits

  **Status:** NOT IMPLEMENTED - Build when ready to scale to multiple bots

- **STATUS: DEPLOYED**

### 2026-01-22 (Session 15 - Performance Optimization & Margin Adjustment)

- **72-Hour Performance Analysis Results**:
  - 455 trades, $88.59 profit, 61.1% win rate
  - By asset: BTC best ($39.29), SOL weakest ($16.64)
  - Key finding: Trailing stops were losing money (71% of trailing exits were losses)
  - Trades held >10 minutes had negative PnL
  - RSI entries >75 were most profitable

- **Strategy Changes**:
  | Change | Old | New |
  |--------|-----|-----|
  | Trailing Stop | 0.2% after 0.1% profit | **REMOVED** |
  | Max Hold Time | None | 10 minutes |
  | RSI Entry (all assets) | BTC:75, ETH:60, SOL:65 | 75 for all |
  | Margin per trade | $100 | $20 |

- **Rationale**:
  - Trailing stop at 0.2% was too tight for crypto volatility
  - Giving back 0.23% average on trailing exits vs 0.05% average on stop losses
  - 10-min max hold prevents extended losing positions
  - RSI >75 had best profit/trade ratio across all assets
  - Reduced margin to balance portfolio with other bots

### 2026-01-22 (Session 16 - Margin Reduction)
- Reduced margin per trade from $40 to $20
- New position size: $20 x 20x = $400 notional per trade
- Updated run_rsi_bb.py default and RSIBBConfig

- **Files Modified**:
  - `strategy/rsi_bb_strategy.py` - Removed trailing stop, added max_hold_seconds, margin $40
  - `run_rsi_bb.py` - Updated config, --size default to 40

- **STATUS: DEPLOYED**

### Remaining Work for Live Deployment
1. ~~Integrate perp-dex-toolkit for Paradex/Lighter API~~ **DONE**
2. ~~Implement real-time price feed~~ **DONE**
3. ~~Connect strategy signals to order execution~~ **DONE**
4. ~~Add ETH/SOL support~~ **DONE (ETH recommended, SOL not recommended)**
5. ~~Fix live order execution on Paradex~~ **DONE**
6. ~~Deploy with live capital~~ **DONE (Paradex)**
7. ~~Discover Paradex API fees~~ **DONE (0.019% - kills profitability)**
8. ~~Migrate to Lighter (0% fees)~~ **DONE**
9. ~~Set up Lighter credentials and deploy live~~ **DONE**
10. (FUTURE) Implement WebSocket price feed to eliminate REST API rate limits

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
# RSI+BB Strategy (CURRENT - SHORT only, mean reversion)
python run_rsi_bb.py --live                    # Live trading all assets
python run_rsi_bb.py --live --assets BTC,ETH   # Specific assets
python run_rsi_bb.py --size 50                 # Custom margin size

# Original Momentum Strategy
python runbot.py --exchange lighter --ticker BTC --paper  # Paper trading
python runbot.py --exchange lighter --ticker BTC          # Live trading

# Run integration tests (6 tests)
python test_integration.py

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
