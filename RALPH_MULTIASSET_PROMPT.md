# Multi-Asset Momentum Trading Bot - Ralph Execution Prompt

You are expanding the existing BTC momentum trading bot to also support **Ethereum (ETH)** and **Solana (SOL)**. The BTC implementation is complete and serves as your template.

---

## EXECUTIVE SUMMARY

| Item | Value |
|------|-------|
| **Assets** | BTC (done), ETH (new), SOL (new) |
| **Capital** | $200 per asset ($600 total) |
| **Strategy** | Optimize each asset separately |
| **Risk** | Combined 25% portfolio max drawdown |
| **Exchanges** | Paradex (primary), Lighter (secondary) - same exchanges only |
| **Execution** | Sequential: ETH first, then SOL |
| **Architecture** | Single multi-asset bot |

---

## EXISTING CODEBASE REFERENCE

Before starting, understand what already exists:

```
btc-momentum-bot/
├── runbot.py                    # Main bot - WILL BE UPDATED
├── test_integration.py          # Tests - WILL BE UPDATED
├── config/
│   ├── optimized_params.json    # BTC params - REFERENCE THIS
│   ├── strategy_params.json     # Strategy config
│   ├── risk_limits.json         # Risk config - WILL BE UPDATED
│   └── exchange_config.json     # Exchange config
├── data/
│   ├── fetch_historical.py      # Data fetcher - WILL BE UPDATED
│   ├── BTCUSDT_1m.csv          # BTC data (exists)
│   └── BTCUSDT_5m.csv          # BTC data (exists)
├── exchange/
│   ├── base.py                  # Exchange interface (reuse as-is)
│   ├── paradex_client.py        # Paradex client (reuse as-is)
│   └── lighter_client.py        # Lighter client (reuse as-is)
├── strategy/
│   ├── indicators.py            # Indicators (reuse as-is)
│   ├── live_strategy.py         # Live strategy (reuse as-is)
│   ├── backtest.py              # Backtester (reuse as-is)
│   └── optimizer.py             # Optimizer (reuse as-is)
└── find_best_volume.py          # Optimization script - WILL BE UPDATED
```

**Key insight**: The strategy logic is asset-agnostic. You only need to:
1. Fetch new data for ETH and SOL
2. Run optimization to find best params for each
3. Update configs to support multi-asset
4. Update bot to track combined portfolio risk

---

## PHASE 1: ETHEREUM (ETH) ANALYSIS & OPTIMIZATION

### Step 1.1: Update Data Fetcher for Multi-Asset Support

First, update `data/fetch_historical.py` to support fetching data for any symbol.

**Read the current file first:**
```bash
cat data/fetch_historical.py
```

**Modify the file to accept a `--symbol` argument:**

The file should support:
```bash
python data/fetch_historical.py --symbol BTCUSDT --days 90
python data/fetch_historical.py --symbol ETHUSDT --days 90
python data/fetch_historical.py --symbol SOLUSDT --days 90
```

**Required changes:**
- Add `--symbol` argument (default: BTCUSDT for backward compatibility)
- Output file should be `data/{SYMBOL}_1m.csv` and `data/{SYMBOL}_5m.csv`
- Keep existing functionality intact

**Verification:**
```bash
python data/fetch_historical.py --help
# Should show --symbol option
```

- [x] **CHECKPOINT 1.1**: Data fetcher updated with --symbol support

---

### Step 1.2: Fetch ETH Historical Data (90 Days)

```bash
cd C:\Users\eliha\btc-momentum-bot
python data/fetch_historical.py --symbol ETHUSDT --days 90
```

**Expected output:**
- `data/ETHUSDT_1m.csv` (~129,600 candles)
- `data/ETHUSDT_5m.csv` (~25,920 candles)

**Verification:**
```bash
wc -l data/ETHUSDT_1m.csv
# Should be ~129,600 lines

head -5 data/ETHUSDT_1m.csv
# Should show: open_time,open,high,low,close,volume
```

- [x] **CHECKPOINT 1.2**: ETH 1m data fetched (129,600+ candles)
- [x] **CHECKPOINT 1.3**: ETH 5m data fetched (25,900+ candles)

---

### Step 1.3: Create ETH Optimizer Script

Create `find_best_volume_eth.py` by copying and modifying `find_best_volume.py`:

```bash
cp find_best_volume.py find_best_volume_eth.py
```

**Modify `find_best_volume_eth.py`:**
1. Change data file from `data/BTCUSDT_1m.csv` to `data/ETHUSDT_1m.csv`
2. Change output file from `config/optimized_params.json` to `config/optimized_params_eth.json`
3. Update print statements to say "ETH" instead of "BTC"

**Key lines to change:**
```python
# Line ~14: Change data source
df = pd.read_csv('data/ETHUSDT_1m.csv', parse_dates=['open_time'], index_col='open_time')

# Line ~108: Change output file
with open('config/optimized_params_eth.json', 'w') as f:
```

- [x] **CHECKPOINT 1.4**: ETH optimizer script created

---

### Step 1.4: Run ETH Parameter Optimization

```bash
python find_best_volume_eth.py
```

**This will test 972 parameter combinations. Expected runtime: 10-30 minutes.**

**Expected output format:**
```
Loading data...
Loaded 129660 candles
Testing 972 combinations...
Progress: 100/972
...
TOP 10 PROFITABLE HIGH-VOLUME CONFIGURATIONS
============================================================
#1 - Score: X.XXXX
  Trades/Day: XX.X
  Win Rate: XX.X%
  Return: XXX.XX%
  Max DD: XX.XX%
  Profit Factor: X.XX
  Params: {...}
```

**Record the results in this format:**

| Metric | ETH Result | Target |
|--------|------------|--------|
| Trades/Day | ___ | 50+ |
| Win Rate | ___% | 35%+ |
| Total Return | ___% | 50%+ |
| Max Drawdown | ___% | <25% |
| Profit Factor | ___ | >1.0 |

- [x] **CHECKPOINT 1.5**: ETH optimization complete
- [x] **CHECKPOINT 1.6**: Results recorded above
- [x] **CHECKPOINT 1.7**: `config/optimized_params_eth.json` created

---

### Step 1.5: Verify ETH Optimized Parameters

```bash
cat config/optimized_params_eth.json
```

**Expected structure:**
```json
{
  "optimized_params": {
    "min_conditions": X,
    "ema_fast": X,
    "ema_slow": X,
    "roc_threshold": X.XX,
    "volume_multiplier": X.X,
    "stop_loss_pct": X.XX,
    "take_profit_1_pct": X.XX
  },
  "expected_metrics": {
    "trades_per_day": XX.X,
    "win_rate": XX.XX,
    "profit_factor": X.XX,
    "total_return_pct": XXX.XX,
    "max_drawdown_pct": XX.XX
  }
}
```

- [x] **CHECKPOINT 1.8**: ETH params verified and documented

---

## PHASE 2: SOLANA (SOL) ANALYSIS & OPTIMIZATION

### Step 2.1: Fetch SOL Historical Data (90 Days)

```bash
python data/fetch_historical.py --symbol SOLUSDT --days 90
```

**Expected output:**
- `data/SOLUSDT_1m.csv` (~129,600 candles)
- `data/SOLUSDT_5m.csv` (~25,920 candles)

**Verification:**
```bash
wc -l data/SOLUSDT_1m.csv
head -5 data/SOLUSDT_1m.csv
```

- [x] **CHECKPOINT 2.1**: SOL 1m data fetched (129,600+ candles)
- [x] **CHECKPOINT 2.2**: SOL 5m data fetched (25,900+ candles)

---

### Step 2.2: Create SOL Optimizer Script

```bash
cp find_best_volume.py find_best_volume_sol.py
```

**Modify `find_best_volume_sol.py`:**
1. Change data file to `data/SOLUSDT_1m.csv`
2. Change output file to `config/optimized_params_sol.json`
3. Update print statements to say "SOL"

- [x] **CHECKPOINT 2.3**: SOL optimizer script created

---

### Step 2.3: Run SOL Parameter Optimization

```bash
python find_best_volume_sol.py
```

**Record the results:**

| Metric | SOL Result | Target |
|--------|------------|--------|
| Trades/Day | ___ | 50+ |
| Win Rate | ___% | 35%+ |
| Total Return | ___% | 50%+ |
| Max Drawdown | ___% | <25% |
| Profit Factor | ___ | >1.0 |

- [x] **CHECKPOINT 2.4**: SOL optimization complete
- [x] **CHECKPOINT 2.5**: Results recorded above
- [x] **CHECKPOINT 2.6**: `config/optimized_params_sol.json` created

---

### Step 2.4: Verify SOL Optimized Parameters

```bash
cat config/optimized_params_sol.json
```

- [x] **CHECKPOINT 2.7**: SOL params verified and documented

---

## PHASE 3: EXCHANGE VERIFICATION

### Step 3.1: Verify Paradex Market Support

Check if Paradex supports ETH and SOL perpetuals.

**Research the Paradex markets:**
- ETH market symbol: `ETH-USD-PERP`
- SOL market symbol: `SOL-USD-PERP`

**Verification method:**
```python
# Quick test script
import requests
response = requests.get("https://api.prod.paradex.trade/v1/markets")
markets = response.json()
for m in markets:
    if "ETH" in m.get("symbol", "") or "SOL" in m.get("symbol", ""):
        print(m["symbol"])
```

**Document findings:**
- [x] ETH-USD-PERP available on Paradex: YES
- [x] SOL-USD-PERP available on Paradex: YES

---

### Step 3.2: Verify Lighter Market Support

Check if Lighter supports ETH and SOL perpetuals.

**Expected market symbols:**
- ETH: `ETH-USD`
- SOL: `SOL-USD`

**Document findings:**
- [x] ETH-USD available on Lighter: YES
- [x] SOL-USD available on Lighter: YES

---

### Step 3.3: Handle Missing Markets

**If SOL is NOT supported on either exchange:**
1. Document which exchange(s) don't support SOL
2. Create `config/optimized_params_sol.json` anyway (for future use)
3. Add a note in the config: `"exchange_support": false`
4. The bot should gracefully skip SOL if exchange doesn't support it

**If ETH is NOT supported:**
- This is unlikely as ETH is major. Document and report.

- [x] **CHECKPOINT 3.1**: Exchange support documented
- [x] **CHECKPOINT 3.2**: Fallback handling implemented (if needed)

---

## PHASE 4: MULTI-ASSET BOT INTEGRATION

### Step 4.1: Create Portfolio Configuration

Create `config/portfolio_config.json`:

```json
{
  "portfolio": {
    "total_capital": 600,
    "max_portfolio_drawdown_pct": 25,
    "assets": {
      "BTC": {
        "enabled": true,
        "capital": 200,
        "config_file": "config/optimized_params.json",
        "paradex_symbol": "BTC-USD-PERP",
        "lighter_symbol": "BTC-USD"
      },
      "ETH": {
        "enabled": true,
        "capital": 200,
        "config_file": "config/optimized_params_eth.json",
        "paradex_symbol": "ETH-USD-PERP",
        "lighter_symbol": "ETH-USD"
      },
      "SOL": {
        "enabled": true,
        "capital": 200,
        "config_file": "config/optimized_params_sol.json",
        "paradex_symbol": "SOL-USD-PERP",
        "lighter_symbol": "SOL-USD"
      }
    }
  },
  "risk_management": {
    "combined_drawdown_tracking": true,
    "pause_all_on_portfolio_drawdown": true,
    "correlation_rules": "none"
  }
}
```

- [x] **CHECKPOINT 4.1**: Portfolio config created

---

### Step 4.2: Update Risk Limits for Portfolio

Update `config/risk_limits.json` to add portfolio-level risk:

**Add this section to the existing config:**
```json
{
  "portfolio": {
    "enabled": true,
    "max_combined_drawdown_pct": 25,
    "pause_all_assets_on_breach": true
  }
}
```

- [x] **CHECKPOINT 4.2**: Risk limits updated for portfolio

---

### Step 4.3: Update runbot.py for Multi-Asset Support

The current `runbot.py` already supports `--ticker` flag. Verify it works with ETH and SOL:

**Test commands (paper mode):**
```bash
# These should work with existing code
python runbot.py --exchange paradex --ticker ETH --paper
python runbot.py --exchange paradex --ticker SOL --paper
```

**If the bot doesn't support ETH/SOL correctly, update:**
1. Market symbol mapping in `MomentumBot.__init__`
2. Config file loading to use asset-specific params

**Required changes to `runbot.py`:**

```python
# In MomentumBot.__init__, add config file mapping:
def __init__(self, ...):
    # Map ticker to config file
    config_map = {
        "BTC": "config/optimized_params.json",
        "ETH": "config/optimized_params_eth.json",
        "SOL": "config/optimized_params_sol.json",
    }
    self.config_file = config_map.get(ticker, "config/optimized_params.json")
```

```python
# In main_async(), load the correct config:
config_file = {
    "BTC": "config/optimized_params.json",
    "ETH": "config/optimized_params_eth.json",
    "SOL": "config/optimized_params_sol.json",
}.get(args.ticker, "config/optimized_params.json")

config = load_config(config_file)
```

- [x] **CHECKPOINT 4.3**: runbot.py updated for multi-asset

---

### Step 4.4: Add Portfolio Risk Tracking

Update `strategy/live_strategy.py` to support combined portfolio drawdown tracking.

**Option A (Simple):** Track drawdown per-asset but check combined total.

**Option B (Advanced):** Create a `PortfolioManager` class that monitors all assets.

**For this phase, implement Option A:**

In `LiveStrategy.__init__`, add:
```python
self.portfolio_equity = 600  # Total across all assets
self.portfolio_peak = 600
```

In `_check_circuit_breakers`, add portfolio check:
```python
# Combined portfolio drawdown check
portfolio_dd = ((self.portfolio_peak - self.portfolio_equity) / self.portfolio_peak) * 100
if portfolio_dd >= 25:  # Combined 25% limit
    logger.warning(f"Portfolio circuit breaker: {portfolio_dd:.1f}% combined drawdown")
    return False
```

- [x] **CHECKPOINT 4.4**: Portfolio risk tracking added

---

## PHASE 5: TESTING & VERIFICATION

### Step 5.1: Update Integration Tests

Update `test_integration.py` to test all three assets.

**Add new test functions:**
```python
async def test_eth_paper_trading():
    """Test ETH paper trading."""
    client = ParadexClient(market="ETH-USD-PERP", paper_mode=True)
    # ... same tests as BTC

async def test_sol_paper_trading():
    """Test SOL paper trading."""
    client = ParadexClient(market="SOL-USD-PERP", paper_mode=True)
    # ... same tests as BTC

async def test_eth_strategy_signals():
    """Test ETH strategy with ETH-optimized params."""
    with open("config/optimized_params_eth.json") as f:
        strategy_params = json.load(f)["optimized_params"]
    # ... same tests as BTC

async def test_sol_strategy_signals():
    """Test SOL strategy with SOL-optimized params."""
    with open("config/optimized_params_sol.json") as f:
        strategy_params = json.load(f)["optimized_params"]
    # ... same tests as BTC
```

- [x] **CHECKPOINT 5.1**: Integration tests updated for ETH
- [x] **CHECKPOINT 5.2**: Integration tests updated for SOL

---

### Step 5.2: Run All Integration Tests

```bash
python test_integration.py
```

**Expected output:**
```
============================================================
BTC MOMENTUM BOT - INTEGRATION TESTS
============================================================
TEST 1: BTC Paper Trading Order Flow - PASSED
TEST 2: BTC Strategy Signal Generation - PASSED
TEST 3: Risk Management - PASSED
TEST 4: Lighter Client - PASSED
TEST 5: ETH Paper Trading Order Flow - PASSED
TEST 6: ETH Strategy Signal Generation - PASSED
TEST 7: SOL Paper Trading Order Flow - PASSED
TEST 8: SOL Strategy Signal Generation - PASSED
============================================================
ALL TESTS PASSED!
============================================================
```

- [x] **CHECKPOINT 5.3**: All integration tests pass

---

### Step 5.3: Paper Trade Each Asset

Run each asset in paper mode for at least 5 minutes to verify:

```bash
# Terminal 1: BTC
python runbot.py --exchange paradex --ticker BTC --paper

# Terminal 2: ETH
python runbot.py --exchange paradex --ticker ETH --paper

# Terminal 3: SOL
python runbot.py --exchange paradex --ticker SOL --paper
```

**Verify for each:**
- [x] BTC bot starts without errors
- [x] ETH bot starts without errors
- [x] SOL bot starts without errors
- [x] Each loads correct optimized params
- [x] Price feeds working
- [x] No crashes after 5 minutes

- [x] **CHECKPOINT 5.4**: Paper trading verified for all assets

---

## PHASE 6: DOCUMENTATION & CLEANUP

### Step 6.1: Update claude.md

Add a new section to `claude.md` documenting the multi-asset expansion:

```markdown
## Multi-Asset Expansion (ETH + SOL)

### ETH Results (90-Day Backtest)
| Metric | Result |
|--------|--------|
| Trades/Day | [FILL IN] |
| Win Rate | [FILL IN]% |
| Total Return | [FILL IN]% |
| Max Drawdown | [FILL IN]% |

### SOL Results (90-Day Backtest)
| Metric | Result |
|--------|--------|
| Trades/Day | [FILL IN] |
| Win Rate | [FILL IN]% |
| Total Return | [FILL IN]% |
| Max Drawdown | [FILL IN]% |

### Portfolio Configuration
- Total Capital: $600 ($200 per asset)
- Combined Max Drawdown: 25%
- Correlation Rules: None (independent trading)
```

- [x] **CHECKPOINT 6.1**: claude.md updated with ETH/SOL results

---

### Step 6.2: Update Project Structure in claude.md

Update the project structure to reflect new files:

```
btc-momentum-bot/
├── ...existing files...
├── find_best_volume_eth.py       # ETH optimizer
├── find_best_volume_sol.py       # SOL optimizer
├── config/
│   ├── optimized_params.json     # BTC params
│   ├── optimized_params_eth.json # ETH params (NEW)
│   ├── optimized_params_sol.json # SOL params (NEW)
│   └── portfolio_config.json     # Multi-asset config (NEW)
├── data/
│   ├── BTCUSDT_1m.csv
│   ├── ETHUSDT_1m.csv           # (NEW)
│   └── SOLUSDT_1m.csv           # (NEW)
```

- [x] **CHECKPOINT 6.2**: Project structure documented

---

### Step 6.3: Commit and Push

```bash
git add -A
git commit -m "Add ETH and SOL support with optimized parameters

Multi-asset expansion:
- ETH: [X] trades/day, [X]% return, [X]% max DD
- SOL: [X] trades/day, [X]% return, [X]% max DD

New files:
- config/optimized_params_eth.json
- config/optimized_params_sol.json
- config/portfolio_config.json
- find_best_volume_eth.py
- find_best_volume_sol.py
- data/ETHUSDT_1m.csv
- data/SOLUSDT_1m.csv

Updated:
- runbot.py: Multi-asset config loading
- test_integration.py: ETH/SOL tests
- claude.md: Documentation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push origin master
```

- [x] **CHECKPOINT 6.3**: Changes committed and pushed

---

## SUCCESS CRITERIA SUMMARY

### Per-Asset Metrics (Each should meet these)
| Metric | Target | Minimum |
|--------|--------|---------|
| Trades/Day | 50+ | 30+ |
| Win Rate | 38%+ | 30%+ |
| Total Return (90d) | 100%+ | 50%+ |
| Max Drawdown | <25% | <30% |
| Profit Factor | >1.0 | >1.0 |

### Portfolio Metrics
| Metric | Target |
|--------|--------|
| Total Capital | $600 |
| Combined Max DD | 25% |
| All Tests Pass | YES |

---

## TROUBLESHOOTING

### Issue: Data fetch fails for ETH/SOL
**Solution:** Check Binance API status, try with fewer days:
```bash
python data/fetch_historical.py --symbol ETHUSDT --days 30
```

### Issue: Optimization finds no profitable configs
**Solution:** Expand parameter ranges in the optimizer script:
```python
param_ranges = {
    'min_conditions': [2, 3, 4],  # Try looser conditions
    'stop_loss_pct': [0.15, 0.20, 0.25],  # Wider stops for volatile assets
}
```

### Issue: Exchange doesn't support SOL
**Solution:**
1. Set `"enabled": false` in portfolio_config.json for SOL
2. Document in claude.md
3. Proceed with BTC + ETH only

### Issue: Bot crashes with ETH/SOL
**Solution:** Check market symbol mapping in exchange clients:
```python
# In paradex_client.py, verify market format
# ETH should be: ETH-USD-PERP
# SOL should be: SOL-USD-PERP
```

---

## FINAL CHECKLIST

Before marking complete, verify:

- [x] ETH data fetched (90 days)
- [x] ETH optimization complete with results documented
- [x] SOL data fetched (90 days)
- [x] SOL optimization complete with results documented
- [x] Exchange support verified for both assets
- [x] Portfolio config created
- [x] runbot.py works with --ticker ETH and --ticker SOL
- [x] All integration tests pass
- [x] Paper trading works for all three assets
- [x] claude.md updated with all results
- [ ] Code committed and pushed to GitHub

---

## EXECUTION ORDER SUMMARY

1. **Phase 1** (ETH): Update data fetcher → Fetch ETH data → Create ETH optimizer → Run optimization → Verify results
2. **Phase 2** (SOL): Fetch SOL data → Create SOL optimizer → Run optimization → Verify results
3. **Phase 3** (Exchanges): Verify Paradex/Lighter support for ETH and SOL
4. **Phase 4** (Integration): Create portfolio config → Update runbot.py → Add portfolio risk tracking
5. **Phase 5** (Testing): Update tests → Run all tests → Paper trade verification
6. **Phase 6** (Documentation): Update claude.md → Commit → Push

---

**BEGIN EXECUTION**
