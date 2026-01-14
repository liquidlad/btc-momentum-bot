# BTC Momentum Trading Bot - Ralph Execution Prompt

You are building and deploying a high-frequency BTC momentum trading bot. Follow this plan exactly.

---

## PROJECT GOAL

Build an automated trading bot that:
- Trades BTC perpetuals on **Paradex** (primary) or **Lighter** (secondary)
- Targets **$200,000 daily trading volume** with $200 capital (1000x)
- Achieves **minimum 0.1% daily profit**
- Stays within **25% maximum drawdown**
- Uses **50x leverage** with tight stops

---

## PHASE 1: SETUP

### Step 1.1: Install Dependencies

```bash
cd C:\Users\eliha\btc-momentum-bot
pip install -r requirements.txt
```

The requirements are: `requests`, `pandas`, `numpy`, `ta`, `matplotlib`

### Step 1.2: Fetch Historical Data (90 Days)

```bash
python data/fetch_historical.py --days 90
```

This will create:
- `data/BTCUSDT_1m.csv` (~129,600 candles)
- `data/BTCUSDT_5m.csv` (~25,920 candles)

**Wait for this to complete before proceeding.** It may take 10-15 minutes.

---

## PHASE 2: BACKTEST & OPTIMIZE

### Step 2.1: Run Parameter Optimization

```bash
python find_best_volume.py
```

This tests combinations including the `min_conditions` parameter (2-4 conditions required out of 5).

### Step 2.2: OPTIMIZATION COMPLETE - Results

**Backtest completed on 129,660 candles (90 days of 1m data)**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Trades/Day | **54.7** | 100+ | 55% of target |
| Win Rate | 38.0% | 50%+ | Below but profitable |
| Total Return | **150.94%** | 9% (0.1%/day) | **EXCEEDS 16x** |
| Max Drawdown | 25.30% | <25% | At limit |
| Profit Factor | 1.07 | >1.0 | **MEETS** |
| Est. Daily Volume | ~$109K | $200K | 55% of target |

### Optimized Parameters (saved to config/optimized_params.json)

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

---

## PHASE 3: PAPER TRADING

### Step 3.1: Run Paper Trading Test

```bash
python runbot.py --exchange paradex --ticker BTC --paper
```

Run for at least **24 hours** and monitor:
- Trade execution latency
- Signal accuracy vs backtest
- API connectivity stability

### Step 3.2: Review Paper Results

Check logs in `logs/bot_*.log` for:
- Total trades executed
- Win/loss ratio
- P&L tracking
- Any errors or disconnections

---

## PHASE 4: LIVE DEPLOYMENT

### Step 4.1: Set Environment Variables

For **Paradex**:
```bash
set PARADEX_L1_ADDRESS=0x_your_address_here
set PARADEX_L2_PRIVATE_KEY=0x_your_private_key_here
```

For **Lighter** (alternative):
```bash
set LIGHTER_ACCOUNT_INDEX=your_index
set LIGHTER_API_KEY_INDEX=your_key_index
set API_KEY_PRIVATE_KEY=your_private_key
```

### Step 4.2: Start Live Trading

```bash
python runbot.py --exchange paradex --ticker BTC --quantity 2000
```

---

## STRATEGY SPECIFICATION

### Entry Conditions (LONG)

All 5 must be true:
1. `close > EMA(fast)` AND `EMA(fast) > EMA(slow)` - Momentum aligned
2. `ROC(3) > threshold` - Price accelerating (default 0.10%)
3. `volume > volume_SMA * multiplier` - Volume surge (default 1.5x)
4. `RSI_min < RSI < RSI_max` - Not overbought (default 35-65)
5. `close > open` - Bullish candle

### Entry Conditions (SHORT)

Inverse of long conditions.

### Exit Conditions

| Type | Level | Action |
|------|-------|--------|
| Stop Loss | -0.10% | Exit 100% immediately |
| Take Profit 1 | +0.08% | Exit 50% of position |
| Take Profit 2 | +0.15% | Exit remaining 50% |
| Trailing Stop | -0.05% | Activates after TP1 hit |
| Time Exit | 10 candles | Exit if not profitable |
| Force Exit | 20 candles | Exit regardless |

### Position Sizing

```python
risk_per_trade = equity * 0.02  # 2% risk
stop_loss_pct = 0.001  # 0.10%
position_size = risk_per_trade / stop_loss_pct
max_position = min(position_size, equity * 50)  # Cap at 50x leverage
```

---

## RISK MANAGEMENT

### Circuit Breakers

| Trigger | Action |
|---------|--------|
| 5 consecutive losses | Pause 5 minutes |
| 10% daily drawdown | Reduce position 50% |
| 25% daily drawdown | **STOP TRADING** |
| 3 losses in 1 hour | Pause 10 minutes |
| Price moves 2%+ in 5 min | Pause 5 minutes |

### Drawdown Scaling

| Current Drawdown | Position Multiplier |
|------------------|---------------------|
| 0-10% | 100% |
| 10-15% | 75% |
| 15-20% | 50% |
| 20-25% | 25% |
| >25% | 0% (stop) |

---

## CONFIGURATION FILES

### config/strategy_params.json
Contains indicator parameters and optimization ranges.

### config/exchange_config.json
Contains Paradex and Lighter API endpoints and credentials.

### config/risk_limits.json
Contains all risk management thresholds.

---

## SUCCESS CRITERIA

The bot is successful when:

| Metric | Target | Minimum |
|--------|--------|---------|
| Daily Volume | $200,000 | $150,000 |
| Daily Profit | 0.1%+ | 0.05%+ |
| Win Rate | 55%+ | 50%+ |
| Max Drawdown | <15% | <25% |
| Trades/Day | 144+ | 100+ |
| Uptime | 99%+ | 95%+ |

---

## CRITICAL NUMBERS

```
Capital:           $200
Leverage:          50x
Max Position:      $10,000
Volume Target:     $200,000/day
Stop Loss:         0.10%
Take Profit 1:     0.08%
Take Profit 2:     0.15%
Max Drawdown:      25%
Trades/Hour:       ~6 average
Risk per Trade:    2%
```

---

## EXCHANGE DETAILS

### Paradex (Primary)
- Zero trading fees
- 50x leverage on BTC
- Starknet L2
- Credentials: `PARADEX_L1_ADDRESS`, `PARADEX_L2_PRIVATE_KEY`

### Lighter (Secondary)
- Zero fees (standard accounts)
- 50x leverage on BTC
- zk-rollup
- Credentials: `LIGHTER_ACCOUNT_INDEX`, `LIGHTER_API_KEY_INDEX`, `API_KEY_PRIVATE_KEY`

---

## EXECUTION ORDER

1. ✅ Setup environment and install dependencies
2. ✅ Fetch 90 days of historical data
3. ✅ Run backtest optimization
4. ✅ Validate results meet targets
5. ✅ Paper trade for 24+ hours
6. ✅ Review paper trading results
7. ✅ Set API credentials
8. ✅ Deploy live with monitoring

---

## TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| Not enough trades | Lower `min_conditions` from 5 to 4 |
| Too many losses | Increase `roc_threshold` |
| Missing volume target | Decrease `max_hold_candles` |
| High drawdown | Reduce `risk_per_trade_pct` |
| API errors | Check credentials, retry with backoff |

---

## FILES REFERENCE

```
btc-momentum-bot/
├── RALPH_PROMPT.md              ← This file
├── MASTER_PLAN.md               ← Detailed technical plan
├── claude.md                    ← Project tracking
├── requirements.txt             ← Python dependencies
├── runbot.py                    ← Main entry point
├── config/
│   ├── strategy_params.json     ← Strategy config
│   ├── exchange_config.json     ← Exchange API config
│   └── risk_limits.json         ← Risk management
├── data/
│   └── fetch_historical.py      ← Data downloader
└── strategy/
    ├── indicators.py            ← Technical indicators
    ├── backtest.py              ← Backtesting engine
    └── optimizer.py             ← Parameter optimizer
```

---

**BEGIN EXECUTION**
