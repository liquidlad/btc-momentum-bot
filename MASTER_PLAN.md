# BTC Momentum Trading Bot - Master Plan

## Executive Summary

Build a high-frequency momentum trading bot for BTC perpetuals on Paradex (primary) and Lighter (secondary) exchanges. The bot will maximize trading volume while maintaining profitability, targeting 1000x daily volume relative to margin capital.

---

## Project Specifications

### Core Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Primary Exchange** | Paradex | Zero fees, 50x leverage, Starknet L2 |
| **Secondary Exchange** | Lighter | Zero fees (standard), 50x leverage, zk-rollup |
| **Starting Capital** | $200 | Test phase |
| **Max Leverage** | 50x | $10,000 max position |
| **Daily Volume Target** | $200,000 | 1000x capital |
| **Min Daily Profit Target** | 0.1% ($0.20) | After all costs |
| **Max Drawdown** | 25% ($50) | Circuit breaker |
| **Direction** | Long + Short | No directional bias |
| **Timeframes** | 1-minute, 5-minute | Primary analysis |
| **Data Requirement** | 90+ days | For backtesting |

### Volume Calculation

```
Capital: $200
Max Position (50x): $10,000
Daily Volume Target: $200,000

Required Round Trips: $200,000 / ($10,000 × 2) = 10 full position round trips
OR: 20+ smaller trades throughout the day

Trades per Hour (24h): ~0.83 full round trips/hour minimum
```

---

## Phase 1: Data Collection & Analysis

### 1.1 Data Sources

Fetch 90+ days of historical 1m and 5m BTC/USDC data from:
1. **Binance** - Highest quality, most liquid
2. **Paradex API** - For exchange-specific price action
3. **Lighter API** - For exchange-specific price action

### 1.2 Data Schema

```python
CANDLE_SCHEMA = {
    "timestamp": "datetime",      # Candle open time
    "open": "float",              # Open price
    "high": "float",              # High price
    "low": "float",               # Low price
    "close": "float",             # Close price
    "volume": "float",            # Base volume
    "quote_volume": "float",      # Quote volume (USDC)
    "trades": "int",              # Number of trades
    "taker_buy_volume": "float",  # Aggressive buy volume
}
```

### 1.3 Analysis Tasks

Ralph must perform the following analysis on 90-day data:

#### A. Indicator Optimization
Test each indicator set and record performance metrics:

**Set 1: EMA + RSI + Volume**
- EMA periods: [5,9], [9,21], [12,26]
- RSI period: [7, 14, 21]
- RSI thresholds: Long [35-65], Short [35-65]
- Volume multiplier: [1.2, 1.5, 2.0]

**Set 2: VWAP + Deviation**
- VWAP anchor: Daily reset
- Entry deviation: [0.1%, 0.2%, 0.3%] from VWAP
- Exit deviation: [0.05%, 0.1%, 0.15%] reversion

**Set 3: Pure Price Action**
- Breakout periods: [5, 10, 20] candles
- ATR multiplier for stops: [0.5, 1.0, 1.5]
- Candle pattern filters: engulfing, momentum bars

#### B. Metrics to Calculate for Each Strategy

```python
REQUIRED_METRICS = [
    "total_trades",
    "win_rate",
    "profit_factor",
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe_ratio",
    "avg_trade_duration_candles",
    "trades_per_day",
    "daily_volume_generated",
    "avg_win_pct",
    "avg_loss_pct",
    "largest_win",
    "largest_loss",
    "consecutive_wins_max",
    "consecutive_losses_max",
]
```

#### C. Selection Criteria

Rank strategies by composite score:
```
Score = (Volume_Score × 0.4) + (Profit_Score × 0.3) + (Risk_Score × 0.3)

Where:
- Volume_Score = daily_trades / target_trades (capped at 1.0)
- Profit_Score = total_return / target_return (capped at 1.0)
- Risk_Score = 1 - (max_drawdown / max_allowed_drawdown)
```

---

## Phase 2: Strategy Definition

### 2.1 Entry Logic (Template)

The final entry logic will be determined by Phase 1 backtesting. Template structure:

```python
def check_long_entry(candle, indicators):
    conditions = {
        "momentum": indicators["ema_fast"] > indicators["ema_slow"],
        "acceleration": indicators["roc"] > PARAMS["roc_threshold"],
        "volume": candle["volume"] > indicators["volume_avg"] * PARAMS["vol_mult"],
        "filter": PARAMS["rsi_min"] < indicators["rsi"] < PARAMS["rsi_max"],
        "confirmation": candle["close"] > candle["open"],  # Bullish candle
    }

    score = sum(conditions.values())
    return score >= PARAMS["min_conditions"], score

def check_short_entry(candle, indicators):
    # Inverse conditions
    ...
```

### 2.2 Exit Logic

**Stop Loss (TIGHT - prioritize volume over holding)**
```python
STOP_LOSS_PCT = 0.10  # 0.10% initial stop (very tight)
# With 50x leverage, 0.10% price move = 5% position P&L
```

**Take Profit (QUICK - exit fast to recycle capital)**
```python
TAKE_PROFIT_1_PCT = 0.08  # Close 50% at +0.08%
TAKE_PROFIT_2_PCT = 0.15  # Close remaining 50% at +0.15%
# OR use trailing stop after TP1
```

**Time-Based Exit**
```python
MAX_HOLD_CANDLES = 10  # Exit after 10 candles (10 min on 1m)
# Ensures capital recycling for volume
```

### 2.3 Position Sizing

```python
def calculate_position_size(equity, leverage=50):
    """
    Size positions to hit volume target while managing risk.

    With $200 and 50x, we have $10,000 max notional.
    To hit $200,000 daily volume:
    - Need 20 round trips of $10,000 each
    - Or 40 round trips of $5,000 each
    - Or continuous smaller trades
    """
    risk_per_trade = equity * 0.02  # 2% risk per trade
    stop_loss_pct = 0.001  # 0.10%

    # Position size based on risk
    risk_based_size = risk_per_trade / stop_loss_pct

    # Cap at leverage limit
    max_size = equity * leverage

    return min(risk_based_size, max_size)
```

---

## Phase 3: Implementation

### 3.1 Codebase Structure

Fork and extend [perp-dex-toolkit](https://github.com/earthskyorg/perp-dex-toolkit):

```
btc-momentum-bot/
├── config/
│   ├── strategy_params.json      # Optimized parameters
│   ├── exchange_config.json      # API keys, endpoints
│   └── risk_limits.json          # Drawdown, position limits
├── data/
│   ├── fetch_historical.py       # Download 90-day data
│   ├── BTCUSDC_1m.csv           # 1-minute candles
│   └── BTCUSDC_5m.csv           # 5-minute candles
├── strategy/
│   ├── indicators.py             # Technical indicators
│   ├── signals.py                # Entry/exit signal generation
│   ├── backtest.py               # Backtesting engine
│   └── optimizer.py              # Parameter optimization
├── bot/
│   ├── paradex_client.py         # Paradex API wrapper
│   ├── lighter_client.py         # Lighter API wrapper
│   ├── order_manager.py          # Order execution
│   ├── position_manager.py       # Position tracking
│   └── risk_manager.py           # Circuit breakers
├── runbot.py                     # Main entry point
├── requirements.txt
├── MASTER_PLAN.md               # This document
└── claude.md                     # Progress tracking
```

### 3.2 Exchange Integration

**Paradex Setup:**
```bash
# Environment variables required
PARADEX_L1_ADDRESS=0x...
PARADEX_L2_PRIVATE_KEY=0x...  # From Profile → Wallet
```

**Lighter Setup:**
```bash
# Environment variables required
LIGHTER_ACCOUNT_INDEX=...
LIGHTER_API_KEY_INDEX=...
API_KEY_PRIVATE_KEY=...
```

### 3.3 Order Types to Use

| Order Type | Use Case |
|------------|----------|
| **Limit (Post-Only)** | Primary entry - captures maker rebates if available |
| **Market** | Fast exit on stop-loss hit |
| **Stop-Loss** | Automated risk management |
| **Take-Profit** | Automated profit capture |

---

## Phase 4: Risk Management

### 4.1 Circuit Breakers

```python
RISK_LIMITS = {
    # Per-trade limits
    "max_position_pct": 100,      # % of max leverage to use
    "stop_loss_pct": 0.10,        # 0.10% hard stop

    # Session limits
    "max_consecutive_losses": 5,  # Pause after 5 losses
    "cooldown_after_losses": 300, # 5 min cooldown

    # Daily limits
    "max_daily_drawdown_pct": 25, # Stop trading at 25% DD
    "max_daily_trades": 500,      # Prevent runaway trading
    "min_daily_profit_pct": -10,  # Warning threshold

    # Position limits
    "max_open_positions": 1,      # One position at a time
    "max_notional": 10000,        # $10,000 max
}
```

### 4.2 Drawdown Recovery

```python
def adjust_for_drawdown(current_drawdown, base_position_size):
    """Scale down position size as drawdown increases."""
    if current_drawdown < 10:
        return base_position_size * 1.0
    elif current_drawdown < 15:
        return base_position_size * 0.75
    elif current_drawdown < 20:
        return base_position_size * 0.50
    elif current_drawdown < 25:
        return base_position_size * 0.25
    else:
        return 0  # Stop trading
```

---

## Phase 5: Execution Checklist

### Pre-Launch Checklist

- [ ] **Data**: Download 90+ days of 1m and 5m BTC data
- [ ] **Backtest**: Run all three indicator sets through backtester
- [ ] **Select**: Choose best-performing strategy based on composite score
- [ ] **Optimize**: Fine-tune parameters using walk-forward optimization
- [ ] **Validate**: Confirm strategy meets targets:
  - [ ] Daily volume ≥ $200,000
  - [ ] Daily profit ≥ 0.1%
  - [ ] Max drawdown ≤ 25%
- [ ] **Setup**: Configure Paradex API credentials
- [ ] **Test**: Run paper trading for 24-48 hours
- [ ] **Deploy**: Start live trading with $200

### Daily Operations Checklist

- [ ] Check daily P&L and drawdown
- [ ] Verify volume target progress
- [ ] Monitor for API errors or disconnections
- [ ] Review largest wins/losses
- [ ] Adjust parameters if market regime changes

---

## Success Criteria

### Minimum Viable Performance

| Metric | Target | Acceptable | Failure |
|--------|--------|------------|---------|
| Daily Volume | $200,000 | $150,000 | < $100,000 |
| Daily Profit | 0.1%+ | 0.05%+ | Negative |
| Win Rate | 55%+ | 50%+ | < 45% |
| Max Drawdown | < 15% | < 25% | > 25% |
| Uptime | 99%+ | 95%+ | < 90% |

### Volume Breakdown Target

```
24 hours × 60 minutes = 1,440 minutes
Target volume: $200,000
Average per minute: $138.89

With 10-candle average hold time:
- ~144 trades per day
- ~$1,389 average trade size
- ~6 trades per hour
```

---

## Appendix A: API References

### Paradex
- Docs: https://docs.paradex.trade/
- API: REST + WebSocket
- Leverage: Up to 50x
- Fees: Zero for retail

### Lighter
- Docs: https://docs.lighter.xyz/
- API: https://apidocs.lighter.xyz/
- Leverage: 50x for BTC
- Fees: Zero for standard accounts

### perp-dex-toolkit
- Repo: https://github.com/earthskyorg/perp-dex-toolkit
- Supports: Paradex, Lighter, and 5 other exchanges
- Features: Limit order strategy, configurable TP, multi-exchange

---

## Appendix B: Quick Reference

### Critical Numbers
```
Capital:        $200
Leverage:       50x
Max Position:   $10,000
Volume Target:  $200,000/day (1000x)
Stop Loss:      0.10%
Take Profit:    0.08% / 0.15%
Max Drawdown:   25% ($50)
```

### Command Reference
```bash
# Fetch historical data
python data/fetch_historical.py --days 90 --interval 1m

# Run backtest
python strategy/backtest.py --config config/strategy_params.json

# Start bot (paper trading)
python runbot.py --exchange paradex --ticker BTC --paper

# Start bot (live)
python runbot.py --exchange paradex --ticker BTC --quantity 1000
```

---

*Document Version: 1.0*
*Created: 2026-01-13*
*For: Ralph Plugin Execution*
