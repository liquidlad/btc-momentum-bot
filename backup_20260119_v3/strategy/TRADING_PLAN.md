# BTC Momentum Scalping Strategy - Trading Plan

## Strategy Overview

**Name**: Momentum Burst Scalper
**Timeframe**: 1-minute primary, 5-minute confirmation
**Style**: High-frequency momentum scalping
**Edge**: Zero fees + tight stops + volume maximization

---

## Entry Triggers

### LONG Entry Conditions (ALL must be true)

| # | Condition | Description |
|---|-----------|-------------|
| 1 | **Price Momentum** | Current candle close > EMA(9) AND EMA(9) > EMA(21) |
| 2 | **Rate of Change** | ROC(3) > 0.15% (price moved 0.15%+ in last 3 candles) |
| 3 | **Volume Surge** | Current volume > 1.5x 20-period volume SMA |
| 4 | **RSI Filter** | RSI(14) between 40 and 70 (not overbought, not oversold) |
| 5 | **Candle Confirmation** | Current candle is bullish (close > open) |

### SHORT Entry Conditions (ALL must be true)

| # | Condition | Description |
|---|-----------|-------------|
| 1 | **Price Momentum** | Current candle close < EMA(9) AND EMA(9) < EMA(21) |
| 2 | **Rate of Change** | ROC(3) < -0.15% (price dropped 0.15%+ in last 3 candles) |
| 3 | **Volume Surge** | Current volume > 1.5x 20-period volume SMA |
| 4 | **RSI Filter** | RSI(14) between 30 and 60 (not oversold, not overbought) |
| 5 | **Candle Confirmation** | Current candle is bearish (close < open) |

---

## Exit Strategy

### Take Profit (TP)

| Level | Target | Action |
|-------|--------|--------|
| TP1 | +0.20% | Close 50% of position |
| TP2 | +0.35% | Close remaining 50% OR trail stop |

### Stop Loss (SL)

| Type | Level | Description |
|------|-------|-------------|
| **Initial Stop** | -0.15% | Hard stop on entry |
| **Trailing Stop** | -0.10% | Activates after +0.15% profit |

### Time-Based Exit

| Condition | Action |
|-----------|--------|
| Position open > 10 candles (10 min on 1m) | Exit at market if not in profit |
| Position open > 20 candles | Force exit regardless |

---

## Position Sizing

### With $200 Capital at 50x Leverage

| Parameter | Calculation | Value |
|-----------|-------------|-------|
| Effective Capital | $200 x 50x | $10,000 notional |
| Risk per Trade | 2% of actual capital | $4 |
| Max Position Size | Based on 0.15% stop | $4 / 0.15% = $2,666 notional |
| Recommended Size | Conservative start | $1,000 - $2,000 notional |

### Position Size Formula
```
position_size = (capital * risk_percent) / stop_loss_percent
position_size = ($200 * 0.02) / 0.0015 = $2,666 max notional
```

---

## Risk Management

### Per-Trade Limits
- Max loss per trade: 2% of capital ($4)
- Max consecutive losses before pause: 5 trades
- Cooldown after 5 losses: 30 minutes

### Daily Limits
- Max daily loss: 10% of capital ($20)
- Max daily trades: 100 (prevents overtrading)
- Daily profit target (optional): 5% ($10)

### Circuit Breakers
| Trigger | Action |
|---------|--------|
| 3 consecutive losses | Reduce position size by 50% |
| 5 consecutive losses | Pause trading for 30 min |
| 10% daily drawdown | Stop trading for the day |
| Price moves 2%+ in 5 min | Pause for 5 min (avoid news events) |

---

## Signal Scoring (Optional Enhancement)

Instead of binary entry, score signals 1-5:

| Score | Conditions Met | Action |
|-------|----------------|--------|
| 5 | All 5 conditions | Full position |
| 4 | 4 conditions | 75% position |
| 3 | 3 conditions | 50% position |
| < 3 | < 3 conditions | No entry |

---

## Indicator Parameters

```python
STRATEGY_PARAMS = {
    # Moving Averages
    "ema_fast": 9,
    "ema_slow": 21,

    # Momentum
    "roc_period": 3,
    "roc_threshold": 0.15,  # percentage

    # Volume
    "volume_sma_period": 20,
    "volume_multiplier": 1.5,

    # RSI
    "rsi_period": 14,
    "rsi_long_min": 40,
    "rsi_long_max": 70,
    "rsi_short_min": 30,
    "rsi_short_max": 60,

    # Risk Management
    "stop_loss_pct": 0.15,
    "take_profit_1_pct": 0.20,
    "take_profit_2_pct": 0.35,
    "trailing_stop_pct": 0.10,
    "trailing_activation_pct": 0.15,

    # Position
    "risk_per_trade_pct": 2.0,
    "max_position_notional": 2666,

    # Time
    "max_candles_in_trade": 20,
    "profit_check_candles": 10,
}
```

---

## Expected Performance Metrics

### Targets (to validate in backtest)
| Metric | Target | Acceptable |
|--------|--------|------------|
| Win Rate | > 55% | > 50% |
| Avg Win / Avg Loss | > 1.2 | > 1.0 |
| Profit Factor | > 1.5 | > 1.2 |
| Max Drawdown | < 15% | < 20% |
| Trades per Day | 20-50 | 10-100 |
| Sharpe Ratio | > 2.0 | > 1.5 |

### Break-Even Analysis
With 0.15% stop and 0.20% TP1:
- Risk/Reward = 1:1.33
- Break-even win rate = 43%
- Target win rate = 55%+ for healthy edge

---

## Trade Examples

### Example LONG Trade
```
Time: 2024-01-15 14:32:00
Entry Price: $42,150
Conditions:
  - EMA(9) = $42,120 > EMA(21) = $42,080 ✓
  - ROC(3) = +0.18% ✓
  - Volume = 1.8x average ✓
  - RSI(14) = 58 ✓
  - Bullish candle ✓

Entry: LONG at $42,150
Stop Loss: $42,087 (-0.15%)
TP1: $42,234 (+0.20%) - Close 50%
TP2: $42,298 (+0.35%) - Close remaining

Outcome: TP1 hit at 14:35, trailing stop hit at $42,260 (+0.26%)
Net P&L: +$5.20 on $2,000 position
```

### Example SHORT Trade
```
Time: 2024-01-15 16:45:00
Entry Price: $41,800
Conditions:
  - EMA(9) = $41,820 < EMA(21) = $41,880 ✓
  - ROC(3) = -0.22% ✓
  - Volume = 2.1x average ✓
  - RSI(14) = 38 ✓
  - Bearish candle ✓

Entry: SHORT at $41,800
Stop Loss: $41,863 (+0.15%)
TP1: $41,716 (-0.20%)

Outcome: Stop loss hit at 16:48
Net P&L: -$3.00 on $2,000 position
```

---

## Implementation Checklist

### Phase 1: Backtest
- [ ] Fetch 30+ days of 1m BTC data
- [ ] Implement indicator calculations
- [ ] Code entry/exit logic
- [ ] Run backtest with default params
- [ ] Optimize parameters
- [ ] Validate on out-of-sample data

### Phase 2: Live Bot
- [ ] Apply for Variational API access
- [ ] Implement order execution
- [ ] Add real-time data feed
- [ ] Implement risk controls
- [ ] Paper trade for 1 week
- [ ] Go live with $200

---

## Notes

1. **Zero fees advantage**: We can enter/exit frequently without fee drag
2. **Tight stops**: -0.15% stop means we exit losers fast
3. **Volume focus**: High trade count compensates for small edge per trade
4. **Scalability**: Strategy should scale with capital increase
5. **Market conditions**: May underperform in ranging/choppy markets

---

*Strategy Version: 1.0*
*Last Updated: 2026-01-13*
