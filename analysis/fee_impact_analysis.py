"""
Fee Impact Analysis for Momentum Trading Strategy

Given the 0.019% taker fee on Paradex API orders, this script analyzes:
1. Current strategy profitability
2. Break-even parameters
3. Profitable parameter combinations
"""

import numpy as np
import pandas as pd

# Constants
POSITION_SIZE = 1000  # $1000 notional per trade
FEE_RATE = 0.00019  # 0.019% per side
ROUND_TRIP_FEE = FEE_RATE * 2  # 0.038%

print("=" * 60)
print("FEE IMPACT ANALYSIS - Paradex Momentum Strategy")
print("=" * 60)

# Current parameters
current_tp = 0.0012  # 0.12%
current_sl = 0.0010  # 0.10%
current_win_rate_btc = 0.38
current_win_rate_eth = 0.40

print("\n### CURRENT STRATEGY ###")
print(f"Position Size: ${POSITION_SIZE}")
print(f"Take Profit: {current_tp*100:.2f}%")
print(f"Stop Loss: {current_sl*100:.2f}%")
print(f"Fee (per side): {FEE_RATE*100:.3f}%")
print(f"Fee (round trip): {ROUND_TRIP_FEE*100:.3f}%")

# Calculate P&L per trade
gross_win = POSITION_SIZE * current_tp
gross_loss = POSITION_SIZE * current_sl
fee_cost = POSITION_SIZE * ROUND_TRIP_FEE

net_win = gross_win - fee_cost
net_loss = gross_loss + fee_cost  # Loss is negative, fee adds to loss

print(f"\nPer Trade (current):")
print(f"  Gross Win:  +${gross_win:.2f}")
print(f"  Gross Loss: -${gross_loss:.2f}")
print(f"  Fee Cost:    ${fee_cost:.2f}")
print(f"  Net Win:    +${net_win:.2f}")
print(f"  Net Loss:   -${net_loss:.2f}")

# Expected value at current win rates
ev_btc = (current_win_rate_btc * net_win) - ((1 - current_win_rate_btc) * net_loss)
ev_eth = (current_win_rate_eth * net_win) - ((1 - current_win_rate_eth) * net_loss)

print(f"\nExpected Value per Trade:")
print(f"  BTC (38% WR): ${ev_btc:.2f}")
print(f"  ETH (40% WR): ${ev_eth:.2f}")

# Break-even win rate
# WR * net_win = (1-WR) * net_loss
# WR * net_win = net_loss - WR * net_loss
# WR * (net_win + net_loss) = net_loss
# WR = net_loss / (net_win + net_loss)
breakeven_wr = net_loss / (net_win + net_loss)
print(f"\nBreak-even Win Rate: {breakeven_wr*100:.1f}%")

print("\n" + "=" * 60)
print("### OPTION 1: INCREASE TAKE PROFIT ###")
print("=" * 60)

# What TP do we need at current win rates?
# For positive EV: WR * (TP - fee) > (1-WR) * (SL + fee)
# TP > [(1-WR) * (SL + fee) / WR] + fee

def required_tp(win_rate, stop_loss, fee):
    """Calculate minimum TP for positive expected value."""
    return ((1 - win_rate) * (stop_loss + fee) / win_rate) + fee

for wr in [0.35, 0.38, 0.40, 0.45, 0.50]:
    min_tp = required_tp(wr, current_sl, ROUND_TRIP_FEE)
    print(f"  At {wr*100:.0f}% win rate: need TP >= {min_tp*100:.2f}%")

print("\n" + "=" * 60)
print("### OPTION 2: WIDER STOPS AND TARGETS ###")
print("=" * 60)

# Test various TP/SL combinations
print("\nExpected Value per $1000 trade at 40% win rate:")
print("-" * 50)
print(f"{'TP %':<8} {'SL %':<8} {'Net Win':<10} {'Net Loss':<10} {'EV':<10}")
print("-" * 50)

test_win_rate = 0.40
for tp_pct in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    for sl_pct in [0.10, 0.15, 0.20]:
        tp = tp_pct / 100
        sl = sl_pct / 100
        nw = POSITION_SIZE * tp - fee_cost
        nl = POSITION_SIZE * sl + fee_cost
        ev = (test_win_rate * nw) - ((1 - test_win_rate) * nl)
        marker = "PROFIT" if ev > 0 else ""
        print(f"{tp_pct:<8.2f} {sl_pct:<8.2f} ${nw:<9.2f} ${nl:<9.2f} ${ev:<8.2f} {marker}")

print("\n" + "=" * 60)
print("### OPTION 3: LIMIT ORDERS (0% FEE) ###")
print("=" * 60)

print("\nWith 0% maker fee (limit orders):")
print("-" * 50)
print(f"{'TP %':<8} {'SL %':<8} {'Net Win':<10} {'Net Loss':<10} {'EV@40%':<10}")
print("-" * 50)

for tp_pct in [0.12, 0.15, 0.20]:
    for sl_pct in [0.10, 0.15]:
        tp = tp_pct / 100
        sl = sl_pct / 100
        nw = POSITION_SIZE * tp  # No fee!
        nl = POSITION_SIZE * sl  # No fee!
        ev = (test_win_rate * nw) - ((1 - test_win_rate) * nl)
        marker = "PROFIT" if ev > 0 else ""
        print(f"{tp_pct:<8.2f} {sl_pct:<8.2f} ${nw:<9.2f} ${nl:<9.2f} ${ev:<8.2f} {marker}")

print("\n" + "=" * 60)
print("### OPTION 4: REDUCE TRADE FREQUENCY ###")
print("=" * 60)

# Current: 87 trades/day, -$0.55/trade = -$47.85/day
# If we can reduce to fewer, higher-quality trades...

print("\nCurrent frequency impact:")
trades_per_day = 87
daily_loss = trades_per_day * ev_btc
print(f"  {trades_per_day} trades/day × ${ev_btc:.2f}/trade = ${daily_loss:.2f}/day")

print("\nTo break even with current parameters:")
print("  Need 0 trades/day (impossible) OR better parameters")

print("\n" + "=" * 60)
print("### RECOMMENDATIONS ###")
print("=" * 60)

print("""
PROFITABLE STRATEGIES (ranked by feasibility):

1. SWITCH TO LIMIT ORDERS (Best Option)
   - Use limit orders instead of market orders
   - 0% maker fee vs 0.019% taker fee
   - Keep current TP/SL parameters
   - Requires code changes to use limit order placement
   - Risk: Orders may not fill in fast-moving markets

2. WIDEN TARGETS (Quick Fix)
   - Increase TP to 0.25-0.30%
   - Keep SL at 0.10% or increase to 0.15%
   - At 40% WR with 0.25% TP / 0.15% SL:
     EV = 0.40 × $2.12 - 0.60 × $1.88 = +$0.72/trade
   - Trade-off: Fewer trades will hit TP, but each winner is larger

3. REDUCE FREQUENCY + HIGHER CONVICTION
   - Only trade when 4/5 or 5/5 conditions met (vs current 3/5)
   - Higher win rate trades only
   - Trade-off: Much fewer trades, but higher quality

4. HYBRID: LIMIT ORDERS + WIDER TARGETS
   - Best of both worlds
   - 0% fees + larger profit targets
   - Most complex to implement
""")

# Calculate specific recommendations
print("\n" + "=" * 60)
print("### SPECIFIC PARAMETER RECOMMENDATIONS ###")
print("=" * 60)

# Scenario 1: Keep market orders, widen targets
print("\nSCENARIO A: Market Orders + Wider Targets")
print("  TP: 0.30%, SL: 0.15%, Min Conditions: 4")
tp, sl = 0.003, 0.0015
nw = POSITION_SIZE * tp - fee_cost
nl = POSITION_SIZE * sl + fee_cost
for wr in [0.35, 0.40, 0.45]:
    ev = (wr * nw) - ((1 - wr) * nl)
    daily = ev * 30  # Assume fewer trades with higher bar
    print(f"  At {wr*100:.0f}% WR: EV=${ev:.2f}/trade, ~${daily:.2f}/day (30 trades)")

# Scenario 2: Limit orders, current targets
print("\nSCENARIO B: Limit Orders + Current Targets (0% fee)")
print("  TP: 0.12%, SL: 0.10%, Min Conditions: 3")
tp, sl = 0.0012, 0.001
nw = POSITION_SIZE * tp  # No fee
nl = POSITION_SIZE * sl  # No fee
for wr in [0.35, 0.40, 0.45]:
    ev = (wr * nw) - ((1 - wr) * nl)
    daily = ev * 87
    print(f"  At {wr*100:.0f}% WR: EV=${ev:.2f}/trade, ~${daily:.2f}/day (87 trades)")

# Scenario 3: Limit orders, wider targets
print("\nSCENARIO C: Limit Orders + Wider Targets (0% fee)")
print("  TP: 0.20%, SL: 0.12%, Min Conditions: 3")
tp, sl = 0.002, 0.0012
nw = POSITION_SIZE * tp  # No fee
nl = POSITION_SIZE * sl  # No fee
for wr in [0.35, 0.40, 0.45]:
    ev = (wr * nw) - ((1 - wr) * nl)
    daily = ev * 60  # Fewer fills with limit orders
    print(f"  At {wr*100:.0f}% WR: EV=${ev:.2f}/trade, ~${daily:.2f}/day (60 trades)")
