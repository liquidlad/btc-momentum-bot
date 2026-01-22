"""
Strategy parameter optimizer.
Tests multiple indicator sets and parameter combinations to find optimal configuration.
"""

import json
import itertools
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import Backtester, DEFAULT_PARAMS
from indicators import calculate_indicators, generate_signals


def load_config(config_path: str) -> dict:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_parameter_combinations(optimization_ranges: dict) -> List[dict]:
    """Generate all combinations of parameters to test."""
    keys = list(optimization_ranges.keys())
    values = list(optimization_ranges.values())

    combinations = []
    for combo in itertools.product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def calculate_composite_score(metrics: dict, targets: dict) -> float:
    """
    Calculate composite score for strategy ranking.

    Score = (Volume_Score × 0.4) + (Profit_Score × 0.3) + (Risk_Score × 0.3)
    """
    # Volume score (trades per day vs target)
    trades_per_day = metrics.get("trades_per_day", 0)
    target_trades = targets.get("target_trades_per_day", 144)
    volume_score = min(trades_per_day / target_trades, 1.0) if target_trades > 0 else 0

    # Profit score (return vs target)
    total_return = metrics.get("total_return_pct", 0)
    target_return = targets.get("min_daily_profit_pct", 0.1) * (metrics.get("trading_days", 90))
    profit_score = min(total_return / target_return, 1.0) if target_return > 0 else 0

    # Risk score (inverse of drawdown)
    max_drawdown = metrics.get("max_drawdown_pct", 100)
    max_allowed_dd = targets.get("max_drawdown_pct", 25)
    risk_score = max(1 - (max_drawdown / max_allowed_dd), 0)

    # Composite score with weights
    composite = (volume_score * 0.4) + (profit_score * 0.3) + (risk_score * 0.3)

    return {
        "composite_score": round(composite, 4),
        "volume_score": round(volume_score, 4),
        "profit_score": round(profit_score, 4),
        "risk_score": round(risk_score, 4),
    }


def optimize_strategy(
    data_path: str,
    optimization_ranges: dict,
    capital: float = 200,
    leverage: float = 50,
    targets: dict = None
) -> List[dict]:
    """
    Run optimization across all parameter combinations.

    Args:
        data_path: Path to historical data CSV
        optimization_ranges: Dict of parameter ranges to test
        capital: Starting capital
        leverage: Max leverage
        targets: Target metrics for scoring

    Returns:
        List of results sorted by composite score
    """
    if targets is None:
        targets = {
            "target_trades_per_day": 144,
            "min_daily_profit_pct": 0.1,
            "max_drawdown_pct": 25,
        }

    # Load data
    df = pd.read_csv(data_path, parse_dates=["open_time"], index_col="open_time")
    trading_days = (df.index[-1] - df.index[0]).days or 1

    # Generate parameter combinations
    combinations = generate_parameter_combinations(optimization_ranges)
    print(f"Testing {len(combinations)} parameter combinations...")

    results = []

    for i, params in enumerate(combinations):
        # Merge with default params
        test_params = DEFAULT_PARAMS.copy()
        test_params.update(params)

        try:
            # Run backtest
            bt = Backtester(test_params, capital, leverage)
            metrics = bt.run(df)

            if "error" in metrics:
                continue

            metrics["trading_days"] = trading_days

            # Calculate composite score
            scores = calculate_composite_score(metrics, targets)
            metrics.update(scores)
            metrics["params"] = params

            results.append(metrics)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(combinations)}", end="\r")

        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    # Sort by composite score
    results.sort(key=lambda x: x["composite_score"], reverse=True)

    print(f"\nTested {len(results)} successful configurations")
    return results


def optimize_all_indicator_sets(
    data_path: str,
    strategy_config_path: str,
    capital: float = 200,
    leverage: float = 50
) -> dict:
    """
    Optimize all three indicator sets and compare.

    Returns:
        Dict with best configuration from each set and overall winner
    """
    config = load_config(strategy_config_path)
    optimization_ranges = config.get("optimization_ranges", {})

    targets = {
        "target_trades_per_day": 144,
        "min_daily_profit_pct": 0.1,
        "max_drawdown_pct": 25,
    }

    print("=" * 60)
    print("STRATEGY OPTIMIZATION")
    print("=" * 60)

    # Run optimization
    results = optimize_strategy(
        data_path,
        optimization_ranges,
        capital,
        leverage,
        targets
    )

    if not results:
        return {"error": "No valid configurations found"}

    # Get top results
    best = results[0]
    top_10 = results[:10]

    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 60)

    for i, r in enumerate(top_10):
        print(f"\n#{i+1} - Composite Score: {r['composite_score']:.4f}")
        print(f"  Trades/Day: {r['trades_per_day']:.1f}")
        print(f"  Return: {r['total_return_pct']:.2f}%")
        print(f"  Max DD: {r['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate: {r['win_rate']:.1f}%")
        print(f"  Params: {r['params']}")

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Composite Score: {best['composite_score']:.4f}")
    print(f"  - Volume Score: {best['volume_score']:.4f}")
    print(f"  - Profit Score: {best['profit_score']:.4f}")
    print(f"  - Risk Score: {best['risk_score']:.4f}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Trades: {best['total_trades']}")
    print(f"  Trades/Day: {best['trades_per_day']:.1f}")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Total Return: {best['total_return_pct']:.2f}%")
    print(f"  Max Drawdown: {best['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
    print(f"\nOptimal Parameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")

    return {
        "best": best,
        "top_10": top_10,
        "all_results": results,
    }


def save_optimized_config(best_result: dict, output_path: str):
    """Save the best configuration to a JSON file."""
    optimized = {
        "optimized_params": best_result["params"],
        "expected_metrics": {
            "trades_per_day": best_result["trades_per_day"],
            "win_rate": best_result["win_rate"],
            "profit_factor": best_result["profit_factor"],
            "total_return_pct": best_result["total_return_pct"],
            "max_drawdown_pct": best_result["max_drawdown_pct"],
            "sharpe_ratio": best_result["sharpe_ratio"],
            "composite_score": best_result["composite_score"],
        },
        "scores": {
            "volume_score": best_result["volume_score"],
            "profit_score": best_result["profit_score"],
            "risk_score": best_result["risk_score"],
        }
    }

    with open(output_path, 'w') as f:
        json.dump(optimized, f, indent=2)

    print(f"\nSaved optimized config to {output_path}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Optimize trading strategy parameters")
    parser.add_argument("--data", type=str, default="data/BTCUSDT_1m.csv",
                        help="Path to historical data CSV")
    parser.add_argument("--config", type=str, default="config/strategy_params.json",
                        help="Path to strategy config JSON")
    parser.add_argument("--capital", type=float, default=200,
                        help="Starting capital")
    parser.add_argument("--leverage", type=float, default=50,
                        help="Max leverage")
    parser.add_argument("--output", type=str, default="config/optimized_params.json",
                        help="Output path for optimized config")
    args = parser.parse_args()

    try:
        results = optimize_all_indicator_sets(
            args.data,
            args.config,
            args.capital,
            args.leverage
        )

        if "error" not in results:
            save_optimized_config(results["best"], args.output)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run fetch_historical.py first to download data.")
        sys.exit(1)
