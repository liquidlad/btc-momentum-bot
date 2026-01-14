"""
Fetch historical BTC/USDT candle data from Binance for backtesting.
Supports 1m and 5m timeframes.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

BINANCE_BASE_URL = "https://api.binance.com/api/v3"

def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> list:
    """
    Fetch klines (candlestick data) from Binance.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval ('1m', '5m', '15m', '1h', etc.)
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        limit: Max candles per request (max 1000)

    Returns:
        List of kline data
    """
    url = f"{BINANCE_BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def fetch_historical_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """
    Fetch historical candle data for the specified number of days.

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Candle interval ('1m', '5m')
        days: Number of days of historical data to fetch

    Returns:
        DataFrame with OHLCV data
    """
    all_klines = []

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Calculate milliseconds per candle
    interval_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
    }

    ms_per_candle = interval_ms.get(interval, 60 * 1000)
    current_start = start_time

    print(f"Fetching {interval} data for {symbol} ({days} days)...")

    while current_start < end_time:
        # Fetch batch
        batch_end = min(current_start + (1000 * ms_per_candle), end_time)

        try:
            klines = fetch_klines(symbol, interval, current_start, batch_end)
            if not klines:
                break

            all_klines.extend(klines)
            current_start = klines[-1][0] + ms_per_candle  # Move to next candle

            # Progress indicator
            progress = (current_start - start_time) / (end_time - start_time) * 100
            print(f"  Progress: {progress:.1f}%", end="\r")

            # Rate limiting - Binance allows 1200 requests/min
            time.sleep(0.05)

        except Exception as e:
            print(f"\nError fetching data: {e}")
            time.sleep(1)
            continue

    print(f"\nFetched {len(all_klines)} candles")

    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df["trades"] = df["trades"].astype(int)

    # Set index
    df.set_index("open_time", inplace=True)

    return df


def save_data(df: pd.DataFrame, symbol: str, interval: str):
    """Save DataFrame to CSV file."""
    os.makedirs("data", exist_ok=True)
    filename = f"data/{symbol}_{interval}.csv"
    df.to_csv(filename)
    print(f"Saved to {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical BTC candle data")
    parser.add_argument("--days", type=int, default=90, help="Number of days to fetch (default: 90)")
    parser.add_argument("--interval", type=str, default="both", choices=["1m", "5m", "both"],
                        help="Candle interval (default: both)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    args = parser.parse_args()

    symbol = args.symbol
    days = args.days

    print(f"Fetching {days} days of data for {symbol}...")
    print(f"1m data will have ~{days * 24 * 60:,} candles")
    print(f"5m data will have ~{days * 24 * 12:,} candles")
    print()

    if args.interval in ["1m", "both"]:
        df_1m = fetch_historical_data(symbol, "1m", days=days)
        save_data(df_1m, symbol, "1m")

    if args.interval in ["5m", "both"]:
        df_5m = fetch_historical_data(symbol, "5m", days=days)
        save_data(df_5m, symbol, "5m")

    print("\nData Summary:")
    if args.interval in ["1m", "both"]:
        print(f"1m candles: {len(df_1m):,} ({df_1m.index[0]} to {df_1m.index[-1]})")
    if args.interval in ["5m", "both"]:
        print(f"5m candles: {len(df_5m):,} ({df_5m.index[0]} to {df_5m.index[-1]})")
