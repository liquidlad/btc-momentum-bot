# Exchange Integration Guide

Complete API integration documentation for Paradex (primary) and Lighter (secondary) exchanges.

---

## Table of Contents

1. [Paradex Integration](#paradex-integration)
2. [Lighter Integration](#lighter-integration)
3. [Implementation Checklist](#implementation-checklist)

---

## Paradex Integration

### Overview
- **Type**: Zero-fee perpetual DEX on Starknet L2
- **Leverage**: Up to 50x on BTC
- **Fees**: Zero for retail accounts
- **SDK**: `paradex-py` (Python)

### Installation

```bash
pip install paradex-py
```

### Authentication

Paradex supports two authentication methods:

#### Method 1: L1 + L2 Traditional (Full Access)
Use when you have both Ethereum and Starknet credentials:

```python
from paradex_py import Paradex
from paradex_py.environment import Environment

paradex = Paradex(
    env=Environment.PROD,  # or Environment.TESTNET
    l1_address="0x_your_eth_address",
    l1_private_key="0x_your_eth_private_key"
)
```

#### Method 2: L2-Only Subkey (Recommended for Bots)
Use when account is already onboarded - no L1 needed:

```python
from paradex_py import ParadexSubkey
from paradex_py.environment import Environment

paradex = ParadexSubkey(
    env=Environment.PROD,
    l2_private_key="0x_your_l2_private_key",
    l2_address="0x_your_l2_address"
)
```

### Environment Variables

```bash
# Windows (set)
set PARADEX_L1_ADDRESS=0x_your_eth_address
set PARADEX_L1_PRIVATE_KEY=0x_your_eth_private_key
set PARADEX_L2_ADDRESS=0x_your_l2_address
set PARADEX_L2_PRIVATE_KEY=0x_your_l2_private_key

# Linux/Mac (export)
export PARADEX_L1_ADDRESS=0x_your_eth_address
export PARADEX_L1_PRIVATE_KEY=0x_your_eth_private_key
export PARADEX_L2_ADDRESS=0x_your_l2_address
export PARADEX_L2_PRIVATE_KEY=0x_your_l2_private_key
```

### Market Data (Public - No Auth Required)

```python
# Get all markets
markets = paradex.api_client.fetch_markets()

# Get BTC market summary/ticker
summary = paradex.api_client.fetch_markets_summary({"market": "BTC-USD-PERP"})

# Get orderbook
orderbook = paradex.api_client.fetch_orderbook("BTC-USD-PERP")

# Get best bid/offer
bbo = paradex.api_client.fetch_bbo("BTC-USD-PERP")

# Get OHLCV candles
klines = paradex.api_client.fetch_klines(
    symbol="BTC-USD-PERP",
    resolution="1",  # 1 minute
    start_at=1704067200,  # Unix timestamp
    end_at=1704153600
)

# Get recent trades
trades = paradex.api_client.fetch_trades({"market": "BTC-USD-PERP"})
```

### Account Information (Private - Auth Required)

```python
# Account summary
account = paradex.api_client.fetch_account_summary()

# Balances
balances = paradex.api_client.fetch_balances()

# Open positions
positions = paradex.api_client.fetch_positions()

# Open orders
orders = paradex.api_client.fetch_orders()

# Order history
history = paradex.api_client.fetch_orders_history()

# Trade fills
fills = paradex.api_client.fetch_fills()

# Funding payments
funding = paradex.api_client.fetch_funding_payments()
```

### Order Placement

```python
from paradex_py.common.order import Order, OrderSide, OrderType

# Create a limit order
order = Order(
    market="BTC-USD-PERP",
    side=OrderSide.BUY,
    type=OrderType.LIMIT,
    size="0.01",
    price="95000",
    client_id="my_order_001"
)

# Submit order
result = paradex.api_client.submit_order(order)

# Submit batch orders
orders = [order1, order2, order3]
result = paradex.api_client.submit_orders_batch(orders)

# Modify order
modified = paradex.api_client.modify_order(order_id="123", order=updated_order)

# Cancel order
paradex.api_client.cancel_order(order_id="123")

# Cancel by client ID
paradex.api_client.cancel_order_by_client_id(client_id="my_order_001")

# Cancel all orders
paradex.api_client.cancel_all_orders()
```

### WebSocket Real-Time Data

```python
import asyncio
from paradex_py.message.websocket import ParadexWebsocketChannel

async def on_message(ws_channel, message):
    print(f"Channel: {ws_channel}, Message: {message}")

async def main():
    # Connect
    await paradex.ws_client.connect()

    # Subscribe to public channels
    await paradex.ws_client.subscribe(
        ParadexWebsocketChannel.BBO,
        callback=on_message,
        params={"market": "BTC-USD-PERP"}
    )

    await paradex.ws_client.subscribe(
        ParadexWebsocketChannel.TRADES,
        callback=on_message,
        params={"market": "BTC-USD-PERP"}
    )

    # Subscribe to private channels (requires auth)
    await paradex.ws_client.subscribe(
        ParadexWebsocketChannel.ORDERS,
        callback=on_message
    )

    await paradex.ws_client.subscribe(
        ParadexWebsocketChannel.POSITIONS,
        callback=on_message
    )

    # Keep running
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
```

### WebSocket Channels

| Channel | Type | Description |
|---------|------|-------------|
| `BBO` | Public | Best bid/ask updates |
| `MARKETS_SUMMARY` | Public | Market ticker updates |
| `TRADES` | Public | Trade updates |
| `ORDER_BOOK` | Public | Orderbook snapshots |
| `FUNDING_DATA` | Public | Funding rate data |
| `ACCOUNT` | Private | Account status updates |
| `ORDERS` | Private | Order updates |
| `FILLS` | Private | Fill notifications |
| `POSITIONS` | Private | Position changes |
| `BALANCE_EVENTS` | Private | PnL calculations |

### Cleanup

```python
await paradex.close()
```

---

## Lighter Integration

### Overview
- **Type**: Zero-fee perpetual DEX on zk-rollup
- **Leverage**: Up to 50x on BTC/ETH
- **Fees**: Zero for standard accounts (0.2 bps maker / 2 bps taker for premium)
- **SDK**: `lighter-v2-python`

### Installation

```bash
pip install lighter-v2-python
```

### Authentication

Lighter uses API keys with account indices:

```python
import lighter

BASE_URL = "https://api.lighter.xyz"

# Initialize SignerClient
client = lighter.SignerClient(
    url=BASE_URL,
    api_private_keys={API_KEY_INDEX: PRIVATE_KEY},
    account_index=ACCOUNT_INDEX
)
```

### Getting Your Credentials

1. **Account Index**: Query via AccountApi if unknown
2. **API Key Index**: Indices 3-254 available (0-2 reserved for apps)
3. **Private Key**: Generate and store securely

### Environment Variables

```bash
# Windows (set)
set LIGHTER_ACCOUNT_INDEX=your_account_index
set LIGHTER_API_KEY_INDEX=your_api_key_index
set API_KEY_PRIVATE_KEY=your_private_key

# Linux/Mac (export)
export LIGHTER_ACCOUNT_INDEX=your_account_index
export LIGHTER_API_KEY_INDEX=your_api_key_index
export API_KEY_PRIVATE_KEY=your_private_key
```

### API Classes

| Class | Purpose |
|-------|---------|
| `AccountApi` | Account data, API keys, subaccounts |
| `TransactionApi` | Nonce management, transaction submission |
| `OrderApi` | Orderbook data, market information |

### Market Data

```python
from lighter.api import OrderApi

order_api = OrderApi(BASE_URL)

# Get all markets/orderbooks
markets = order_api.get_orderbooks()

# Get specific orderbook
btc_book = order_api.get_orderbook("BTC-USD")
```

### Account Information

```python
from lighter.api import AccountApi

account_api = AccountApi(BASE_URL)

# Get account info
account = account_api.get_account(account_index=ACCOUNT_INDEX)

# Get API keys
api_keys = account_api.get_api_keys(account_index=ACCOUNT_INDEX)
```

### Order Placement

```python
from lighter.api import TransactionApi

tx_api = TransactionApi(BASE_URL, client)

# Get nonce for signing
nonce = tx_api.next_nonce()

# Order types supported:
# - LIMIT
# - MARKET
# - STOP_LOSS
# - STOP_LOSS_LIMIT
# - TAKE_PROFIT
# - TAKE_PROFIT_LIMIT
# - TWAP

# Time-in-force options:
# - IMMEDIATE_OR_CANCEL (IOC)
# - GOOD_TILL_TIME (GTT)
# - POST_ONLY

# Create and submit order
order = {
    "market": "BTC-USD",
    "side": "BUY",
    "type": "LIMIT",
    "base_amount": 1000000,  # Integer value
    "price": 95000000000,    # Integer value
    "time_in_force": "GOOD_TILL_TIME",
    "client_order_index": 1
}

result = tx_api.send_tx(order)

# Batch orders
orders = [order1, order2, order3]
result = tx_api.send_tx_batch(orders)
```

### WebSocket

```python
from lighter.ws import WsClient

ws_client = WsClient(BASE_URL)

# Subscribe to account updates
ws_client.subscribe_account(account_index=ACCOUNT_INDEX)

# Subscribe to orderbook
ws_client.subscribe_orderbook("BTC-USD")
```

---

## Implementation Checklist

### Phase 1: Environment Setup

- [ ] Install SDKs
  ```bash
  pip install paradex-py lighter-v2-python
  ```

- [ ] Create `.env` file (add to `.gitignore`)
  ```
  # Paradex
  PARADEX_L1_ADDRESS=0x...
  PARADEX_L1_PRIVATE_KEY=0x...
  PARADEX_L2_ADDRESS=0x...
  PARADEX_L2_PRIVATE_KEY=0x...

  # Lighter
  LIGHTER_ACCOUNT_INDEX=...
  LIGHTER_API_KEY_INDEX=...
  API_KEY_PRIVATE_KEY=...
  ```

- [ ] Test connectivity
  ```python
  # Paradex
  from paradex_py import Paradex
  paradex = Paradex(env=Environment.TESTNET, ...)
  print(paradex.api_client.fetch_markets())

  # Lighter
  import lighter
  client = lighter.SignerClient(...)
  ```

### Phase 2: Data Feed Integration

- [ ] Connect WebSocket for real-time BTC price
- [ ] Subscribe to BBO (best bid/offer) updates
- [ ] Build local orderbook from WebSocket
- [ ] Calculate mid price for strategy signals

### Phase 3: Strategy Integration

- [ ] Port `indicators.py` to use live data
- [ ] Connect signal generator to WebSocket feed
- [ ] Implement position tracking
- [ ] Add order state management

### Phase 4: Order Execution

- [ ] Implement market order function
- [ ] Implement limit order function
- [ ] Add stop-loss order logic
- [ ] Add take-profit order logic
- [ ] Test order lifecycle (place, fill, cancel)

### Phase 5: Risk Management

- [ ] Implement circuit breakers
- [ ] Add drawdown monitoring
- [ ] Position size limits
- [ ] Daily loss limits

### Phase 6: Paper Trading

- [ ] Run with paper trading flag
- [ ] Verify signal timing
- [ ] Check order execution latency
- [ ] Monitor for 24+ hours

### Phase 7: Live Deployment

- [ ] Switch to production environment
- [ ] Start with minimum position size
- [ ] Monitor real-time P&L
- [ ] Scale up gradually

---

## Code Templates

### Paradex Bot Skeleton

```python
import asyncio
import os
from paradex_py import ParadexSubkey
from paradex_py.environment import Environment
from paradex_py.message.websocket import ParadexWebsocketChannel

class ParadexBot:
    def __init__(self):
        self.paradex = ParadexSubkey(
            env=Environment.PROD,
            l2_private_key=os.environ["PARADEX_L2_PRIVATE_KEY"],
            l2_address=os.environ["PARADEX_L2_ADDRESS"]
        )
        self.current_position = 0
        self.last_price = None

    async def on_bbo(self, channel, message):
        """Handle best bid/offer updates"""
        self.last_price = (message["bid"] + message["ask"]) / 2
        await self.check_signals()

    async def on_position(self, channel, message):
        """Handle position updates"""
        self.current_position = message.get("size", 0)

    async def check_signals(self):
        """Check strategy signals and execute trades"""
        # Implement your strategy logic here
        pass

    async def place_order(self, side, size, price=None):
        """Place an order"""
        from paradex_py.common.order import Order, OrderSide, OrderType

        order = Order(
            market="BTC-USD-PERP",
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            type=OrderType.LIMIT if price else OrderType.MARKET,
            size=str(size),
            price=str(price) if price else None
        )
        return self.paradex.api_client.submit_order(order)

    async def run(self):
        """Main bot loop"""
        await self.paradex.ws_client.connect()

        # Subscribe to price feed
        await self.paradex.ws_client.subscribe(
            ParadexWebsocketChannel.BBO,
            callback=self.on_bbo,
            params={"market": "BTC-USD-PERP"}
        )

        # Subscribe to position updates
        await self.paradex.ws_client.subscribe(
            ParadexWebsocketChannel.POSITIONS,
            callback=self.on_position
        )

        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        finally:
            await self.paradex.close()

if __name__ == "__main__":
    bot = ParadexBot()
    asyncio.run(bot.run())
```

### Lighter Bot Skeleton

```python
import os
import lighter
from lighter.api import AccountApi, OrderApi, TransactionApi

class LighterBot:
    def __init__(self):
        self.base_url = "https://api.lighter.xyz"
        self.account_index = int(os.environ["LIGHTER_ACCOUNT_INDEX"])
        self.api_key_index = int(os.environ["LIGHTER_API_KEY_INDEX"])
        self.private_key = os.environ["API_KEY_PRIVATE_KEY"]

        self.client = lighter.SignerClient(
            url=self.base_url,
            api_private_keys={self.api_key_index: self.private_key},
            account_index=self.account_index
        )

        self.order_api = OrderApi(self.base_url)
        self.tx_api = TransactionApi(self.base_url, self.client)

    def get_price(self):
        """Get current BTC price"""
        book = self.order_api.get_orderbook("BTC-USD")
        bid = book["bids"][0]["price"] if book["bids"] else 0
        ask = book["asks"][0]["price"] if book["asks"] else 0
        return (bid + ask) / 2

    def place_order(self, side, amount, price, order_type="LIMIT"):
        """Place an order"""
        nonce = self.tx_api.next_nonce()

        order = {
            "market": "BTC-USD",
            "side": side.upper(),
            "type": order_type,
            "base_amount": int(amount * 1e6),  # Convert to integer
            "price": int(price * 1e6),
            "time_in_force": "GOOD_TILL_TIME",
            "nonce": nonce
        }

        return self.tx_api.send_tx(order)

    def run(self):
        """Main bot loop"""
        while True:
            price = self.get_price()
            # Implement your strategy logic here
            import time
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    bot = LighterBot()
    bot.run()
```

---

## Resources

### Paradex
- Documentation: https://tradeparadex.github.io/paradex-py/
- SDK GitHub: https://github.com/tradeparadex/paradex-py
- App: https://app.paradex.trade

### Lighter
- Documentation: https://docs.lighter.xyz
- API Docs: https://apidocs.lighter.xyz
- App: https://app.lighter.xyz

### perp-dex-toolkit
- GitHub: https://github.com/earthskyorg/perp-dex-toolkit
- Supports: Paradex, Lighter, and 5+ other exchanges
