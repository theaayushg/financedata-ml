import asyncio
import websockets
import json
import duckdb
import numpy as np
from sklearn.ensemble import IsolationForest

# ✅ Connect to DuckDB
conn = duckdb.connect("market_data.duckdb")

# ✅ Create a table if it doesn't exist
conn.execute("""
CREATE TABLE IF NOT EXISTS btc_prices (
    price DOUBLE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# ✅ Train Isolation Forest Model from Real Data (Rolling Window)
def train_initial_model():
    """Trains the ML model with real BTC price changes instead of raw prices."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 750").fetchall()
    
    if len(prices) < 50:  # If not enough data, use synthetic BTC range
        print("⚠️ Not enough real data. Using estimated price range for training.")
        price_data = np.random.normal(95500, 500, 500).reshape(-1, 1)
    else:
        print(f"✅ Training model on {len(prices)} real BTC prices.")
        price_data = np.array(prices).reshape(-1, 1)
        price_changes = np.diff(price_data, axis=0)  # Track price deltas

        # If changes are empty, fall back to price values
        if len(price_changes) < 10:
            price_changes = price_data

    model = IsolationForest(n_estimators=100, contamination=0.02)  # Lower contamination for fewer false positives
    model.fit(price_changes)
    return model

# ✅ Initialize Model
model = train_initial_model()

def detect_anomaly(price):
    """Detects anomalies based on price changes instead of raw prices."""
    latest_prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 2").fetchall()
    
    if len(latest_prices) < 2:
        return False  # Not enough data for comparison

    prev_price = latest_prices[1][0]
    price_change = price - prev_price  # Calculate difference

    prediction = model.predict([[price_change]])  # Predict based on price movement
    return prediction[0] == -1  # True if anomaly detected

# ✅ Function to store BTC price in DuckDB
def store_price(price):
    conn.execute("INSERT INTO btc_prices (price) VALUES (?)", (price,))
    print(f"Stored BTC Price: {price}")

# ✅ Function to analyze BTC price trend
def get_price_trend():
    """Calculate BTC price change over the last 10 prices."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 10").fetchall()
    
    if len(prices) < 2:
        return "Not enough data for trend analysis."

    latest_price = prices[0][0]
    previous_price = prices[-1][0]

    percent_change = ((latest_price - previous_price) / previous_price) * 100
    return f"BTC price changed by {percent_change:.2f}% in last 10 updates."

# ✅ Function to Retrain ML Model Every 100 Price Updates
def retrain_model():
    """Retrains the Isolation Forest model with real BTC price changes."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 500").fetchall()
    
    if len(prices) < 50:  # Needs at least 50 prices to train
        print("⚠️ Not enough real data for retraining.")
        return

    new_data = np.array(prices).reshape(-1, 1)
    price_changes = np.diff(new_data, axis=0)  # Compute changes in price
    
    if len(price_changes) < 10:
        price_changes = new_data  # Use price values if changes aren't enough

    global model
    model = IsolationForest(n_estimators=100, contamination=0.02)
    model.fit(price_changes)
    print(f"🔄 Model retrained on {len(prices)} BTC price changes.")

# ✅ Coinbase WebSocket API to Fetch Live BTC Prices
price_counter = 0  # Track number of updates for retraining

async def fetch_coinbase_data():
    global price_counter
    url = "wss://ws-feed.exchange.coinbase.com"
    async with websockets.connect(url) as ws:
        subscribe_message = {
            "type": "subscribe",
            "channels": [{"name": "ticker", "product_ids": ["BTC-USD"]}]
        }
        await ws.send(json.dumps(subscribe_message))

        while True:
            data = await ws.recv()
            trade = json.loads(data)

            if "price" in trade:
                price = float(trade["price"])

                # ✅ Store in DuckDB
                store_price(price)
                price_counter += 1

                # ✅ Retrain ML model every 100 price updates
                if price_counter % 100 == 0:
                    retrain_model()

                # ✅ Detect anomalies
                is_anomaly = detect_anomaly(price)
                if is_anomaly:
                    print(f"🚨 Anomaly Detected: BTC Price {price}")
                else:
                    print(f"✅ Normal Price: {price}")

asyncio.run(fetch_coinbase_data())