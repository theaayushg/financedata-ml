import asyncio
import websockets
import json
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Connect to DuckDB
conn = duckdb.connect("market_data.duckdb")

# Keep only the latest 1000 entries (avoid DB bloat)
conn.execute("""
DELETE FROM btc_prices WHERE timestamp NOT IN (
    SELECT timestamp FROM btc_prices ORDER BY timestamp DESC LIMIT 1000
)
""")

# Create table if it doesn't exist
conn.execute("""
CREATE TABLE IF NOT EXISTS btc_prices (
    price DOUBLE,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")


# Train Isolation Forest Model (Rolling Window)
def train_initial_model():
    """Trains ML model on BTC price changes."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 750").fetchall()
    
    if len(prices) < 50:
        print("⚠️ Not enough real data. Using synthetic training data.")
        price_data = np.random.normal(95500, 500, 500).reshape(-1, 1)
    else:
        print(f"✅ Training model on {len(prices)} BTC prices.")
        price_data = np.array(prices).reshape(-1, 1)
        price_changes = np.diff(price_data, axis=0)
        if len(price_changes) < 10:
            price_changes = price_data  # Default to price values if deltas are insufficient

    model = IsolationForest(n_estimators=100, contamination=0.02)
    model.fit(price_changes)
    return model


# Initialize Model
model = train_initial_model()


# Detect Anomalies
def detect_anomaly(price):
    """Detects anomalies based on price changes."""
    latest_prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 2").fetchall()
    if len(latest_prices) < 2:
        return False

    prev_price = latest_prices[1][0]
    price_change = price - prev_price
    prediction = model.predict([[price_change]])
    return prediction[0] == -1  # True if anomaly detected

# Store BTC Price
def store_price(price):
    conn.execute("INSERT INTO btc_prices (price) VALUES (?)", (price,))
    print(f"Stored BTC Price: {price}")


# Compute Moving Averages
def compute_moving_averages():
    """Calculates SMA (10, 50) and EMA (10, 50)."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 200").fetchall()
    if len(prices) < 50:
        return "⚠️ Not enough data for moving averages."

    df = pd.DataFrame(prices, columns=["price"])
    df["SMA_10"] = df["price"].rolling(window=10).mean()
    df["SMA_50"] = df["price"].rolling(window=50).mean()
    df["EMA_10"] = df["price"].ewm(span=10, adjust=False).mean()
    df["EMA_50"] = df["price"].ewm(span=50, adjust=False).mean()

    return df


# Compute Volatility
def compute_volatility():
    """Computes rolling standard deviation over last 50 prices."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 50").fetchall()
    if len(prices) < 10:
        return "⚠️ Not enough data for volatility calculation."

    df = pd.DataFrame(prices, columns=["price"])
    volatility = df["price"].rolling(window=10).std().iloc[-1]
    return f"📉 BTC Volatility (10-period rolling): {volatility:.2f}"


# Compute Relative Strength Index (RSI)
def compute_rsi():
    """Calculates 14-period RSI."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 200").fetchall()
    if len(prices) < 14:
        return "⚠️ Not enough data for RSI calculation."

    df = pd.DataFrame(prices, columns=["price"])
    df["change"] = df["price"].diff()
    df["gain"] = np.where(df["change"] > 0, df["change"], 0)
    df["loss"] = np.where(df["change"] < 0, abs(df["change"]), 0)

    avg_gain = df["gain"].rolling(window=14).mean()
    avg_loss = df["loss"].rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return f"📈 BTC RSI (14-period): {df['RSI'].iloc[-1]:.2f}"


# Compute Bollinger Bands
def compute_bollinger_bands():
    """Calculates Bollinger Bands (20-period SMA, ±2 std deviations)."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 200").fetchall()
    if len(prices) < 20:
        return "⚠️ Not enough data for Bollinger Bands."

    df = pd.DataFrame(prices, columns=["price"])
    df["SMA_20"] = df["price"].rolling(window=20).mean()
    df["Upper_Band"] = df["SMA_20"] + (df["price"].rolling(window=20).std() * 2)
    df["Lower_Band"] = df["SMA_20"] - (df["price"].rolling(window=20).std() * 2)

    return df


# Compute Sharpe Ratio
def compute_sharpe_ratio():
    """Calculates Sharpe Ratio for BTC price changes."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 100").fetchall()
    if len(prices) < 20:
        return "⚠️ Not enough data for Sharpe Ratio."

    df = pd.DataFrame(prices, columns=["price"])
    df["returns"] = df["price"].pct_change()
    
    mean_return = df["returns"].mean()
    std_dev = df["returns"].std()
    
    sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
    return f"📊 BTC Sharpe Ratio (100-period): {sharpe_ratio:.2f}"


# Function to Retrain ML Model
def retrain_model():
    """Retrains ML model on new BTC price data."""
    prices = conn.execute("SELECT price FROM btc_prices ORDER BY timestamp DESC LIMIT 500").fetchall()
    if len(prices) < 50:
        print("⚠️ Not enough real data for retraining.")
        return

    new_data = np.array(prices).reshape(-1, 1)
    price_changes = np.diff(new_data, axis=0)
    if len(price_changes) < 10:
        price_changes = new_data

    global model
    model = IsolationForest(n_estimators=100, contamination=0.02)
    model.fit(price_changes)
    print(f"🔄 Model retrained on {len(prices)} BTC price changes.")


# Coinbase WebSocket API to Fetch Live BTC Prices
price_counter = 0

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
                store_price(price)
                price_counter += 1

                # Retrain every 100 price updates
                if price_counter % 100 == 0:
                    retrain_model()

                # Display Quant Indicators
                print(detect_anomaly(price))
                print(compute_volatility())
                print(compute_rsi())
                print(compute_sharpe_ratio())

asyncio.run(fetch_coinbase_data())
