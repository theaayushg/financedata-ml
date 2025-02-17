import duckdb
conn = duckdb.connect("market_data.duckdb")

# ✅ Retrieve the last 10 stored prices
results = conn.execute("SELECT * FROM btc_prices ORDER BY timestamp DESC LIMIT 10").fetchall()

# ✅ Print results
for row in results:
    print(row)
