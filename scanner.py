import os
from datetime import datetime
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- LOAD SECRETS FROM ENVIRONMENT ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

# Optional Paper Trading Credentials (Binance Testnet)
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY")

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
LOG_FILE = "trade_log.csv"


def send_telegram_alert(message):
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
  try:
    requests.post(url, json=payload, timeout=10)
  except Exception as e:
    print(f"❌ Error sending Telegram alert: {e}")


def execute_paper_order(symbol, direction, price):
  """Executes a simulated paper order on Binance Testnet via CCXT."""
  if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    print("⚠️ Binance API keys missing. Skipping live paper execution.")
    return "SKIPPED"

  try:
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET_KEY,
        "enableRateLimit": True,
    })
    exchange.set_sandbox_mode(True)  # Enable testnet sandbox

    side = "buy" if direction == "BULLISH" else "sell"
    amount = round(100 / price, 5)  # $100 virtual position size

    order = exchange.create_order(
        symbol=symbol, type="market", side=side, amount=amount
    )
    print(f"✅ Paper order executed successfully: {order.get('id')}")
    return "EXECUTED"
  except Exception as e:
    print(f"❌ Paper execution error: {e}")
    return "FAILED"


def log_trade_to_csv(
    timestamp,
    asset,
    direction,
    price,
    confidence,
    stop_loss,
    take_profit,
    status,
):
  file_exists = os.path.exists(LOG_FILE)
  log_data = {
      "Timestamp": [timestamp],
      "Asset": [asset],
      "Direction": [direction],
      "Entry_Price": [price],
      "Confidence_%": [confidence],
      "Stop_Loss": [stop_loss],
      "Take_Profit": [take_profit],
      "Status": [status],
  }
  df_log = pd.DataFrame(log_data)
  df_log.to_csv(LOG_FILE, mode="a", index=False, header=not file_exists)
  print(f"📝 Logged trade for {asset} to {LOG_FILE}")


def run_scan_cycle():
  print("🤖 Executing AI Scan Cycle with Paper Trading Execution...")
  current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

  exchange = ccxt.binance({"enableRateLimit": True})
  for symbol in CRYPTO_SYMBOLS:
    try:
      ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=500)
      df = pd.DataFrame(
          ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
      )

      df["rsi"] = ta.rsi(df["close"], length=14)
      df["sma_50"] = ta.sma(df["close"], length=50)
      df["ma_dist"] = (df["close"] - df["sma_50"]) / df["sma_50"]
      df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
      df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
      df = df.dropna()

      X = df[["rsi", "ma_dist", "atr", "volume"]]
      y = df["target"]

      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, shuffle=False
      )
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X_train, y_train)

      latest_features = X.iloc[[-1]]
      prediction = model.predict(latest_features)[0]
      prediction_proba = model.predict_proba(latest_features)[0]
      current_price = df["close"].iloc[-1]
      current_atr = df["atr"].iloc[-1]

      confidence = (
          prediction_proba[1] * 100
          if prediction == 1
          else prediction_proba[0] * 100
      )

      if confidence >= 60.0:
        direction = "BULLISH" if prediction == 1 else "BEARISH"
        if prediction == 1:
          stop_loss = current_price - (2 * current_atr)
          take_profit = current_price + (3 * current_atr)
        else:
          stop_loss = current_price + (2 * current_atr)
          take_profit = current_price - (3 * current_atr)

        order_status = execute_paper_order(symbol, direction, current_price)

        msg = (
            f"🚨 *AI TRADE SETUP & EXECUTION* 🚨\n\nAsset: `{symbol}`\nDirection:"
            f" *{direction}*\nEntry Price: `{current_price:.2f}`\nAI Confidence:"
            f" *{confidence:.1f}%*\nOrder Status: *{order_status}*\n\n🛡️ *Risk"
            f" Management:*\nStop-Loss: `{stop_loss:.2f}`\nTake-Profit:"
            f" `{take_profit:.2f}`"
        )
        send_telegram_alert(msg)

        log_trade_to_csv(
            current_time,
            symbol,
            direction,
            current_price,
            round(confidence, 1),
            round(stop_loss, 2),
            round(take_profit, 2),
            order_status,
        )
    except Exception as e:
      print(f"Error processing {symbol}: {e}")

  # Commodity Scan (Gold via Alpha Vantage)
  try:
    url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=XAU&to_symbol=USD&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    time_series = data.get("Time Series FX (Daily)", {})

    if time_series:
      df = pd.DataFrame.from_dict(time_series, orient="index")
      df = df.rename(
          columns={
              "1. open": "open",
              "2. high": "high",
              "3. low": "low",
              "4. close": "close",
          }
      )
      df = df.astype(float).sort_index()
      df["rsi"] = ta.rsi(df["close"], length=14)
      df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

      current_rsi = df["rsi"].iloc[-1]
      current_price = df["close"].iloc[-1]
      current_atr = df["atr"].iloc[-1]

      if current_rsi < 30 or current_rsi > 70:
        cond = "OVERSOLD" if current_rsi < 30 else "OVERBOUGHT"
        direction = "BULLISH" if current_rsi < 30 else "BEARISH"

        if current_rsi < 30:
          stop_loss = current_price - (2 * current_atr)
          take_profit = current_price + (3 * current_atr)
        else:
          stop_loss = current_price + (2 * current_atr)
          take_profit = current_price - (3 * current_atr)

        msg = (
            f"🚨 *COMMODITY SETUP* 🚨\n\nAsset: `XAU/USD (Gold)`\nCondition:"
            f" *{cond} (RSI: {current_rsi:.2f})*\nPrice:"
            f" `{current_price:.2f}`\n\n🛡️ *Risk"
            f" Management:*\nStop-Loss: `{stop_loss:.2f}`\nTake-Profit:"
            f" `{take_profit:.2f}`"
        )
        send_telegram_alert(msg)

        log_trade_to_csv(
            current_time,
            "XAU/USD",
            direction,
            current_price,
            99.9,
            round(stop_loss, 2),
            round(take_profit, 2),
            "LOG_ONLY",
        )
  except Exception as e:
    print(f"Alpha Vantage Error: {e}")


if __name__ == "__main__":
  run_scan_cycle()
