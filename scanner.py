import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

# --- WATCHLIST ---
ALL_ASSETS = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
    "XRP/USDT": "XRP-USD",
    "ADA/USDT": "ADA-USD",
    "AVAX/USDT": "AVAX-USD",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
}


def send_telegram_alert(message):
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print(f"[Telegram Mock Alert]: {message}")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")


def compute_indicators(df):
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["trend_filter"] = (df["close"] > df["ema_200"]).astype(int)

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_middle"] = df["close"].rolling(window=20).mean()
    std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (2 * std)
    df["bb_lower"] = df["bb_middle"] - (2 * std)

    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    df["target"] = (df["close"].shift(-3) > df["close"]).astype(int)
    return df.dropna()


def run_live_scanner():
    feature_columns = [
        "rsi",
        "ema_20",
        "ema_50",
        "macd",
        "macd_hist",
        "bb_lower",
        "bb_upper",
        "atr",
        "trend_filter",
        "volume",
    ]

    print("🚀 Running Live Optimized Market Scanner...")

    for label, ticker in ALL_ASSETS.items():
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df = (
                df[["open", "high", "low", "close", "volume"]]
                .astype(float)
                .dropna()
            )
            if len(df) < 200:
                continue

            df = compute_indicators(df)

            X = df[feature_columns]
            y = df["target"]

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_split=30,
                random_state=42,
            )
            model.fit(X, y)

            latest_row = df.iloc[[-1]]
            X_latest = latest_row[feature_columns]
            pred = model.predict(X_latest)[0]
            proba = model.predict_proba(X_latest)[0]
            conf = max(proba) * 100

            if conf >= 60.0:
                current_price = float(latest_row["close"].values[0])
                trend = int(latest_row["trend_filter"].values[0])
                atr = float(latest_row["atr"].values[0])
                entry_time = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )

                if pred == 1 and trend == 1:
                    sl = current_price - (1.0 * atr)
                    tp = current_price + (3.0 * atr)
                    msg = (
                        f"🚨 **OPTIMIZED BUY SIGNAL** 🚨\n\n"
                        f"**Asset:** `{label}`\n"
                        f"**Time:** `{entry_time}`\n"
                        f"**Entry Price:** `{current_price:.5g}`\n"
                        f"**Take Profit:** `{tp:.5g}`\n"
                        f"**Stop Loss:** `{sl:.5g}`\n"
                        f"**Confidence:** `{conf:.1f}%`\n"
                        f"**Strategy:** Trend-Aligned Long"
                    )
                    send_telegram_alert(msg)
                    print(f"Signal sent for {label} (BUY)")

                elif pred == 0 and trend == 0:
                    sl = current_price + (1.0 * atr)
                    tp = current_price - (3.0 * atr)
                    msg = (
                        f"🚨 **OPTIMIZED SELL SIGNAL** 🚨\n\n"
                        f"**Asset:** `{label}`\n"
                        f"**Time:** `{entry_time}`\n"
                        f"**Entry Price:** `{current_price:.5g}`\n"
                        f"**Take Profit:** `{tp:.5g}`\n"
                        f"**Stop Loss:** `{sl:.5g}`\n"
                        f"**Confidence:** `{conf:.1f}%`\n"
                        f"**Strategy:** Trend-Aligned Short"
                    )
                    send_telegram_alert(msg)
                    print(f"Signal sent for {label} (SELL)")

        except Exception as e:
            print(f"Error scanning {label}: {e}")


if __name__ == "__main__":
    run_live_scanner()
