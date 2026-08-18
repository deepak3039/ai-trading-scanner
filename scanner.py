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
    "XAU/USD": "GC=F",
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

LOG_FILE = "trade_log.csv"


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


def check_active_trades():
    if not os.path.exists(LOG_FILE):
        return

    try:
        df_log = pd.read_csv(LOG_FILE)
        if df_log.empty or "asset" not in df_log.columns or "status" not in df_log.columns:
            return

        updated_rows = []
        for idx, row in df_log.iterrows():
            if row["status"] != "OPEN":
                updated_rows.append(row)
                continue

            label = row["asset"]
            ticker = ALL_ASSETS.get(label)
            if not ticker:
                updated_rows.append(row)
                continue

            df_recent = yf.download(
                ticker, period="5d", interval="1d", progress=False
            )
            if df_recent.empty:
                updated_rows.append(row)
                continue
            if isinstance(df_recent.columns, pd.MultiIndex):
                df_recent.columns = df_recent.columns.get_level_values(0)

            latest_high = float(df_recent["High"].iloc[-1])
            latest_low = float(df_recent["Low"].iloc[-1])
            current_price = float(df_recent["Close"].iloc[-1])

            direction = row["direction"]
            tp = float(row["tp"])
            sl = float(row["sl"])

            hit_status = "OPEN"
            if direction == "LONG":
                if latest_high >= tp:
                    hit_status = "TP_HIT"
                    msg = (
                        f"🎯 **TAKE PROFIT HIT** 🎯\n\n"
                        f"**Asset:** `{label}`\n"
                        f"**Direction:** `LONG`\n"
                        f"**Entry Price:** `{row['entry_price']}`\n"
                        f"**Target Hit (TP):** `{tp}`\n"
                        f"**Current Price:** `{current_price:.5g}`\n"
                        f"Status: Target Achieved Successfully! 🚀"
                    )
                    send_telegram_alert(msg)
                elif latest_low <= sl:
                    hit_status = "SL_HIT"
                    msg = (
                        f"🛑 **STOP LOSS HIT** 🛑\n\n"
                        f"**Asset:** `{label}`\n"
                        f"**Direction:** `LONG`\n"
                        f"**Entry Price:** `{row['entry_price']}`\n"
                        f"**Stop Loss Hit (SL):** `{sl}`\n"
                        f"**Current Price:** `{current_price:.5g}`\n"
                        f"Status: Risk management stop triggered."
                    )
                    send_telegram_alert(msg)
            elif direction == "SHORT":
                if latest_low <= tp:
                    hit_status = "TP_HIT"
                    msg = (
                        f"🎯 **TAKE PROFIT HIT** 🎯\n\n"
                        f"**Asset:** `{label}`\n"
                        f"**Direction:** `SHORT`\n"
                        f"**Entry Price:** `{row['entry_price']}`\n"
                        f"**Target Hit (TP):** `{tp}`\n"
                        f"**Current Price:** `{current_price:.5g}`\n"
                        f"Status: Target Achieved Successfully! 🚀"
                    )
                    send_telegram_alert(msg)
                elif latest_high >= sl:
                    hit_status = "SL_HIT"
                    msg = (
                        f"🛑 **STOP LOSS HIT** 🛑\n\n"
                        f"**Asset:** `{label}`\n"
                        f"**Direction:** `SHORT`\n"
                        f"**Entry Price:** `{row['entry_price']}`\n"
                        f"**Stop Loss Hit (SL):** `{sl}`\n"
                        f"**Current Price:** `{current_price:.5g}`\n"
                        f"Status: Risk management stop triggered."
                    )
                    send_telegram_alert(msg)

            row["status"] = hit_status
            updated_rows.append(row)

        pd.DataFrame(updated_rows).to_csv(LOG_FILE, index=False)
    except Exception as e:
        print(f"Error checking active trades: {e}")


def log_new_trade(label, direction, entry, tp, sl, timestamp):
    new_row = {
        "asset": label,
        "direction": direction,
        "entry_price": entry,
        "tp": tp,
        "sl": sl,
        "timestamp": timestamp,
        "status": "OPEN",
    }
    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE)
            if df_log.empty or "asset" not in df_log.columns:
                df_log = pd.DataFrame([new_row])
            else:
                if not df_log[
                    (df_log["asset"] == label) & (df_log["status"] == "OPEN")
                ].empty:
                    return
                df_log = pd.concat([df_log, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            df_log = pd.DataFrame([new_row])
    else:
        df_log = pd.DataFrame([new_row])
    df_log.to_csv(LOG_FILE, index=False)


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
    print("🚀 Running Optimized Market Scanner (55% Threshold & 2:1 R:R)...")

    check_active_trades()

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

            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)
            else:
                df["volume"] = 0

            df = (
                df[["open", "high", "low", "close", "volume"]]
                .astype(float)
                .dropna()
            )
            if len(df) < 200:
                print(f"⚠️ Skipped {label}: Insufficient data (<200 rows)")
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

            current_price = float(latest_row["close"].values[0])
            trend = int(latest_row["trend_filter"].values[0])
            atr = float(latest_row["atr"].values[0])
            entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            print(f"🔍 [Scan] {label} | Price: {current_price:.5g} | Conf: {conf:.1f}% | Trend: {trend} | Pred: {pred}")

            # --- LOWERED THRESHOLD TO 55.0% ---
            if conf >= 55.0:
                if pred == 1 and trend == 1:
                    sl = current_price - (1.0 * atr)
                    tp = current_price + (2.0 * atr)
                    log_new_trade(label, "LONG", current_price, tp, sl, entry_time)
                    msg = (
                        f"🚨 **OPTIMIZED BUY SIGNAL (55%+ Conf | 2:1 R:R)** 🚨\n\n"
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
                    tp = current_price - (2.0 * atr)
                    log_new_trade(label, "SHORT", current_price, tp, sl, entry_time)
                    msg = (
                        f"🚨 **OPTIMIZED SELL SIGNAL (55%+ Conf | 2:1 R:R)** 🚨\n\n"
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
