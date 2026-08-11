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
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.environ.get("BINANCE_SECRET_KEY")

# --- EXPANDED WATCHLISTS ---
CRYPTO_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
]

# Forex & Commodities mapped to Alpha Vantage currency parameters
FOREX_COMMODITY_SYMBOLS = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "EUR/JPY": ("EUR", "JPY"),
    "GBP/JPY": ("GBP", "JPY"),
    "USD/JPY": ("USD", "JPY"),
    "AUD/USD": ("AUD", "USD"),
    "XAU/USD (Gold)": ("XAU", "USD"),
    "XAG/USD (Silver)": ("XAG", "USD"),
}

LOG_FILE = "trade_log.csv"


def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram credentials missing. Skipping alert.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"❌ Error sending Telegram alert: {e}")


def execute_paper_order(symbol, direction, price):
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        print("⚠️ Binance API keys missing. Skipping live paper execution.")
        return "SKIPPED"
    if "/" not in symbol or "USDT" not in symbol:
        return "LOG_ONLY"

    try:
        exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_SECRET_KEY,
            "enableRateLimit": True,
        })
        exchange.set_sandbox_mode(True)

        side = "buy" if direction == "BULLISH" else "sell"
        amount = round(100 / price, 5)

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


def add_advanced_features(df):
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df["macd"] = macd_df.iloc[:, 0]
        df["macd_hist"] = macd_df.iloc[:, 1]
        df["macd_signal"] = macd_df.iloc[:, 2]
    else:
        df["macd"] = 0
        df["macd_hist"] = 0
        df["macd_signal"] = 0

    bb_df = ta.bbands(df["close"], length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        df["bb_lower"] = bb_df.iloc[:, 0]
        df["bb_middle"] = bb_df.iloc[:, 1]
        df["bb_upper"] = bb_df.iloc[:, 2]
    else:
        df["bb_lower"] = df["close"]
        df["bb_middle"] = df["close"]
        df["bb_upper"] = df["close"]

    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    df["swing_high"] = df["high"][
        (df["high"] == df["high"].rolling(5, center=True).max())
    ].fillna(0)
    df["swing_low"] = df["low"][
        (df["low"] == df["low"].rolling(5, center=True).min())
    ].fillna(0)

    df["support"] = df["low"].rolling(window=50).min()
    df["resistance"] = df["high"].rolling(window=50).max()

    df["bullish_fvg"] = (df["low"] > df["high"].shift(2)).astype(int)
    df["bearish_fvg"] = (df["high"] < df["low"].shift(2)).astype(int)

    df["prev_high"] = df["high"].shift(1).rolling(10).max()
    df["prev_low"] = df["low"].shift(1).rolling(10).min()
    df["liquidity_sweep_bullish"] = (
        (df["low"] < df["prev_low"]) & (df["close"] > df["open"])
    ).astype(int)
    df["liquidity_sweep_bearish"] = (
        (df["high"] > df["prev_high"]) & (df["close"] < df["open"])
    ).astype(int)

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()


def evaluate_and_trade(df, symbol, current_time):
    feature_columns = [
        "rsi",
        "ema_20",
        "ema_50",
        "macd",
        "macd_hist",
        "bb_lower",
        "bb_upper",
        "atr",
        "bullish_fvg",
        "bearish_fvg",
        "liquidity_sweep_bullish",
        "liquidity_sweep_bearish",
        "volume",
    ]

    X = df[feature_columns]
    y = df["target"]

    if len(X) < 50:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
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
            f"🧠 *AI MULTI-ASSET SETUP* 🧠\n\nAsset: `{symbol}`\nDirection:"
            f" *{direction}*\nPrice: `{current_price:.5f}`\nAI Confidence:"
            f" *{confidence:.1f}%*\nStatus: *{order_status}*\n\n🛡️ *Risk"
            f" Management:*\nStop-Loss: `{stop_loss:.5f}`\nTake-Profit:"
            f" `{take_profit:.5f}`"
        )
        send_telegram_alert(msg)

        log_trade_to_csv(
            current_time,
            symbol,
            direction,
            current_price,
            round(confidence, 1),
            round(stop_loss, 5),
            round(take_profit, 5),
            order_status,
        )


def run_scan_cycle():
    print(
        "🤖 Executing Full Multi-Asset Scan Cycle (Crypto via CCXT +"
        " Forex/Commodities via Alpha Vantage)..."
    )
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Scan Crypto via CCXT
    exchange = ccxt.binance({"enableRateLimit": True})
    for symbol in CRYPTO_SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=500)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df = add_advanced_features(df)
            evaluate_and_trade(df, symbol, current_time)
        except Exception as e:
            print(f"Error processing crypto {symbol}: {e}")

    # 2. Scan Forex & Commodities via Alpha Vantage FX_DAILY
    for label, (from_sym, to_sym) in FOREX_COMMODITY_SYMBOLS.items():
        try:
            url = f"https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_sym}&to_symbol={to_sym}&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url)
            data = response.json()
            time_series = data.get("Time Series FX (Daily)", {})

            if not time_series:
                print(f"No data returned for {label} from Alpha Vantage.")
                continue

            df = pd.DataFrame.from_dict(time_series, orient="index")
            df = df.rename(
                columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                }
            )
            df["volume"] = 0.0  # FX endpoints default volume to zero
            df = (
                df[["open", "high", "low", "close", "volume"]]
                .astype(float)
                .sort_index()
            )
            df = add_advanced_features(df)
            evaluate_and_trade(df, label, current_time)
        except Exception as e:
            print(f"Error processing forex/commodity {label}: {e}")


if __name__ == "__main__":
    run_scan_cycle()
