import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# --- OPTIMIZED WATCHLIST ---
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


def add_optimized_features(df):
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


def run_backtest():
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

    total_trades = 0
    wins = 0
    pnl_records = []

    print(
        "🚀 Starting Historical Backtest & Feature Optimization Evaluation..."
    )

    for label, ticker in ALL_ASSETS.items():
        try:
            df = yf.download(ticker, period="1y", interval="1h", progress=False)
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
            df = df[["open", "high", "low", "close", "volume"]].astype(float).dropna()
            if len(df) < 200:
                continue

            df = add_optimized_features(df)

            # Walk-forward split: Train on 70%, Test on 30%
            train_size = int(len(df) * 0.7)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]

            X_train = train_df[feature_columns]
            y_train = train_df["target"]

            # Optimized Model Parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
            )
            model.fit(X_train, y_train)

            for i in range(len(test_df) - 5):
                row = test_df.iloc[[i]]
                X_test_row = row[feature_columns]
                pred = model.predict(X_test_row)[0]
                proba = model.predict_proba(X_test_row)[0]
                conf = max(proba) * 100

                # Stricter confidence filter for backtesting optimization
                if conf >= 65.0:
                    total_trades += 1
                    entry_price = float(row["close"].values[0])
                    atr = float(row["atr"].values[0])

                    future_window = test_df.iloc[i + 1 : i + 6]
                    if future_window.empty:
                        continue

                    if pred == 1:
                        sl = entry_price - (2 * atr)
                        tp = entry_price + (3 * atr)
                        hit_tp = (future_window["high"] >= tp).any()
                        hit_sl = (future_window["low"] <= sl).any()

                        if hit_tp and not hit_sl:
                            wins += 1
                            pnl_records.append(3.0)
                        elif hit_sl:
                            pnl_records.append(-2.0)
                    else:
                        sl = entry_price + (2 * atr)
                        tp = entry_price - (3 * atr)
                        hit_tp = (future_window["low"] <= tp).any()
                        hit_sl = (future_window["high"] >= sl).any()

                        if hit_tp and not hit_sl:
                            wins += 1
                            pnl_records.append(3.0)
                        elif hit_sl:
                            pnl_records.append(-2.0)

        except Exception as e:
            print(f"Error backtesting {label}: {e}")

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    print("\n==============================")
    print("📊 OPTIMIZED BACKTEST RESULTS")
    print("==============================")
    print(f"Total Trades Simulated : {total_trades}")
    print(f"Winning Trades         : {wins}")
    print(f"Win Rate               : {win_rate:.2f}%")
    print(f"Cumulative Return (R)  : {sum(pnl_records):.2f}R")
    print("==============================\n")


if __name__ == "__main__":
    run_backtest()
