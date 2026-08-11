import os
import pandas as pd

LOG_FILE = "trade_log.csv"


def analyze_performance():
  if not os.path.exists(LOG_FILE):
    print(
        "📊 No trade log file found yet. Let the scanner run to generate trade"
        " data."
    )
    return

  df = pd.read_csv(LOG_FILE)

  if df.empty:
    print("📊 Trade log is empty.")
    return

  total_trades = len(df)
  unique_assets = df["Asset"].nunique()
  bullish_count = len(df[df["Direction"] == "BULLISH"])
  bearish_count = len(df[df["Direction"] == "BEARISH"])

  print("=" * 40)
  print("📈 AI TRADING BOT PERFORMANCE ANALYTICS")
  print("=" * 40)
  print(f"Total Signals Logged: {total_trades}")
  print(f"Unique Assets Traded: {unique_assets}")
  print(
      f"Direction Breakdown: {bullish_count} Bullish | {bearish_count} Bearish"
  )
  print("\nBreakdown by Asset:")
  print(df["Asset"].value_counts())
  print("-" * 40)
  print("Recent Activity:")
  print(df.tail(5))
  print("=" * 40)


if __name__ == "__main__":
  analyze_performance()
