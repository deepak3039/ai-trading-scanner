import os
import pandas as pd
import requests

LOG_FILE = "trade_log.csv"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_alert(message):
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
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
    print(f"❌ Error sending Telegram performance summary: {e}")


def analyze_performance():
  if not os.path.exists(LOG_FILE):
    print("📊 No trade log file found yet.")
    return

  df = pd.read_csv(LOG_FILE)
  if df.empty:
    print("📊 Trade log is empty.")
    return

  total_trades = len(df)
  unique_assets = df["Asset"].nunique()
  bullish_count = len(df[df["Direction"] == "BULLISH"])
  bearish_count = len(df[df["Direction"] == "BEARISH"])

  # Format summary report text
  report = (
      f"📊 *AI TRADING PERFORMANCE REPORT* 📊\n\n"
      f"• Total Signals Logged: `{total_trades}`\n"
      f"• Unique Assets Traded: `{unique_assets}`\n"
      f"• Bullish / Bearish: `{bullish_count} / {bearish_count}`\n\n"
      f"*Recent Signals:*\n"
  )

  for _, row in df.tail(3).iterrows():
    report += (
        f"• `{row['Asset']}` | *{row['Direction']}* @ `{row['Entry_Price']}`"
        f" (Conf: `{row['Confidence_%']}%`)\n"
    )

  print(report)
  send_telegram_alert(report)


if __name__ == "__main__":
  analyze_performance()
