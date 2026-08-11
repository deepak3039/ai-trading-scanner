from datetime import datetime
import os
import pandas as pd
import yfinance as yf
from smc_engine import SMCTradingEngine


def run_smc_scanner():
  # Define your assets to scan (Yahoo Finance symbols)
  assets = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD"]
  all_signals = []

  print("Starting Multi-Timeframe SMC Scan...")

  for asset in assets:
    try:
      # Fetch Higher Timeframe (1H) for structure/OB and Lower Timeframe (5M) for confirmation
      htf_df = yf.download(asset, interval="1h", period="5d", progress=False)
      ltf_df = yf.download(asset, interval="5m", period="1d", progress=False)

      if htf_df.empty or ltf_df.empty:
        continue

      # Clean multi-index columns if returned by yfinance
      if isinstance(htf_df.columns, pd.MultiIndex):
        htf_df.columns = htf_df.columns.get_level_values(0)
      if isinstance(ltf_df.columns, pd.MultiIndex):
        ltf_df.columns = ltf_df.columns.get_level_values(0)

      # Ensure standard lowercase/capitalized column names
      htf_df = htf_df.rename(
          columns=lambda x: x.capitalize() if x.lower() in ['open', 'high', 'low', 'close', 'volume'] else x
      )
      ltf_df = ltf_df.rename(
          columns=lambda x: x.capitalize() if x.lower() in ['open', 'high', 'low', 'close', 'volume'] else x
      )

      # Initialize and run SMC Engine
      engine = SMCTradingEngine(htf_df, ltf_df)
      signals = engine.scan_setup()

      for sig in signals:
        sig["Asset"] = asset.replace("-", "/")
        sig["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sig["status"] = "OPEN"
        all_signals.append(sig)

    except Exception as e:
      print(f"Error processing {asset}: {e}")

  # Save results to trade_log.csv
  if all_signals:
    new_df = pd.DataFrame(all_signals)
    if os.path.exists("trade_log.csv"):
      existing_df = pd.read_csv("trade_log.csv")
      combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
      combined_df = new_df
    
    combined_df.to_csv("trade_log.csv", index=False)
    print(f"Successfully logged {len(all_signals)} new SMC signals to trade_log.csv")
  else:
    print("Scan complete. No valid SMC setups found matching criteria.")


if __name__ == "__main__":
  run_smc_scanner()
