import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Trading Scanner Dashboard", page_icon="📈", layout="wide"
)

LOG_FILE = "trade_log.csv"

st.title("📈 AI Multi-Asset Trading Scanner Dashboard")
st.markdown(
    "Control live indicators with on/off toggles, adjust confidence filters,"
    " and analyze trade logs in real time."
)


@st.cache_data(ttl=10)
def load_data():
  if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    
    # Prioritize the lowercase 'confidence' column where the actual data lives
    if "confidence" in df.columns:
      df["Confidence_%"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
    elif "Confidence_%" in df.columns:
      df["Confidence_%"] = pd.to_numeric(df["Confidence_%"], errors="coerce").fillna(0)
    else:
      df["Confidence_%"] = 0.0
      
    return df
  return pd.DataFrame()


df = load_data()

if df.empty:
  st.warning(
      "⚠️ No trade logs found yet. Run your GitHub Actions workflow to"
      " generate signals!"
  )
else:
  # --- SIDEBAR CONTROLS & FILTERS ---
  st.sidebar.header("🎛️ Indicator & Filter Controls")

  # 1. Confidence Threshold
  selected_conf = st.sidebar.slider(
      "Minimum AI Confidence (%)",
      min_value=0.0,
      max_value=100.0,
      value=0.0,
      step=1.0,
  )

  # 2. Asset Multi-select Filter
  all_assets = list(df["Asset"].unique()) if "Asset" in df.columns else []
  selected_assets = st.sidebar.multiselect(
      "Filter by Assets", all_assets, default=all_assets
  )

  # 3. Direction Filter
  directions = (
      ["ALL"] + list(df["Direction"].unique())
      if "Direction" in df.columns
      else ["ALL"]
  )
  selected_direction = st.sidebar.selectbox("Filter by Direction", directions)

  # 4. Signal Type Filter
  signal_col = (
      "signal"
      if "signal" in df.columns
      else ("signal_type" if "signal_type" in df.columns else None)
  )
  if signal_col:
    signals_list = ["ALL"] + list(df[signal_col].dropna().unique())
    selected_signal = st.sidebar.selectbox("Filter by Signal Type", signals_list)
  else:
    selected_signal = "ALL"

  # 5. Indicator On/Off Toggles and Parameters
  st.sidebar.subheader("📊 Indicator Controls & Toggles")

  use_rsi = st.sidebar.checkbox("Enable RSI Indicator", value=True)
  rsi_period = 14
  if use_rsi:
    rsi_period = st.sidebar.slider("RSI Period", min_value=5, max_value=30, value=14)

  use_macd = st.sidebar.checkbox("Enable MACD Indicator", value=True)
  macd_fast, macd_slow = 12, 26
  if use_macd:
    macd_fast = st.sidebar.slider("MACD Fast Period", min_value=5, max_value=20, value=12)
    macd_slow = st.sidebar.slider("MACD Slow Period", min_value=21, max_value=40, value=26)

  use_bb = st.sidebar.checkbox("Enable Bollinger Bands", value=True)
  bb_period = 20
  if use_bb:
    bb_period = st.sidebar.slider("Bollinger Bands Period", min_value=10, max_value=50, value=20)

  use_ema = st.sidebar.checkbox("Enable EMA (Exponential Moving Average)", value=True)
  ema_period = 50
  if use_ema:
    ema_period = st.sidebar.slider("EMA Period", min_value=10, max_value=200, value=50)

  # --- APPLY FILTERS ---
  filtered_df = df.copy()

  if "Asset" in filtered_df.columns and selected_assets:
    filtered_df = filtered_df[filtered_df["Asset"].isin(selected_assets)]

  if "Direction" in filtered_df.columns and selected_direction != "ALL":
    filtered_df = filtered_df[filtered_df["Direction"] == selected_direction]

  if signal_col and selected_signal != "ALL":
    filtered_df = filtered_df[filtered_df[signal_col] == selected_signal]

  if "Confidence_%" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Confidence_%"] >= selected_conf]

  # --- TOP METRICS ROW ---
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Signals Logged", len(df))
  col2.metric("Filtered Signals Shown", len(filtered_df))
  if not filtered_df.empty:
    avg_conf = filtered_df["Confidence_%"].mean()
    col3.metric("Avg Filtered Confidence", f"{avg_conf:.1f}%")
  else:
    col3.metric("Avg Filtered Confidence", "0.0%")
  col4.metric("Active Assets Tracked", len(all_assets))

  st.divider()

  # --- ACTIVE INDICATOR STATUS DISPLAY ---
  st.subheader("⚙️ Active Technical Indicators Configuration")
  active_indicators = []
  if use_rsi:
    active_indicators.append(f"RSI (Period: {rsi_period})")
  if use_macd:
    active_indicators.append(f"MACD (Fast: {macd_fast}, Slow: {macd_slow})")
  if use_bb:
    active_indicators.append(f"Bollinger Bands (Period: {bb_period})")
  if use_ema:
    active_indicators.append(f"EMA (Period: {ema_period})")

  if active_indicators:
    st.info("**Currently Enabled Indicators:** " + " | ".join(active_indicators))
  else:
    st.warning("⚠️ All technical indicators are currently disabled.")

  # --- MAIN INTERACTIVE TABLE ---
  st.subheader("📊 Interactive Trade Log Explorer")
  st.dataframe(filtered_df, use_container_width=True)

  # --- CHARTS & VISUALIZATIONS ---
  if not filtered_df.empty:
    st.subheader("📈 AI Confidence Breakdown by Asset")
    st.bar_chart(filtered_df, x="Asset", y="Confidence_%")
