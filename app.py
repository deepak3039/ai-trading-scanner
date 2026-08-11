import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Trading Bot Dashboard", page_icon="🤖", layout="wide"
)

st.title("🤖 AI Trading Scanner & Telemetry Dashboard")
st.write(
    "Live paper trading analytics and multi-asset signals powered by Machine"
    " Learning & Smart Money Concepts."
)

LOG_FILE = "trade_log.csv"

if not os.path.exists(LOG_FILE):
  st.warning(
      "⚠️ No `trade_log.csv` file found yet. Let your GitHub Actions workflow"
      " run to populate trade data."
  )
else:
  df = pd.read_csv(LOG_FILE)

  if df.empty:
    st.warning("⚠️ `trade_log.csv` is currently empty.")
  else:
    # --- Top Metrics Row ---
    st.subheader("📌 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    total_trades = len(df)
    unique_assets = df["Asset"].nunique()
    avg_confidence = (
        df["Confidence_%"].mean() if "Confidence_%" in df.columns else 0
    )
    executed_trades = (
        len(df[df["Status"] == "EXECUTED"]) if "Status" in df.columns else 0
    )

    col1.metric("Total Signals Logged", total_trades)
    col2.metric("Assets Monitored", unique_assets)
    col3.metric("Avg AI Confidence", f"{avg_confidence:.1f}%")
    col4.metric("Testnet Orders Executed", executed_trades)

    st.divider()

    # --- Charts Section ---
    col_left, col_right = st.columns(2)

    with col_left:
      st.subheader("📊 Bullish vs Bearish Ratio")
      if "Direction" in df.columns:
        fig_dir = px.pie(
            df,
            names="Direction",
            title="Signal Direction Distribution",
            color="Direction",
            color_discrete_map={"BULLISH": "#00CC96", "BEARISH": "#EF553B"},
            hole=0.4,
        )
        st.plotly_chart(fig_dir, use_container_width=True)

    with col_right:
      st.subheader("📈 Signal Volume by Asset")
      if "Asset" in df.columns:
        asset_counts = df["Asset"].value_counts().reset_index()
        asset_counts.columns = ["Asset", "Count"]
        fig_asset = px.bar(
            asset_counts,
            x="Asset",
            y="Count",
            title="Signals Detected Per Asset",
            color="Count",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_asset, use_container_width=True)

    st.divider()

    # --- Data Filter & Table ---
    st.subheader("📋 Interactive Trade Log Explorer")

    selected_asset = st.multiselect(
        "Filter by Asset:",
        options=df["Asset"].unique(),
        default=df["Asset"].unique(),
    )
    filtered_df = df[df["Asset"].isin(selected_asset)]

    st.dataframe(filtered_df, use_container_width=True)
