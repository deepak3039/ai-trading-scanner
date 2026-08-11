import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Trading Scanner Dashboard", page_icon="📈", layout="wide"
)

LOG_FILE = "trade_log.csv"

st.title("📈 AI Multi-Asset Trading Scanner Dashboard")
st.markdown(
    "Control live indicators, adjust confidence filters, and analyze trade"
    " logs in real time."
)


@st.cache_data(ttl=10)
def load_data():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        # Normalize/Clean Confidence columns to prevent NaN filtering bugs
        if "Confidence_%" in df.columns:
            df["Confidence_%"] = pd.to_numeric(
                df["Confidence_%"], errors="coerce"
            ).fillna(0)
        elif "confidence" in df.columns:
            df["Confidence_%"] = (
                pd.to_numeric(df["confidence"], errors="coerce").fillna(0)
                * 100
            )
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
    selected_direction = st.sidebar.selectbox(
        "Filter by Direction", directions
    )

    # 4. Signal Type Filter
    signal_col = (
        "signal"
        if "signal" in df.columns
        else ("signal_type" if "signal_type" in df.columns else None)
    )
    if signal_col:
        signals_list = ["ALL"] + list(df[signal_col].dropna().unique())
        selected_signal = st.sidebar.selectbox(
            "Filter by Signal Type", signals_list
        )
    else:
        selected_signal = "ALL"

    # 5. Technical Indicator Settings (Custom controls for scanner parameters)
    st.sidebar.subheader("📊 Indicator Parameters")
    rsi_period = st.sidebar.slider(
        "RSI Period", min_value=5, max_value=30, value=14
    )
    macd_fast = st.sidebar.slider(
        "MACD Fast Period", min_value=5, max_value=20, value=12
    )
    macd_slow = st.sidebar.slider(
        "MACD Slow Period", min_value=21, max_value=40, value=26
    )

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

    # --- MAIN INTERACTIVE TABLE ---
    st.subheader("📊 Interactive Trade Log Explorer")
    st.dataframe(filtered_df, use_container_width=True)

    # --- CHARTS & VISUALIZATIONS ---
    if not filtered_df.empty:
        st.subheader("📈 AI Confidence Breakdown by Asset")
        st.bar_chart(filtered_df, x="Asset", y="Confidence_%")
