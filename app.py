import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Trading Scanner Dashboard", page_icon="📈", layout="wide"
)

LOG_FILE = "trade_log.csv"

st.title("📈 AI Multi-Asset Trading Scanner Dashboard")
st.markdown(
    "Monitor live AI-generated signals, adjust confidence thresholds, and filter"
    " by asset and direction in real time."
)


@st.cache_data(ttl=30)
def load_data():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
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
    st.sidebar.header("🎛️ Dashboard Controls & Filters")

    # Confidence Threshold Slider
    if "Confidence_%" in df.columns:
        selected_conf = st.sidebar.slider(
            "Minimum AI Confidence (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
        )
    else:
        selected_conf = 0.0

    # Direction Filter
    directions = (
        ["ALL"] + list(df["Direction"].unique())
        if "Direction" in df.columns
        else ["ALL"]
    )
    selected_direction = st.sidebar.selectbox(
        "Filter by Direction", directions
    )

    # Asset Multi-select Filter
    all_assets = list(df["Asset"].unique()) if "Asset" in df.columns else []
    selected_assets = st.sidebar.multiselect(
        "Filter by Assets", all_assets, default=all_assets
    )

    # --- APPLY FILTERS ---
    filtered_df = df.copy()

    if "Asset" in filtered_df.columns and selected_assets:
        filtered_df = filtered_df[filtered_df["Asset"].isin(selected_assets)]

    if "Direction" in filtered_df.columns and selected_direction != "ALL":
        filtered_df = filtered_df[filtered_df["Direction"] == selected_direction]

    if "Confidence_%" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Confidence_%"] >= selected_conf]

    # --- TOP METRICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Signals Logged", len(df))
    col2.metric("Filtered Signals Shown", len(filtered_df))
    if "Confidence_%" in filtered_df.columns and not filtered_df.empty:
        avg_conf = filtered_df["Confidence_%"].mean()
        col3.metric("Avg Filtered Confidence", f"{avg_conf:.1f}%")
    else:
        col3.metric("Avg Filtered Confidence", "N/A")
    col4.metric("Active Assets Tracked", len(all_assets))

    st.divider()

    # --- MAIN INTERACTIVE TABLE ---
    st.subheader("📊 Interactive Trade Log Explorer")
    st.dataframe(filtered_df, use_container_width=True)

    # --- CHARTS & VISUALIZATIONS ---
    if not filtered_df.empty and "Confidence_%" in filtered_df.columns:
        st.subheader("📈 AI Confidence Breakdown by Asset")
        st.bar_chart(filtered_df, x="Asset", y="Confidence_%")
