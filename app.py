import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Explosive Sensor Intelligence Dashboard")


# ─────────────────────────────────────────────
# DATA PROCESSING  (was process_data in utils)
# ─────────────────────────────────────────────
def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Map event strings → numeric codes on the NEXT row (matches original logic)
    event_map = {
        "IRLED Turned On":  1,
        "IRLED Turned Off": 2,
        "Pump Turned On":   3,
        "Pump Turned Off":  4,
    }
    for event_str, code in event_map.items():
        indices = df[df["Event"] == event_str].index
        for idx in indices:
            if idx + 1 in df.index:
                df.at[idx + 1, "Event"] = code

    # Drop rows where ALL sensor channels are '--NA--'
    sensor_cols = ["Channel 1(ohms)", "Channel 2(ohms)", "Channel 3(ohms)", "Channel 4(ohms)"]
    df = df[~(df[sensor_cols] == "--NA--").all(axis=1)]

    # Convert sensor columns to numeric
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert Event to numeric, fill remaining strings with 0
    df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# SLOPE CALCULATION  (per-channel np.diff)
# ─────────────────────────────────────────────
def compute_slope(df, col):
    """Returns (time_midpoints, slope_values) for a channel."""
    time_vals = df["Time(secs)"].values
    raw_vals  = df[col].values
    valid     = ~np.isnan(raw_vals)
    t = time_vals[valid]
    r = raw_vals[valid]
    if len(t) < 2:
        return np.array([]), np.array([])
    slope     = np.diff(r) / np.diff(t)
    t_mid     = (t[:-1] + t[1:]) / 2
    return t_mid, slope


# ─────────────────────────────────────────────
# VERTICAL LINE HELPERS
# ─────────────────────────────────────────────
def add_event_lines(fig, df, row, col):
    """Add red dashed (IR) and blue dashed (Pump) vertical lines."""
    ir_times   = df[df["Event"].isin([1, 2])]["Time(secs)"]
    pump_times = df[df["Event"].isin([3, 4])]["Time(secs)"]

    for t in ir_times:
        fig.add_vline(x=t, line=dict(color="red",  dash="dash", width=1), row=row, col=col)
    for t in pump_times:
        fig.add_vline(x=t, line=dict(color="blue", dash="dash", width=1), row=row, col=col)


# ─────────────────────────────────────────────
# PLOT BUILDERS
# ─────────────────────────────────────────────
CHANNELS = {
    "Channel 2(ohms)": "Channel 2 (Coating: PAB)",
    "Channel 3(ohms)": "Channel 3 (Coating: PEG)",
    "Channel 4(ohms)": "Channel 4 (Coating: Pristine)",
}

def plot_raw(df, file_title=""):
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(CHANNELS.values()))

    for i, (col, label) in enumerate(CHANNELS.items(), start=1):
        fig.add_trace(
            go.Scatter(x=df["Time(secs)"], y=df[col],
                       mode="lines", name=label,
                       line=dict(width=1.5)),
            row=1, col=i
        )
        add_event_lines(fig, df, row=1, col=i)
        fig.update_yaxes(tickformat=".0f", exponentformat="none", row=1, col=i)
        fig.update_xaxes(title_text="Time (secs)", row=1, col=i)

    fig.update_layout(
        title_text=f"Raw Signal — {file_title}" if file_title else "Raw Signal",
        height=450,
        showlegend=False,
    )
    return fig


def plot_slope(df, file_title=""):
    slope_titles = [f"Slope — {v}" for v in CHANNELS.values()]
    fig = make_subplots(rows=1, cols=3, subplot_titles=slope_titles)

    for i, (col, label) in enumerate(CHANNELS.items(), start=1):
        t_mid, slope = compute_slope(df, col)
        if len(t_mid) == 0:
            continue
        fig.add_trace(
            go.Scatter(x=t_mid, y=slope,
                       mode="lines", name=f"Slope {label}",
                       line=dict(width=1.5, color="darkorange")),
            row=1, col=i
        )
        add_event_lines(fig, df, row=1, col=i)
        fig.update_yaxes(tickformat=".4f", exponentformat="none", row=1, col=i)
        fig.update_xaxes(title_text="Time (secs)", row=1, col=i)

    fig.update_layout(
        title_text=f"Slope Plot — {file_title}" if file_title else "Slope Plot",
        height=450,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# EXPLOSIVE DETECTION  (was detect_explosive)
# ─────────────────────────────────────────────
def detect_explosive(df):
    """
    Simple heuristic: check if any channel drops significantly
    between IR LED ON and IR LED OFF events.
    Replace this logic with your actual detection algorithm.
    """
    ir_on_times  = df[df["Event"] == 1]["Time(secs)"].values
    ir_off_times = df[df["Event"] == 2]["Time(secs)"].values

    if len(ir_on_times) == 0 or len(ir_off_times) == 0:
        return "Insufficient event data for detection"

    detected_channels = []
    for col in CHANNELS:
        for t_on, t_off in zip(ir_on_times, ir_off_times):
            window = df[(df["Time(secs)"] >= t_on) & (df["Time(secs)"] <= t_off)][col].dropna()
            if len(window) < 2:
                continue
            pct_change = abs((window.iloc[-1] - window.iloc[0]) / window.iloc[0]) * 100
            if pct_change > 5:   # >5% change during IR window → flag
                detected_channels.append(col)
                break

    if detected_channels:
        return f" Explosive signature detected on: {', '.join(detected_channels)}"
    return " No explosive detected"


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Sensor CSV", type="csv")

if uploaded_file:
    df = process_data(uploaded_file)
    file_title = uploaded_file.name.replace(".csv", "")

    st.sidebar.header("Controls")

    plot_type = st.sidebar.radio(
        "Select Plot Type",
        ["Raw Signal", "Slope Plot", "Both"]
    )

    selected_channels = st.sidebar.multiselect(
        "Select Channels to Display",
        list(CHANNELS.keys()),
        default=list(CHANNELS.keys()),
    )

    show_data = st.sidebar.checkbox("Show Data Preview")

    # Legend info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Event Line Legend**")
    st.sidebar.markdown(" Red dashed = IR LED On / Off")
    st.sidebar.markdown(" Blue dashed = Pump On / Off")

    # ── PLOTS ──
    # Filter CHANNELS to only selected
    active_channels = {k: v for k, v in CHANNELS.items() if k in selected_channels}

    if plot_type in ["Raw Signal", "Both"]:
        st.subheader("Raw Signal")
        fig = plot_raw(df, file_title)
        st.plotly_chart(fig, use_container_width=True)

    if plot_type in ["Slope Plot", "Both"]:
        st.subheader("Slope Plot")
        fig = plot_slope(df, file_title)
        st.plotly_chart(fig, use_container_width=True)

    # ── DETECTION ──
    st.subheader("Detection Result")
    result = detect_explosive(df)
    if "Explosive signature" in result:
        st.error(result)
    else:
        st.success(result)

    # ── EVENT SUMMARY TABLE ──
    st.subheader("Event Log")
    event_labels = {0: "—", 1: "IR LED On", 2: "IR LED Off", 3: "Pump On", 4: "Pump Off"}
    event_df = df[df["Event"] != 0][["Time(secs)", "Event"]].copy()
    event_df["Event Label"] = event_df["Event"].map(event_labels)
    st.dataframe(event_df.reset_index(drop=True), use_container_width=True)

    # ── DATA PREVIEW ──
    if show_data:
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
