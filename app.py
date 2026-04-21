import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
st.title("Explosive Sensor Intelligence Dashboard")


# ─────────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────────
def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

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

    sensor_cols = ["Channel 1(ohms)", "Channel 2(ohms)", "Channel 3(ohms)", "Channel 4(ohms)"]
    df = df[~(df[sensor_cols] == "--NA--").all(axis=1)]

    for col in ["Time(secs)"] + sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# SLOPE CALCULATION
# ─────────────────────────────────────────────
def compute_slope(df, col):
    time_vals = df["Time(secs)"].values
    raw_vals  = df[col].values
    valid     = ~np.isnan(raw_vals)
    t = time_vals[valid]
    r = raw_vals[valid]
    if len(t) < 2:
        return np.array([]), np.array([])
    slope = np.diff(r) / np.diff(t)
    t_mid = (t[:-1] + t[1:]) / 2
    return t_mid, slope


# ─────────────────────────────────────────────
# VERTICAL LINE HELPERS
# ─────────────────────────────────────────────
def add_event_lines(fig, df, row, col):
    ir_times   = df[df["Event"].isin([1, 2])]["Time(secs)"]
    pump_times = df[df["Event"].isin([3, 4])]["Time(secs)"]
    for t in ir_times:
        fig.add_vline(x=t, line=dict(color="red",  dash="dash", width=1), row=row, col=col)
    for t in pump_times:
        fig.add_vline(x=t, line=dict(color="blue", dash="dash", width=1), row=row, col=col)


# ─────────────────────────────────────────────
# CHANNEL CONFIG & THRESHOLDS
# ─────────────────────────────────────────────
CHANNELS = {
    "Channel 2(ohms)": "Channel 2 (Coating: PAB)",
    "Channel 3(ohms)": "Channel 3 (Coating: PEG)",
    "Channel 4(ohms)": "Channel 4 (Coating: Pristine)",
}

THRESHOLDS = {
    "Channel 2(ohms)": 100,
    "Channel 3(ohms)": 20,
    "Channel 4(ohms)": 10,
}


# ─────────────────────────────────────────────
# REAL EXPLOSIVE DETECTION
# Logic ported directly from original detection script:
#   - Window = Pump On (Event=3) → Pump Off (Event=4)
#   - R0 = channel value at pump-on row
#   - Detect if (max_value − R0) > threshold per channel
# Bug fixed: original used undefined max_index_channel_X → now uses idxmax()
# Enhancement: handles multiple pump cycles in one file
# ─────────────────────────────────────────────
def detect_explosive(df):
    pump_on_rows  = df[df["Event"] == 3]
    pump_off_rows = df[df["Event"] == 4]

    if pump_on_rows.empty or pump_off_rows.empty:
        return pd.DataFrame([{
            "Cycle": "—", "Channel": "—", "R0 (Ohms)": None,
            "Max (Ohms)": None, "ΔR (Max−R0)": None,
            "Threshold": None, "Explosive": False,
            "_explosive_flag": False,
        }]), False

    rows = []
    on_indices  = pump_on_rows.index.tolist()
    off_indices = pump_off_rows.index.tolist()

    for cycle_num, (start_idx, end_idx) in enumerate(zip(on_indices, off_indices), start=1):
        if end_idx <= start_idx:
            continue
        df2 = df.loc[start_idx:end_idx]

        for col, label in CHANNELS.items():
            r0_value  = df2.loc[start_idx, col]
            max_value = df2[col].max()
            diff      = max_value - r0_value
            threshold = THRESHOLDS[col]
            explosive = bool(pd.notna(diff) and diff > threshold)

            rows.append({
                "Cycle":       cycle_num,
                "Channel":     label,
                "R0 (Ohms)":   round(r0_value, 2) if pd.notna(r0_value) else None,
                "Max (Ohms)":  round(max_value, 2) if pd.notna(max_value) else None,
                "ΔR (Max−R0)": round(diff, 2)      if pd.notna(diff)     else None,
                "Threshold":   threshold,
                "Explosive":   explosive,
            })

    result_df    = pd.DataFrame(rows)
    any_explosive = result_df["Explosive"].any()
    return result_df, any_explosive


# ─────────────────────────────────────────────
# PLOT BUILDERS
# ─────────────────────────────────────────────
def plot_raw(df, file_title=""):
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(CHANNELS.values()))
    for i, (col, label) in enumerate(CHANNELS.items(), start=1):
        fig.add_trace(
            go.Scatter(x=df["Time(secs)"], y=df[col],
                       mode="lines", name=label, line=dict(width=1.5)),
            row=1, col=i,
        )
        add_event_lines(fig, df, row=1, col=i)
        fig.update_yaxes(tickformat=".0f", exponentformat="none", row=1, col=i)
        fig.update_xaxes(title_text="Time (secs)", row=1, col=i)
    fig.update_layout(
        title_text=f"Raw Signal — {file_title}" if file_title else "Raw Signal",
        height=450, showlegend=False,
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
            row=1, col=i,
        )
        add_event_lines(fig, df, row=1, col=i)
        fig.update_yaxes(tickformat=".4f", exponentformat="none", row=1, col=i)
        fig.update_xaxes(title_text="Time (secs)", row=1, col=i)
    fig.update_layout(
        title_text=f"Slope Plot — {file_title}" if file_title else "Slope Plot",
        height=450, showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Sensor CSV", type="csv")

if uploaded_file:
    df = process_data(uploaded_file)
    file_title = uploaded_file.name.replace(".csv", "")

    st.sidebar.header("Controls")
    plot_type = st.sidebar.radio("Select Plot Type", ["Raw Signal", "Slope Plot", "Both"])
    show_data = st.sidebar.checkbox("Show Data Preview")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Event Line Legend**")
    st.sidebar.markdown(" Red dashed = IR LED On / Off")
    st.sidebar.markdown(" Blue dashed = Pump On / Off")

    # ── PLOTS ──
    if plot_type in ["Raw Signal", "Both"]:
        st.subheader("Raw Signal")
        st.plotly_chart(plot_raw(df, file_title), use_container_width=True)

    if plot_type in ["Slope Plot", "Both"]:
        st.subheader("Slope Plot")
        st.plotly_chart(plot_slope(df, file_title), use_container_width=True)

    # ── DETECTION ──
    st.subheader("Detection Result")
    result_df, any_explosive = detect_explosive(df)

    if any_explosive:
        flagged = result_df[result_df["Explosive"]]["Channel"].tolist()
        st.error(f" Explosive Detected on: {', '.join(flagged)}")
    else:
        st.success(" No Explosive Detected")

    # Highlight explosive rows
    def highlight_explosive(row):
        color = "background-color: #ffcccc" if row["Explosive"] else ""
        return [color] * len(row)

    st.dataframe(
        result_df.style.apply(highlight_explosive, axis=1),
        use_container_width=True,
    )

    st.download_button(
        " Download Detection Report",
        result_df.to_csv(index=False),
        file_name="explosive_detection_report.csv",
        mime="text/csv",
    )

    # ── EVENT LOG ──
    st.subheader("Event Log")
    event_labels = {0: "—", 1: "IR LED On", 2: "IR LED Off", 3: "Pump On", 4: "Pump Off"}
    event_df = df[df["Event"] != 0][["Time(secs)", "Event"]].copy()
    event_df["Event Label"] = event_df["Event"].map(event_labels)
    st.dataframe(event_df.reset_index(drop=True), use_container_width=True)

    # ── DATA PREVIEW ──
    if show_data:
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
