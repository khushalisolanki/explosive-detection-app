import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CLEAN + PROCESS DATA
# -----------------------------
def process_data(uploaded_file):

    df = pd.read_csv(uploaded_file)

    # Convert columns to numeric
    cols = [
        'Time(secs)',
        'Channel 1(ohms)',
        'Channel 2(ohms)',
        'Channel 3(ohms)',
        'Channel 4(ohms)'
    ]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0)

    return df


# -----------------------------
# SLOPE CALCULATION
# -----------------------------
def add_slope(df):

    channels = [
        'Channel 1(ohms)',
        'Channel 2(ohms)',
        'Channel 3(ohms)',
        'Channel 4(ohms)'
    ]

    for ch in channels:
        df[ch + "_slope"] = np.gradient(df[ch])

    return df


# -----------------------------
# RAW PLOT
# -----------------------------
def plot_raw(df):

    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes = axes.flatten()

    channels = [
        'Channel 1(ohms)',
        'Channel 2(ohms)',
        'Channel 3(ohms)',
        'Channel 4(ohms)'
    ]

    for i, ch in enumerate(channels):
        axes[i].plot(df["Time(secs)"], df[ch])
        axes[i].set_title(ch)

        # Highlight gas region
        if 3 in df["Event"].values and 4 in df["Event"].values:
            start = df[df["Event"]==3]["Time(secs)"].iloc[0]
            end = df[df["Event"]==4]["Time(secs)"].iloc[0]
            axes[i].axvspan(start, end, alpha=0.3)

    return fig


# -----------------------------
# SLOPE PLOT
# -----------------------------
def plot_slope(df):

    fig, axes = plt.subplots(2, 2, figsize=(12,8))
    axes = axes.flatten()

    channels = [
        'Channel 1(ohms)_slope',
        'Channel 2(ohms)_slope',
        'Channel 3(ohms)_slope',
        'Channel 4(ohms)_slope'
    ]

    for i, ch in enumerate(channels):
        axes[i].plot(df["Time(secs)"], df[ch])
        axes[i].set_title(ch)

    return fig


# -----------------------------
# GAS DETECTION
# -----------------------------
def detect_explosive(df):

    if not (3 in df["Event"].values and 4 in df["Event"].values):
        return "No Gas Window Found"

    start = df[df["Event"]==3].index[0]
    end = df[df["Event"]==4].index[0]

    df2 = df.loc[start:end]

    r0 = df2.iloc[0]

    diff2 = abs(df2["Channel 2(ohms)"].max() - r0["Channel 2(ohms)"])
    diff3 = abs(df2["Channel 3(ohms)"].max() - r0["Channel 3(ohms)"])
    diff4 = abs(df2["Channel 4(ohms)"].max() - r0["Channel 4(ohms)"])

    if diff2 > 80:
        return "Explosive Detected → Channel 2"
    elif diff3 > 20:
        return "Explosive Detected → Channel 3"
    elif diff4 > 10:
        return "Explosive Detected → Channel 4"
    else:
        return "No Explosive Detected"
