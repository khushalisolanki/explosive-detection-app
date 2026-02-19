import streamlit as st
from utils import process_data, add_slope, plot_raw, plot_slope, detect_explosive

st.set_page_config(layout="wide")

st.title("Explosive Sensor Intelligence Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload Sensor CSV", type="csv")

if uploaded_file:

    # Process data
    df = process_data(uploaded_file)
    df = add_slope(df)

    st.sidebar.header("Controls")

    plot_type = st.sidebar.radio(
        "Select Plot Type",
        ["Raw Signal", "Slope Plot", "Both"]
    )

    # SHOW PLOTS
    if plot_type == "Raw Signal":
        fig = plot_raw(df)
        st.pyplot(fig)

    elif plot_type == "Slope Plot":
        fig = plot_slope(df)
        st.pyplot(fig)

    else:
        st.subheader("Raw Signal")
        st.pyplot(plot_raw(df))

        st.subheader("Slope Plot")
        st.pyplot(plot_slope(df))

    # Detection result
    st.subheader("Detection Result")

    result = detect_explosive(df)

    if "Explosive" in result:
        st.error(result)
    else:
        st.success(result)

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

