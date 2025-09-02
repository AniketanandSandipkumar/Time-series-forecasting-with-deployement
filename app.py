# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from train_models import run_all_models   # your training + forecasting pipeline
from eda_utils import plot_resampled, plot_rolling, plot_decomposition

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Stock Forecasting Dashboard", layout="wide")

st.title("📈 Stock Forecasting with ARIMA, SARIMA, Prophet, and LSTM")

# Sidebar
st.sidebar.header("Upload or Load Data")
upload_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if upload_file:
    df = pd.read_csv(upload_file, parse_dates=True, index_col=0)
else:
    st.sidebar.info("Using sample dataset")
    # Load a fallback dataset (replace with your own)
    df = pd.read_csv("sample_stock.csv", parse_dates=True, index_col=0)

st.write("### Raw Dataset Preview")
st.dataframe(df.head())

# --------------------------
# EDA Section
# --------------------------
st.header("🔍 Exploratory Data Analysis")

feature = st.selectbox("Select Feature for EDA", df.columns)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Resampled Trend (Monthly)")
    fig1 = plot_resampled(df, feature, rule="M")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Rolling Mean & Std (30-day)")
    fig2 = plot_rolling(df, feature, window=30)
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Seasonal Decomposition")
fig3 = plot_decomposition(df, feature, period=30)
st.pyplot(fig3)

# --------------------------
# Model Training & Forecasting
# --------------------------
st.header("🤖 Train & Forecast")

models = ["ARIMA", "SARIMA", "Prophet", "LSTM"]
selected_models = st.multiselect("Select Models to Run", models, default=models)

if st.button("Run Models"):
    with st.spinner("Training models... this may take a while ⏳"):
        results_df, forecast_plots = run_all_models(df, selected_models)

    st.success("Training Complete ✅")

    st.subheader("📊 Model Evaluation Results")
    st.dataframe(results_df)

    # Plot forecasts
    for model_name, fig in forecast_plots.items():
        st.subheader(f"Forecast - {model_name}")
        st.pyplot(fig)

# --------------------------
# Download Results
# --------------------------
if 'results_df' in locals():
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Results as CSV",
        data=csv,
        file_name="model_results.csv",
        mime="text/csv"
    )
