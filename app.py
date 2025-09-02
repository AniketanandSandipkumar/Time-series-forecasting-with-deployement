# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Import custom modules
from train_models import train_models   # updated import
from eda_utils import plot_resampling, plot_rolling, plot_decomposition

st.set_page_config(page_title="Stock Forecasting", layout="wide")
st.title("📈 Time Series Forecasting Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Stock CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

   
    features = st.sidebar.multiselect(
        "Select Features for Forecasting",
        options=df.columns.tolist(),
        default=["Close"]
    )

    models_to_run = st.sidebar.multiselect(
        "Select Models",
        ["ARIMA", "SARIMA", "Prophet", "LSTM"],
        default=["ARIMA"]  # let’s start with ARIMA by default
    )

    test_ratio = st.sidebar.slider("Test Data Ratio", 0.1, 0.4, 0.2, step=0.05)

    if st.sidebar.button("Run Forecasting"):
        with st.spinner("Training selected models..."):
            results_df, predictions = train_models(df, features, models_to_run=models_to_run, test_ratio=test_ratio)

        st.success("Forecasting complete ✅")

      
        st.subheader("📊 Model Evaluation Results")
        st.dataframe(results_df)

        # Plot actual vs predicted
        for feature in features:
            st.subheader(f"Feature: {feature}")
            for model_name, (x, actual, pred) in predictions[feature].items():
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(x, actual, label="Actual", color="black")
                ax.plot(x, pred, label=f"{model_name} Prediction")
                ax.legend()
                st.pyplot(fig)

  
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Exploratory Data Analysis")

    if st.sidebar.checkbox("Show Resampling (Monthly Mean)"):
        st.plotly_chart(plot_resampling(df), use_container_width=True)

    if st.sidebar.checkbox("Show Rolling Mean (30 days)"):
        st.plotly_chart(plot_rolling(df), use_container_width=True)

    if st.sidebar.checkbox("Show Decomposition (Trend/Seasonality/Residuals)"):
        feature_for_decomp = st.sidebar.selectbox("Select Feature for Decomposition", df.columns.tolist())
        st.plotly_chart(plot_decomposition(df, feature_for_decomp), use_container_width=True)

