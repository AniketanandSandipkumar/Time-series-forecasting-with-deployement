# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Import custom modules
from train_models import train_all_models
from eda_utils import plot_resampled, plot_rolling, plot_decomposition

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Stock Forecasting App", layout="wide")
st.title("📈 Time Series Forecasting with ARIMA, SARIMA, Prophet & LSTM")

# Sidebar for user input
st.sidebar.header("User Options")
uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type=["csv"])
default_features = ["Open", "High", "Low", "Close", "Volume"]

# --------------------------
# Load Dataset
# --------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    st.subheader("📊 Raw Data")
    st.write(df.head())

    # Plot Closing Price trend
    st.subheader("📈 Stock Closing Price Trend")
    fig = px.line(df, x=df.index, y="Close", title="Closing Price Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # EDA Section
    # --------------------------
    st.subheader("🔎 Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("Resampled (Monthly)")
        fig1 = plot_resampled(df, "Close", rule="M")
        st.pyplot(fig1)

    with col2:
        st.write("Rolling Mean & Std")
        fig2 = plot_rolling(df, "Close", window=30)
        st.pyplot(fig2)

    with col3:
        st.write("Decomposition (Trend/Seasonality/Residuals)")
        fig3 = plot_decomposition(df, "Close", model="additive", freq=30)
        st.pyplot(fig3)

    # --------------------------
    # Model Training & Results
    # --------------------------
    st.subheader("🤖 Model Training & Evaluation")

    if st.button("Run All Models"):
        with st.spinner("Training models... this may take a while ⏳"):
            results_df, predictions = train_all_models(df, default_features)

        st.success("✅ Training complete!")

        st.write("📊 Model Performance Metrics")
        st.dataframe(results_df)

        # Visualization of predictions
        st.subheader("📉 Model Predictions vs Actuals")
        for feature in predictions.keys():
            st.markdown(f"### Feature: **{feature}**")
            for model_name, (x, y_true, y_pred) in predictions[feature].items():
                fig = px.line(x=x, y=y_true.squeeze(), labels={'x':'Date','y':'Value'}, title=f"{model_name} - {feature}")
                fig.add_scatter(x=x, y=pd.Series(y_pred).squeeze(), mode="lines", name="Prediction")
                st.plotly_chart(fig, use_container_width=True)

