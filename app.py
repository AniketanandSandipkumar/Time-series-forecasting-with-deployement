# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Import custom modules
from train_models import train_models   # updated import
from eda_utils import plot_resampling, plot_rolling, plot_decomposition


# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Stock Forecasting", layout="wide")
st.title("📈 Time Series Forecasting Dashboard")


# --------------------------
# File Upload
# --------------------------
uploaded_file = st.sidebar.file_uploader("Upload Stock CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)

    # --------------------------
    # Dataset View with Date Filter
    # --------------------------
    st.subheader("📂 Dataset Viewer")
    min_date, max_date = df.index.min(), df.index.max()
    start_date = st.date_input(
        "Choose start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    df_filtered = df.loc[start_date:]
    st.dataframe(df_filtered)

    # --------------------------
    # Sidebar Options
    # --------------------------
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

        # --------------------------
        # Show Results
        # --------------------------
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

            # --------------------------
            # Download Forecasted Data
            # --------------------------
            # Build downloadable DataFrame
            y_true = y_true.flatten() if hasattr(y_true, "flatten") else y_true
            y_pred = y_pred.flatten() if hasattr(y_pred, "flatten") else y_pred

            pred_df = pd.DataFrame({
              "Date": pd.to_datetime(x),
              "Actual": y_true,
              "Predicted": y_pred
            })

            csv_preds = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
            f"📥 Download {model} Predictions for {feature}",
            csv_preds,
            file_name=f"{feature}_{model}_predictions.csv",
            mime="text/csv"
            )


    # --------------------------
    # EDA Tools
    # --------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Exploratory Data Analysis")

    if st.sidebar.checkbox("Show Resampling (Monthly Mean)"):
        st.plotly_chart(plot_resampling(df), use_container_width=True)

    if st.sidebar.checkbox("Show Rolling Mean (30 days)"):
        st.plotly_chart(plot_rolling(df), use_container_width=True)

    if st.sidebar.checkbox("Show Decomposition (Trend/Seasonality/Residuals)"):
        feature_for_decomp = st.sidebar.selectbox("Select Feature for Decomposition", df.columns.tolist())
        st.plotly_chart(plot_decomposition(df, feature_for_decomp), use_container_width=True)


