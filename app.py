# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Import custom modules
from train_models import train_all_models
from eda_utils import plot_resampling, plot_rolling, plot_decomposition

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="Stock Time Series Forecasting", layout="wide")

st.title("📈 Stock Time Series Forecasting Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("Upload your stock dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    # --------------------------
    # Data exploration
    # --------------------------
    st.subheader("🔍 Data Overview")

    # Date picker for full dataset
    min_date, max_date = df.index.min(), df.index.max()
    start_date = st.date_input("Choose start date to view dataset",
                               value=min_date,
                               min_value=min_date,
                               max_value=max_date)

    df_filtered = df.loc[start_date:]
    st.dataframe(df_filtered)   # Show full dataset from selected date

    st.write("### 📊 Quick EDA")
    st.plotly_chart(plot_resampling(df), use_container_width=True)
    st.plotly_chart(plot_rolling(df), use_container_width=True)

    feature_for_decomp = st.selectbox("Choose feature for decomposition", df.columns)
    st.plotly_chart(plot_decomposition(df, feature_for_decomp), use_container_width=True)

    # --------------------------
    # Model Training & Forecasting
    # --------------------------
    st.subheader("🤖 Model Training & Forecasting")

    features = st.multiselect("Select features to forecast", df.columns.tolist(), default=["Close"])

    if st.button("Run Forecasting"):
        with st.spinner("Training models... this may take a while ⏳"):
            results_df, predictions = train_all_models(df, features)

        st.success("✅ Forecasting completed!")

        # Show results
        st.subheader("📑 Model Evaluation Results")
        st.dataframe(results_df)

        # --------------------------
        # Forecast Plots + Download
        # --------------------------
        for feature in features:
            st.write(f"### 🔮 Predictions for {feature}")
            for model, (x, y_true, y_pred) in predictions[feature].items():
                fig = px.line(x=x, y=[y_true.flatten(), y_pred.flatten()],
                              labels={"x": "Date", "y": feature},
                              title=f"{model} Predictions for {feature}")
                fig.update_traces(mode="lines")
                fig.data[0].name = "Actual"
                fig.data[1].name = "Predicted"
                st.plotly_chart(fig, use_container_width=True)

                # Add download button for forecasted data
                pred_df = pd.DataFrame({
                    "Date": x,
                    "Actual": y_true.flatten(),
                    "Predicted": y_pred.flatten()
                })
                csv_preds = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"📥 Download {model} Predictions for {feature}",
                    data=csv_preds,
                    file_name=f"{model}_{feature}_predictions.csv",
                    mime="text/csv"
                )
