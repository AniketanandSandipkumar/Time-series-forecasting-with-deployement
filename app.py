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
                future_periods = st.sidebar.slider(
                    f"Select future forecast periods for {model_name} - {feature}", 
                    5, 60, 30
                )
                
                if model_name in ["ARIMA", "SARIMA"]:
                    from statsmodels.tsa.arima.model import ARIMA
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                
                    full_series = df[[feature]].dropna()
                    if model_name == "ARIMA":
                        fitted = ARIMA(full_series, order=(1,1,1)).fit()
                    else:
                        fitted = SARIMAX(full_series, order=(0,0,18), seasonal_order=(0,1,0,18)).fit(disp=False)
                
                    forecast_future = fitted.forecast(steps=future_periods)
                
                    st.line_chart(forecast_future, height=300)
                    st.download_button(
                        f"📥 Download Future {model_name} Forecast for {feature}",
                        forecast_future.to_csv().encode("utf-8"),
                        file_name=f"{feature}_{model_name}_future.csv",
                        mime="text/csv"
                    )
                
                elif model_name == "Prophet":
                    from prophet import Prophet
                    full_series = pd.DataFrame({"ds": df.index, "y": df[feature]})
                    model = Prophet()
                    model.fit(full_series)
                    future = model.make_future_dataframe(periods=future_periods)
                    forecast = model.predict(future)
                
                    fig = px.line(forecast, x="ds", y="yhat", title=f"Future Forecast ({model_name} - {feature})")
                    st.plotly_chart(fig, use_container_width=True)
                
                    st.download_button(
                        f"📥 Download Future {model_name} Forecast for {feature}",
                        forecast.to_csv(index=False).encode("utf-8"),
                        file_name=f"{feature}_{model_name}_future.csv",
                        mime="text/csv"
                    )
                
                elif model_name == "LSTM":
                    from sklearn.preprocessing import MinMaxScaler
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense
                    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
                
                    # Scale full data
                    full_series = df[[feature]].dropna()
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_full = scaler.fit_transform(full_series)
                
                    n_input = 18
                    n_features = 1
                    generator = TimeseriesGenerator(scaled_full, scaled_full, length=n_input, batch_size=1)
                
                    # Build model
                    model = Sequential()
                    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
                    model.add(LSTM(50, activation='relu'))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(generator, epochs=5, verbose=0)
                
                    # Start with last observed batch
                    current_batch = scaled_full[-n_input:].reshape((1, n_input, n_features))
                    future_predictions = []
                
                    for _ in range(future_periods):
                        current_pred = model.predict(current_batch, verbose=0)[0]
                        future_predictions.append(current_pred)
                        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
                
                    forecast_future = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                
                    # Create forecast dates
                    last_date = full_series.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=future_periods+1, freq="D")[1:]
                
                    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted": forecast_future.flatten()})
                
                    st.line_chart(forecast_df.set_index("Date"))
                    st.download_button(
                        f"📥 Download Future {model_name} Forecast for {feature}",
                        forecast_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{feature}_{model_name}_future.csv",
                        mime="text/csv"
                    )


            # --------------------------
            # Download Forecasted Data
            # --------------------------
  
            forecast_df = pd.DataFrame({
                "Date": x,
                "Actual": actual.flatten() if hasattr(actual, "flatten") else actual,
                "Predicted": pred.flatten() if hasattr(pred, "flatten") else pred
            })
            csv_preds = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Download {model_name} Predictions for {feature}",
                csv_preds,
                f"{model_name}_{feature}_predictions.csv",
                "text/csv"
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

    if st.sidebar.checkbox("Show Resampling (Monthly Mean)"):
        st.plotly_chart(plot_resampling(df), use_container_width=True)
    
    if st.sidebar.checkbox("Show Rolling Mean (30 days)"):
        st.plotly_chart(plot_rolling(df), use_container_width=True)
    
    if st.sidebar.checkbox("Show Decomposition (Trend/Seasonality/Residuals)"):
        feature_for_decomp = st.sidebar.selectbox("Select Feature for Decomposition", df.columns.tolist())
        st.plotly_chart(plot_decomposition(df, feature_for_decomp), use_container_width=True)
    
    if st.sidebar.checkbox("Show Candlestick Chart (OHLC)"):
        st.plotly_chart(plot_candlestick(df), use_container_width=True)
    
    if st.sidebar.checkbox("Show Moving Averages (20, 50, 200 days)"):
        st.plotly_chart(plot_moving_averages(df), use_container_width=True)
    
    if st.sidebar.checkbox("Show Daily Returns Distribution"):
        st.plotly_chart(plot_daily_returns(df), use_container_width=True)
    
    if st.sidebar.checkbox("Show Volume vs Price Scatter"):
        st.plotly_chart(plot_volume_vs_price(df), use_container_width=True)
    
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)




