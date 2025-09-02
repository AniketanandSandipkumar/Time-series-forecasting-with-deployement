import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# --------------------------
# Evaluation helper
# --------------------------
def evaluate_forecast(true, pred):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return rmse, mae, mape


# --------------------------
# Individual models
# --------------------------
def run_arima(train, test, order=(1,1,1)):
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    pred = model_fit.forecast(len(test))
    return pred


def run_sarima(train, test, order=(0,0,18), seasonal_order=(0,1,0,18)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    pred = model_fit.forecast(len(test))
    return pred


def run_prophet(train_df, test_df):
    model = Prophet()
    model.fit(train_df)
    future = model.make_future_dataframe(periods=len(test_df))
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-len(test_df):].values


def run_lstm(train, test, n_input=18, epochs=5):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)

    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_input, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=epochs, verbose=0)

    test_predictions = []
    current_batch = scaled_train[-n_input:].reshape((1, n_input, n_features))

    for _ in range(len(test)):
        current_pred = model.predict(current_batch, verbose=0)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

    return scaler.inverse_transform(np.array(test_predictions).reshape(-1,1))


# --------------------------
# Main training function
# --------------------------
def train_models(df, features, models_to_run=None, test_ratio=0.2):
    """
    Train selected models for each feature.
    
    Args:
        df (pd.DataFrame): stock data
        features (list): list of features e.g. ["Open","Close"]
        models_to_run (list): subset of ["ARIMA","SARIMA","Prophet","LSTM"]
        test_ratio (float): test size

    Returns:
        results_df (pd.DataFrame)
        predictions (dict)
    """

    if models_to_run is None:
        models_to_run = ["ARIMA", "SARIMA", "Prophet", "LSTM"]

    results = []
    predictions = {}

    df = df.loc["2010-01-01":]

    for feature in features:
        print(f"\n=== Running models for {feature} ===")
        series = df[[feature]].dropna()
        split = int(len(series) * (1 - test_ratio))
        train, test = series[:split], series[split:]
        predictions[feature] = {}

        # ---------------- ARIMA ----------------
        if "ARIMA" in models_to_run:
            try:
                pred = run_arima(train, test)
                rmse, mae, mape = evaluate_forecast(test.values, pred.values)
                results.append(["ARIMA", feature, rmse, mae, mape])
                predictions[feature]["ARIMA"] = (test.index, test.values, pred)
                print(f"✅ ARIMA finished for {feature} (RMSE={rmse:.2f})")
            except Exception as e:
                print(f"❌ ARIMA failed for {feature}: {e}")

        # ---------------- SARIMA ----------------
        if "SARIMA" in models_to_run:
            try:
                pred = run_sarima(train, test)
                rmse, mae, mape = evaluate_forecast(test.values, pred.values)
                results.append(["SARIMA", feature, rmse, mae, mape])
                predictions[feature]["SARIMA"] = (test.index, test.values, pred)
                print(f"✅ SARIMA finished for {feature} (RMSE={rmse:.2f})")
            except Exception as e:
                print(f"❌ SARIMA failed for {feature}: {e}")

        # ---------------- Prophet ----------------
        if "Prophet" in models_to_run:
            try:
                df_fb = pd.DataFrame({"ds": series.index, "y": series[feature]})
                df_train, df_test = df_fb.iloc[:split], df_fb.iloc[split:]
                preds = run_prophet(df_train, df_test)
                rmse, mae, mape = evaluate_forecast(df_test["y"].values, preds)
                results.append(["Prophet", feature, rmse, mae, mape])
                predictions[feature]["Prophet"] = (df_test["ds"], df_test["y"], preds)
                print(f"✅ Prophet finished for {feature} (RMSE={rmse:.2f})")
            except Exception as e:
                print(f"❌ Prophet failed for {feature}: {e}")

        # ---------------- LSTM ----------------
        if "LSTM" in models_to_run:
            print("⚠️ LSTM may take longer to train, please wait...")
            try:
                preds = run_lstm(train, test)
                rmse, mae, mape = evaluate_forecast(test.values, preds)
                results.append(["LSTM", feature, rmse, mae, mape])
                predictions[feature]["LSTM"] = (test.index, test.values, preds)
                print(f"✅ LSTM finished for {feature} (RMSE={rmse:.2f})")
            except Exception as e:
                print(f"❌ LSTM failed for {feature}: {e}")

    results_df = pd.DataFrame(results, columns=["Model", "Feature", "RMSE", "MAE", "MAPE"])
    return results_df, predictions
