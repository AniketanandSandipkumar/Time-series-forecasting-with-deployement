
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

#trend
def plot_stock_trends(df, feature="Close"):
    st.subheader(f"📈 {feature} Price Trend")
    fig = px.line(df, x=df.index, y=feature, title=f"{feature} over time", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

#Rolling
def plot_rolling(df, feature="Close", window=30):
    st.subheader(f"🔄 Rolling Mean & Std for {feature}")
    roll_mean = df[feature].rolling(window=window).mean()
    roll_std = df[feature].rolling(window=window).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=roll_mean, mode="lines", name=f"{window}-Day Rolling Mean"))
    fig.add_trace(go.Scatter(x=df.index, y=roll_std, mode="lines", name=f"{window}-Day Rolling Std"))
    fig.update_layout(title=f"Rolling Stats ({window} days) - {feature}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# Seasonal Decomposition

def plot_decomposition(df, feature="Close", period=365):
    st.subheader(f"🔍 Seasonal Decomposition for {feature}")
    result = seasonal_decompose(df[feature].dropna(), model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    result.observed.plot(ax=axes[0], title="Observed")
    result.trend.plot(ax=axes[1], title="Trend")
    result.seasonal.plot(ax=axes[2], title="Seasonality")
    result.resid.plot(ax=axes[3], title="Residuals")
    st.pyplot(fig)

# Actual vs Predicted Comparison

def plot_actual_vs_pred(test, preds, model_name="Model", feature="Close"):
    st.subheader(f"🔮 Actual vs Predicted ({model_name} - {feature})")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test.values.flatten(), mode="lines", name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=test.index, y=preds, mode="lines", name=f"{model_name} Prediction", line=dict(dash="dot")))
    fig.update_layout(title=f"{model_name} Predictions vs Actual ({feature})", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# Resampling (Monthly, Quarterly, Yearly)

def plot_resampling(df, feature="Close"):
    st.subheader(f"📊 Resampling for {feature}")
    resampled_monthly = df[feature].resample("M").mean()
    resampled_quarterly = df[feature].resample("Q").mean()
    resampled_yearly = df[feature].resample("Y").mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resampled_monthly.index, y=resampled_monthly, mode="lines", name="Monthly"))
    fig.add_trace(go.Scatter(x=resampled_quarterly.index, y=resampled_quarterly, mode="lines", name="Quarterly"))
    fig.add_trace(go.Scatter(x=resampled_yearly.index, y=resampled_yearly, mode="lines", name="Yearly"))
    fig.update_layout(title=f"Resampled Data ({feature})", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
