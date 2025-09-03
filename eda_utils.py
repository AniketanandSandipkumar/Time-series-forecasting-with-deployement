# eda_utils.py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ----------------------
# Time-Series Utilities
# ----------------------

def plot_resampling(df, freq="M"):
    """Resample data and plot monthly mean using Plotly."""
    df_resampled = df.resample(freq).mean()
    fig = px.line(df_resampled, x=df_resampled.index, y=df_resampled.columns,
                  title=f"Resampled Data ({freq} mean)")
    return fig

def plot_rolling(df, window=30):
    """Plot rolling mean (default 30 days)."""
    df_rolling = df.rolling(window).mean()
    fig = px.line(df_rolling, x=df_rolling.index, y=df_rolling.columns,
                  title=f"Rolling Mean ({window} days)")
    return fig

def plot_decomposition(df, feature, model="additive", period=30):
    """Decompose time series into trend/seasonal/residual components."""
    result = seasonal_decompose(df[feature].dropna(), model=model, period=period)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

    fig.add_trace(go.Scatter(x=df.index, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(title=f"Decomposition of {feature}", height=900)
    return fig

def plot_correlation_heatmap(df):
    """Correlation Heatmap of numerical features."""
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    return fig

def plot_volume_vs_price(df, price_col="Close", volume_col="Volume"):
    """Scatter plot between Volume and Price."""
    fig = px.scatter(df, x=volume_col, y=price_col, color=price_col, size=volume_col,
                     title=f"{volume_col} vs {price_col}", template="plotly_dark")
    return fig

def plot_daily_returns(df, feature="Close"):
    """Distribution of daily returns for a feature."""
    returns = df[feature].pct_change().dropna()
    fig = px.histogram(returns, nbins=50, marginal="box",
                       title=f"Daily Returns Distribution for {feature}")
    return fig

def plot_moving_averages(df, feature="Close"):
    """Overlay moving averages on stock prices."""
    df_ma = df.copy()
    df_ma["MA20"] = df[feature].rolling(window=20).mean()
    df_ma["MA50"] = df[feature].rolling(window=50).mean()
    df_ma["MA200"] = df[feature].rolling(window=200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=df_ma["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df_ma["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df_ma["MA200"], mode="lines", name="MA200"))
    fig.update_layout(title=f"Moving Averages ({feature})", template="plotly_white")
    return fig

def plot_acf_plot(df, lags=50):
    """Autocorrelation function plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(df["Close"].dropna(), lags=lags, ax=ax)
    plt.title("Autocorrelation (ACF)")
    return fig

def plot_pacf_plot(df, lags=50):
    """Partial autocorrelation function plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(df["Close"].dropna(), lags=lags, ax=ax, method="ywm")
    plt.title("Partial Autocorrelation (PACF)")
    return fig

def plot_candlestick(df):
    """Candlestick chart (requires OHLC data)."""
    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        )])
        fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False,
                          template="plotly_dark")
        return fig
    else:
        return None



