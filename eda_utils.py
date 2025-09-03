# eda_utils.py
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

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

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

    fig.add_trace(go.Scatter(x=df.index, y=result.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=result.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=result.resid, name="Residual"), row=4, col=1)

    fig.update_layout(title=f"Decomposition of {feature}", height=900)
    return fig 
# Correlation Heatmap
def plot_correlation_heatmap(df):
    st.subheader("📊 Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# Volume vs Price Scatter
def plot_volume_vs_price(df, price_col="Close", volume_col="Volume"):
    st.subheader("🔍 Volume vs Price Relationship")
    fig = px.scatter(df, x=volume_col, y=price_col, color=price_col, size=volume_col,
                     title=f"{volume_col} vs {price_col}", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Daily Returns Distribution
def plot_daily_returns(df, feature="Close"):
    st.subheader(f"📉 Distribution of Daily Returns - {feature}")
    returns = df[feature].pct_change().dropna()
    fig = px.histogram(returns, nbins=50, marginal="box", title=f"Daily Returns Distribution for {feature}")
    st.plotly_chart(fig, use_container_width=True)

# Moving Average Comparison
def plot_moving_averages(df, feature="Close"):
    st.subheader(f"📈 Moving Averages for {feature}")
    df["MA20"] = df[feature].rolling(window=20).mean()
    df["MA50"] = df[feature].rolling(window=50).mean()
    df["MA200"] = df[feature].rolling(window=200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[feature], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))
    fig.update_layout(title=f"Moving Averages ({feature})", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Candlestick Chart
def plot_candlestick(df):
    st.subheader("📊 Candlestick Chart")
    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        )])
        fig.update_layout(title="Candlestick Chart", xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Candlestick requires Open, High, Low, and Close columns in dataset.")


