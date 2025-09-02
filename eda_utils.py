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
