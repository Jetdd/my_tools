'''
Author: Jet Deng
Date: 2023-11-01 15:56:21
LastEditTime: 2023-11-27 09:16:41
Description: Some Useful Plotly Functions (PnL)
'''

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

save_path = Path('D:/projects/plot_html/')

def my_plot_rolling_correlation(futures_df: pd.DataFrame, market_df: pd.Series, rolling_n: int) -> None:
    """
    Plot rolling correlation between each future in the futures_df and the market object.

    :param futures_df: (pd.DataFrame) e.g. DataFrame of future returns where each column is a different future.
    :param market_df: (pd.Series) e.g. Series of market returns.
    :param rolling_n: (int) Rolling window size.
    """
    # Calculate rolling correlation
    correlations = futures_df.apply(
        lambda x: x.rolling(window=rolling_n).corr(market_df))

    # Melt the data for plotting
    correlations_melted = correlations.reset_index().melt(id_vars=market_df.index.name or 'index',
                                                          value_vars=correlations.columns)

    # Plot using plotly.express
    fig = px.line(correlations_melted,
                  x=futures_df.index.name or 'index',
                  y='value',
                  color='variable',
                  labels={'value': 'Correlation'},
                  title=f"Rolling {rolling_n}-day correlation with Market Return")
    fig.write_html(save_path / 'plot.html')


def my_plot_price_trend(futures_df: pd.DataFrame) -> None:
    """
    Plot the price ('normalized') trend of each future

    :param futures_df: (pd.DataFrame) e.g. close price
    """
    # Single-digitalized futures price
    if isinstance(futures_df, pd.Series):
        futures_df = futures_df.to_frame()
    for tp in futures_df.columns:
        integer_part = int(futures_df[tp].max())
        norm_divider = len(str(abs(integer_part))) * 10
        futures_df[tp] = futures_df[tp] / norm_divider

    melted_df = futures_df.reset_index().melt(id_vars=futures_df.index.name or 'index',
                                              value_vars=futures_df.columns)
    fig = px.line(melted_df,
                  x=futures_df.index.name or 'index',
                  y='value',
                  color='variable',
                  title='Normalized Trend of Futures Close Prices')
    fig.write_html(save_path / 'plot.html')


def my_line_plot(futures_df: pd.DataFrame, title='Line Plot') -> None:
    """
    Plot the pnl of each future
    :param futures_df: (pd.DataFrame) e.g. pnl
    """
    if isinstance(futures_df, pd.Series):
        futures_df = futures_df.to_frame()

    melted_df = futures_df.reset_index().melt(id_vars=futures_df.index.name or 'index',
                                              value_vars=futures_df.columns)
    fig = px.line(melted_df,
                  x=futures_df.index.name or 'index',
                  y='value',
                  color='variable',
                  title=title)
    fig.write_html(save_path / 'plot.html')


def my_scatter_plot(futures_df: pd.DataFrame, title='Scatter Plot') -> None:
    """
    Plot the pnl of each future
    :param futures_df: (pd.DataFrame) e.g. pnl
    """
    if isinstance(futures_df, pd.Series):
        futures_df = futures_df.to_frame()

    melted_df = futures_df.reset_index().melt(id_vars=futures_df.index.name or 'index',
                                              value_vars=futures_df.columns)
    fig = px.scatter(melted_df,
                     x=futures_df.index.name or 'index',
                     y='value',
                     color='variable',
                     title=title)
    fig.write_html(save_path / 'plot.html')


def my_pnl_plot(pnl: pd.DataFrame | pd.Series, *args):
    if isinstance(pnl, pd.DataFrame):
        pnl = pnl.sum(axis=1)
    cumulative_pnl = pnl.cumsum()
    
    # Create a line chart with Plotly
    fig = go.Figure(data=go.Scatter(x=cumulative_pnl.index, y=cumulative_pnl, mode='lines'))

    # Set title
    title = ' '.join(args) if args else 'pnl_plot'
    fig.update_layout(title=title)
    
    # Show the plot
    fig.write_html(save_path / 'plot.html')
    

def my_train_test_pnl_plot(train_pnl: pd.DataFrame, test_pnl: pd.DataFrame, plot_title: str):
    # Create a subplot figure with 2 rows
    fig = make_subplots(rows=2, cols=1, subplot_titles=(f'{plot_title}_train_pnl', f'{plot_title}_test_pnl'))

    # Sum across columns and calculate the cumulative sum for train pnl
    train_cumulative_pnl = train_pnl.sum(axis=1).cumsum()
    # Sum across columns and calculate the cumulative sum for test pnl
    test_cumulative_pnl = test_pnl.sum(axis=1).cumsum()

    # Add train pnl plot to the first row
    fig.add_trace(
        go.Scatter(x=train_cumulative_pnl.index, y=train_cumulative_pnl, mode='lines', name='Train PnL'),
        row=1, col=1
    )

    # Add test pnl plot to the second row
    fig.add_trace(
        go.Scatter(x=test_cumulative_pnl.index, y=test_cumulative_pnl, mode='lines', name='Test PnL'),
        row=2, col=1
    )

    # Update layout for a cleaner look
    fig.update_layout(height=600, width=800, title_text=plot_title, showlegend=False)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Cumulative PnL")

    # Show the plot
    fig.write_html(save_path / 'plot.html')
    


def my_bbands_signals(data, moving_average_period, bollinger_band_width, long_open, long_close, short_open, short_close):
    # Calculate the moving average
    moving_average = data.rolling(window=moving_average_period).mean()

    # Calculate the standard deviation
    std_dev = data.rolling(window=moving_average_period).std()

    # Calculate the upper and lower Bollinger Bands
    upper_band = moving_average + (bollinger_band_width * std_dev)
    lower_band = moving_average - (bollinger_band_width * std_dev)

    # Create traces for the price, moving average, and Bollinger Bands
    price_trace = go.Scatter(x=data.index, y=data, mode='lines', name='Price', line=dict(color='blue'))
    ma_trace = go.Scatter(x=data.index, y=moving_average, mode='lines', name='Moving Average', line=dict(color='green', dash='dot'))
    upper_band_trace = go.Scatter(x=data.index, y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='red'))
    lower_band_trace = go.Scatter(x=data.index, y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='purple'))

    # Traces for trading signals
    long_open_trace = go.Scatter(x=long_open.index, y=data[long_open == 1], mode='markers', name='Long Open', marker=dict(color='lime', size=10, symbol='triangle-up'))
    long_close_trace = go.Scatter(x=long_close.index, y=data[long_close == 1], mode='markers', name='Long Close', marker=dict(color='green', size=10, symbol='triangle-down'))
    short_open_trace = go.Scatter(x=short_open.index, y=data[short_open == 1], mode='markers', name='Short Open', marker=dict(color='magenta', size=10, symbol='triangle-up'))
    short_close_trace = go.Scatter(x=short_close.index, y=data[short_close == 1], mode='markers', name='Short Close', marker=dict(color='purple', size=10, symbol='triangle-down'))

    # Fill between Bollinger Bands
    fill_trace = go.Scatter(x=data.index.tolist() + data.index[::-1].tolist(),
                            y=upper_band.tolist() + lower_band[::-1].tolist(),
                            fill='toself', fillcolor='rgba(128, 128, 128, 0.3)', line=dict(color='rgba(255,255,255,0)'),
                            name='Bollinger Band Area')

    # Define the layout
    layout = go.Layout(title='Moving Average and Bollinger Bands with Trading Signals', xaxis_title='Date', yaxis_title='Price', legend=dict(x=0.05, y=0.95))

    # Combine all traces and layout in a figure
    fig = go.Figure(data=[price_trace, ma_trace, upper_band_trace, lower_band_trace, fill_trace, long_open_trace, long_close_trace, short_open_trace, short_close_trace], layout=layout)

    fig.show()
