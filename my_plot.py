'''
Author: Jet Deng
Date: 2023-11-01 15:56:21
LastEditTime: 2023-11-15 15:20:30
Description: Some Useful Plotly Functions (PnL)
'''

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    fig.show()


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
    fig.show()


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
    fig.show()


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
    fig.show()


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
    fig.show()
    return 

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
    fig.show()
    return
