'''
Author: Jet Deng
Date: 2023-11-06 09:58:29
LastEditTime: 2024-03-14 22:15:59
Description: Trading-related Modules
'''
import pandas as pd
import numpy as np
from .my_operators import * 

def my_signal(long_open: pd.DataFrame, 
              long_close: pd.DataFrame, 
              short_open: pd.DataFrame, 
              short_close: pd.DataFrame) -> pd.DataFrame:
    '''
    Input signals output integrated signal dataframe
    :param long_open: (pd.DataFrame) 
    :param long_close: (pd.DataFrame) 
    :param short_open: (pd.DataFrame)
    :param short_open: (pd.DataFrame)
    '''
    if isinstance(long_open, pd.Series):
        long_alpha = pd.Series(np.full(long_open.shape, np.nan), index=long_open.index, name=long_open.name)
        short_alpha = pd.Series(np.full(long_open.shape, np.nan), index=long_open.index, name=long_open.name)
    else:
        long_alpha = pd.DataFrame(np.full(long_open.shape, np.nan), index=long_open.index, columns=long_open.columns)
        short_alpha = pd.DataFrame(np.full(long_open.shape, np.nan), index=long_open.index, columns=long_open.columns)

    long_alpha[long_open] = 1
    long_alpha[long_close] = 0
    short_alpha[short_open] = -1
    short_alpha[short_close] = 0
    long_alpha = long_alpha.fillna(method='ffill').fillna(0)
    short_alpha = short_alpha.fillna(method='ffill').fillna(0)
    alpha = long_alpha + short_alpha
    return alpha

def my_sharpe(pnl: pd.DataFrame | pd.Series):
    if isinstance(pnl, pd.DataFrame):
        pnl = pnl.sum(axis=1)
    res = np.round(pnl.mean() / pnl.std() * 16, 3)
    return res

def my_max_drawdown(pnl) -> pd.Series:
    '''
    根据品种计算最大回撤
    :param pnl: (pd.Series) 
    :return:
    '''
    peak_value = -np.inf
    max_drawdown = -np.inf
    cum_pnl = pnl.cumsum()
    for row in cum_pnl:
        peak_value = max(peak_value, row)
        max_drawdown = max(max_drawdown, (peak_value - row))
    return np.round(max_drawdown * 100, 2)
                    
def my_turnover(pos: pd.DataFrame):
    res = (pos.diff(1).abs().sum(axis=1)/(pos.abs().sum(axis=1).shift(1))).mean()
    return res

""" OTHERS """
def my_vol_improve(alpha: pd.DataFrame, ret_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用波动率优化因子值
    """
    volatility = rolling_std(ret_df, 20) * 16
    volatility[volatility < 0.1] = 0.1
    volatility_coef = 0.1 / volatility
    alpha = alpha * volatility_coef
    return alpha

def my_neutralize(alpha: pd.DataFrame) -> pd.DataFrame:
    """中性化因子, 不暴露多空敞口

    Args:
        alpha (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: alpha
    """
    alpha[alpha > 0] = alpha[alpha > 0] / alpha[alpha > 0].sum(axis=1).values[:, None] * 0.5
    alpha[alpha < 0] = -alpha[alpha < 0] / alpha[alpha < 0].sum(axis=1).values[:, None] * 0.5
    
    return alpha

def my_normalize(alpha: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """标准化因子, 去除因子量纲

    Args:
        alpha (pd.DataFrame): _description_
        method (str): _description_

    Returns:
        pd.DataFrame: new alpha
    """
    window = kwargs.get('window', 240)
    cut = kwargs.get('cut', 0.6)
    
    # 各品种资金等权
    normed_alpha = alpha.apply(lambda x: x / alpha.count(axis=1))
    if method == "rolling_rank":
        normed_alpha = rolling_rank(alpha, window)
    elif method == "cross_rank":
        normed_alpha = cross_rank(alpha)
    elif method == "cross_cut":
        normed_alpha = cross_cut(x=alpha, cut=cut)
        normed_alpha = my_neutralize(alpha=normed_alpha)  # 多空中性化
    elif method == "rolling_zscore":
        normed_alpha = rolling_zscore(
            x=alpha, window=window
        )
    else:
        normed_alpha = normed_alpha
    return normed_alpha

def my_stop_loss(pnl: pd.Series) -> pd.Series:
    """移动止损, 输出止损后的出场信号, 需进行重新回测

    Args:
        pnl (pd.Series): 原策略的pnl

    Returns:
        pd.Series: 止损后的出场信号
    """
    new_signal = pd.Series(np.full(pnl.shape, np.nan), index=pnl.index)
    max_dd_series = pd.Series(np.full(pnl.shape, np.nan), index=pnl.index)
    peak_value = -np.inf
    for i in range(pnl.shape[0]):
        peak_value = max(peak_value, pnl.iloc[i])
        max_dd_series.iloc[i] = peak_value