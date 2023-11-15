'''
Author: Jet Deng
Date: 2023-10-24 15:24:10
LastEditTime: 2023-11-15 15:19:49
Description: Time Series and Cross Sectional Operators
'''

import numpy as np
import pandas as pd
import bottleneck as bn
from numpy_ext import rolling_apply
import statsmodels.api as sm
from joblib import Parallel, delayed
from types import FunctionType as function

""" TIME SERIES """
def rolling_std(x, window):
    if isinstance(x, pd.Series):
        return pd.Series(bn.move_std(x.T, window, min_count=max(int(window / 2), 2), ddof=1).T, index=x.index,
                         name=x.name)
    else:
        return pd.DataFrame(bn.move_std(x.T, window, min_count=max(int(window / 2), 2), ddof=1).T, index=x.index,
                            columns=x.columns)


def rolling_mean(x, window):
    if window <= 1:
        return x

    if isinstance(x, pd.Series):
        return pd.Series(bn.move_mean(x.T, window, min_count=1).T, index=x.index, name=x.name)
    else:
        return pd.DataFrame(bn.move_mean(x.T, window, min_count=1).T, index=x.index, columns=x.columns)


def rolling_zscore(x, window):
    mean_x = rolling_mean(x, window)
    std_x = rolling_std(x, window)
    norm_x = (x - mean_x) / std_x
    norm_x[norm_x > 3] = 3
    norm_x[norm_x < -3] = -3
    return norm_x


def rolling_rank(x, window):
    if isinstance(x, pd.Series):
        return pd.Series(bn.move_rank(x.T, window, min_count=1).T, index=x.index, name=x.name)
    else:
        return pd.DataFrame(bn.move_rank(x.T, window, min_count=1).T, index=x.index, columns=x.columns)


def ts_rank(x, d):
    if isinstance(x, pd.Series):
        return pd.Series(bn.move_rank(x.T, d, min_count=1).T, index=x.index, name=x.name)
    else:
        return pd.DataFrame(bn.move_rank(x.T, d, min_count=1).T, index=x.index, columns=x.columns)
    
def ts_delay(x, d) -> pd.Series:
    '''
    get x from d days ago
    :param x: (pd.Series)
    :param d: (int)
    '''
    if isinstance(x, pd.Series):
        return x.shift(d)
    else:
        raise ValueError('x has to be series')

def ts_delta(x, d) -> pd.Series:
    '''
    get the difference between x and ts_delay(x, d)
    :param x: (pd.Series)
    :param d: (int)
    '''
    x_delay = ts_delay(x, d)
    res = x - x_delay
    return res

def ts_mean(x, d) -> pd.Series:
    '''
    mean(x, d)
    '''
    x_mean = x.rolling(d).mean()
    return x_mean

def ts_moment(x, d, k) -> pd.Series:
    '''
    get the the k th moment of x from the last d days
    :param x: (pd.Series)
    :param d: (int) days
    :param k: (int) k the moment
    '''
    mean_ = ts_mean(x, d)
    x_ = (x - mean_) ** k
    return ts_mean(x_, d)

def ts_sum(x, d) -> pd.Series:
    '''
    get rolling sum
    '''
    res = x.rolling(d).sum()
    return res

def ts_std_dev(x, d) -> pd.Series:
    '''
    get the standard deviation of x from rolling d days
    '''
    x_std = x.rolling(d).std()
    return x_std

def ts_skewness(x, d) -> pd.Series:
    '''
    get the skewness of x from the last d days
    :param x: (pd.Series)
    :param d: (int) days
    '''
    up = ts_moment(x, d, 3)
    down = ts_moment(x, d, 2) ** (3/2)
    res = up / down
    return res

def ts_kurtosis(x, d) -> pd.Series:
    '''
    get the kurtosis from the last d days
    :param x: (pd.Series)
    :param d: (int)
    '''
    up = ts_moment(x, d, 4)
    down = ts_moment(x, d, 2) ** 2
    res = up / down - 3
    return res

def ts_arg_max(x, d) -> pd.Series:
    '''
    get the relative index of the max value in the time series
    '''
    def get_argmax(mx):
        return np.argmax(mx)    
    df = pd.DataFrame()
    df['x'] = x
    df['arg_max'] = rolling_apply(get_argmax, d, df['x'].values)
    return df['arg_max']

def ts_av_diff(x, d) -> pd.Series:
    '''
    x - ts_mean(x, d)
    '''
    return x - ts_mean(x, d)

def ts_returns(x, d, mode=1) -> pd.Series:
    '''
    mode 1: (x - ts_delay(x, d)) / ts_delay(x, d)
    mode 2: (x - ts_delay(x, d)) / ((x + ts_delay(x, d))/2)
    '''
    if mode == 1:
        res = (x - ts_delay(x, d)) / ts_delay(x, d)
    elif mode == 2:
        res = (x - ts_delay(x, d)) / ((x + ts_delay(x, d))/2)
    return res

def ts_product(x, d):
    '''
    get the product of x from the last d days
    '''
    res = x.rolling(d).agg(lambda x: x.prod())
    return res

def ts_regression(y, x, window, vanilla=False, rettype=2) -> pd.Series:
    '''
    rettype 0 : error term
    rettype 1 : intercept
    rettype 2: slope
    rettype 3: SSE
    rettype 4: SST
    rettype 5: R^2
    '''
    df = pd.DataFrame({'x': x, 'y':y}).dropna()
    res = np.zeros(len(df) - window)
    for i in range(window, len(df)):
        if vanilla == True:
            x = np.arange(window)
        x_ = df['x'].iloc[i-window:i-1].values
        y_ = df['y'].iloc[i-window:i-1].values
        x_ = sm.add_constant(x_)
        x_last = df['x'].iloc[i]
        y_last = df['y'].iloc[i]
        # Fit OLS regression model
        results = sm.OLS(y_, x_).fit()

        # print(results.params)
        if len(results.params) < 2:
            intercept = np.nan
            slope = np.nan
            coef_err = np.nan
        else:
            intercept, slope = results.params
            int_err, coef_err = results.bse
        rsquared = results.rsquared
        sse = results.ssr
        sst = results.centered_tss
        pred = intercept + x_last * slope
        if rettype == 0:
            res[i-window] = coef_err 
        elif rettype == 1:
            res[i-window] = intercept
        elif rettype == 2:
            res[i-window] = slope
        elif rettype == 3:
            res[i-window] = sse
        elif rettype == 4:
            res[i-window] = sst
        elif rettype == 5:
            res[i-window] = rsquared
        elif rettype == 6:
            res[i-window] = pred
    return pd.Series(data=res, index=df.index[window:])
            
def hump(x, threshold):
    '''
    if difference.abs() < threshold: 
        return ts_delay(hump(x, 1)) 
    else: 
        return ts_delay(hump(x, 1)) + threshold * sign(x-ts_delay(hump(x, 1)))
    :param threshold: (float)
    '''
    diff = ts_delta(x, 1)
    res = np.zeros(len(diff))
    # for i in range(len(res)):
    pass 


""" CROSS SECTIONAL OPERATORS """
def cross_cut(x: pd.DataFrame, cut: float) -> pd.DataFrame:
    res = cross_rank(x.copy())
    res[res >= cut] = 1
    res[res <= -cut] = -1
    res[res.abs() < cut] = 0
    return res

def cross_rank(x):
    if isinstance(x, pd.Series):
        x = x.to_frame()
    return x.rank(axis=1, pct=True)*2-1

def cross_zscore(x):
    mean_x = x.mean(axis=1)
    std_x = x.std(axis=1)
    norm_x = (x - mean_x.values[:, None]) / (std_x.values[:, None])
    norm_x[norm_x > 3] = 3
    norm_x[norm_x < -3] = -3
    return norm_x

""" PARALLEL COMPUTE """
def wq_paralell(df: pd.DataFrame, operator: function, **kwargs) -> pd.DataFrame:
    '''
    :param colums: (pd.Series) 品种
    :param operator: (function) 因子计算方式
    '''
    res_list = Parallel(n_jobs=-1)(delayed(operator)(df[col], **kwargs) for col in df.columns)
    res = pd.DataFrame()
    for i, col in enumerate(df.columns):
        res[col] = res_list[i]
    return res

