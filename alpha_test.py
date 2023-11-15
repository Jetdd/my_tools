'''
Author: Credit to ZJT
Date: 2023-10-24 15:24:10
LastEditTime: 2023-11-15 15:21:35
Description: Original Alpha Test Template
'''

from dataclasses import dataclass

import pandas as pd
from joblib import Parallel, delayed
from .my_operators import *


@dataclass
class Report:
    alpha: pd.DataFrame
    signal: pd.DataFrame
    position: pd.DataFrame
    product_pnl: pd.DataFrame
    product_nav: pd.DataFrame
    pnl: pd.Series
    nav: pd.Series
    sharpe_pnl: float
    sharpe_nav: float
    alpha_sign: int

class Simulate:
    signal = pd.DataFrame()
    position = pd.DataFrame()
    report = None

    def __init__(self, alpha, hold_ret, **kwargs):
        self.alpha = alpha
        self.hold_ret = hold_ret
        self.norm = kwargs.get("norm", "cross_rank")
        self.cash = kwargs.get("cash", "equal")
        self.fee = kwargs.get("fee", 3e-4)
        self.intraday = kwargs.get('intraday', False)
        
    def run(self):
        self._normalize()
        self._get_position()
        self._stats()

    def _stats(self):
        pos = self.position.shift(1)
        product_pnl = pos * self.hold_ret
        pnl = product_pnl.sum(axis=1)

        alpha_sign = np.sign(pnl.sum())
        product_pnl = product_pnl * alpha_sign
        pnl = pnl * alpha_sign
        pos = pos * alpha_sign

        if self.intraday:
            product_nav = product_pnl - (pos.abs() * self.fee)
        else:
            product_nav = product_pnl - (pos.diff(1).abs() * self.fee)
        nav = product_nav.sum(axis=1)

        pnl.index = pd.to_datetime(pnl.index)
        nav.index = pd.to_datetime(nav.index)

        sharpe_pnl = np.round(pnl[pnl.abs() > 0].mean() / pnl[pnl.abs() > 0].std() * 16, 3)
        sharpe_nav = np.round(nav[nav.abs() > 0].mean() / nav[nav.abs() > 0].std() * 16, 3)

        self.report = Report(self.alpha,
                             self.signal,
                             self.position,
                             product_pnl,
                             product_nav,
                             pnl,
                             nav,
                             sharpe_pnl,
                             sharpe_nav,
                             alpha_sign,
                             )

    def _normalize(self):
        window = 240
        x = self.alpha.copy()

        # 标准化
        if self.norm == 'cross_rank':
            # 截面rank
            res = cross_rank(x)
        elif self.norm == 'ts_rank':
            # 时序rank
            res = rolling_rank(x, window)
        elif self.norm == 'cross_zscore':
            # 截面归一化
            res = cross_zscore(x)
        elif self.norm == 'ts_zscore':
            # 时序归一化
            res = rolling_zscore(x, window)
        else:
            res = x

        self.signal = res.copy()

    def _get_position(self):
        if self.cash == "netural":
            pos = self.signal.apply(lambda x: x - self.signal.mean(axis=1))
            self.position = pos.apply(lambda x: x / pos.abs().sum(axis=1))
        elif self.cash == "equal":
            self.position = self.signal.apply(lambda x: x / self.signal.count(axis=1))
        else:
            self.position = self.signal


def get_hold_ret(hot_adj):
    # open_hot = hot_adj.pivot('datetime', 'product', 'open')
    # close_hot = hot_adj.pivot('datetime', 'product', 'close')
    try:
        open_hot = pd.pivot_table(hot_adj, index='datetime', columns='types', values='open')
        close_hot = pd.pivot_table(hot_adj, index='datetime', columns='types', values='close')
    except:
        open_hot = pd.pivot_table(hot_adj, index='date', columns='types', values='open')
        close_hot = pd.pivot_table(hot_adj, index='date', columns='types', values='close')

    ret = open_hot.shift(-1) / open_hot - 1
    ret.iloc[-1, :] = close_hot.iloc[-1, :] / open_hot.iloc[-1, :] - 1

    return ret


def sim(alpha, hold_ret, **kwargs):
    """
    单因子回测
    """
    cl_sim = Simulate(alpha, hold_ret, **kwargs)
    cl_sim.run()
    return cl_sim.report


def batch_sim(alpha_list, hold_ret, n_jobs, **kwargs):
    """
    多进程跑回测
    """

    with Parallel(n_jobs=n_jobs) as pa:
        reports = pa(delayed(sim)(alpha0, hold_ret, **kwargs) for alpha0 in alpha_list)

    return reports


if __name__ == "__main__":
    norm = "ts_rank"  # ts_zscore, cross_rank, cross_zscore 因子标准化方式
    cash = "equal" # 品种资金分配方式

    hot_adj = pd.read_pickle("hot_adj.pkl")  # 复权数据
    alpha = pd.read_pickle("alpha.pkl")
    hold_ret = get_hold_ret(hot_adj)

    report: Report = sim(alpha, hold_ret, norm=norm, cash=cash)

    print(f"sharpe_nav = {report.sharpe_nav}, sharpe_pnl = {report.sharpe_pnl}, sign = {report.alpha_sign}")
