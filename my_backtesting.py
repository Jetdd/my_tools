'''
Author: Jet Deng
Date: 2023-10-24 15:24:10
LastEditTime: 2023-11-15 15:22:23
Description: Backtesting Template
'''
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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
    max_drawdown: pd.DataFrame
    # profit_per_trade: float
    # turn_over: pd.DataFrame
    # sheet: pd.DataFrame

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

    def _max_drawdown(self, product_pnl) -> pd.Series:
        '''
        根据品种计算最大回撤
        :param product_pnl: (pd.DataFrame) 
        :return:
        '''
        res = pd.DataFrame()
        def compute(x: pd.Series):
            peak_value = 0
            max_drawdown = 0
            cum_pnl = x.cumsum()
            for row in cum_pnl:
                if row > peak_value:
                    peak_value = row
                drawdown = peak_value - row
                max_drawdown = max(drawdown, max_drawdown)
            return np.round(max_drawdown * 100, 2)
        if isinstance(product_pnl, pd.Series):
            max_drawdown = compute(product_pnl)
            res['total'] = [max_drawdown]
        else:
            for col in product_pnl.columns:
                max_drawdown = compute(product_pnl[col])
                res[col] = [max_drawdown]
            res['total'] = compute(product_pnl.sum(axis=1))
        return res
    
    # def _turn_over(self):
    #     pos = self.position.shift(1).fillna(0)
    #     change = pos.diff(1)
    #     res = pd.DataFrame()
    #     def compute(x: pd.Series):
    #         turn_over_index = x.reset_index().index[x!=0].tolist()
    #         distances = [turn_over_index[i] - turn_over_index[i-1] for i in range(1, len(turn_over_index))]
    #         if distances:
    #             turn_over = sum(distances) / len(distances)
    #             return turn_over
    #         else:
    #             return 0
    #     res_list = Parallel(n_jobs=-1)(delayed(compute)(change[col]) for col in change.columns)
    #     for i, col in enumerate(change.columns):
    #         res[col] = [res_list[i]]
    #     return res
    
    def _stats(self):
        pos = self.position.shift(1).fillna(0)
        product_pnl = pos * self.hold_ret
        pnl = product_pnl.sum(axis=1)

        alpha_sign = np.sign(pnl.sum())
        product_pnl = product_pnl * alpha_sign
        pnl = pnl * alpha_sign
        pos = pos * alpha_sign

        if self.intraday:
            product_nav = product_pnl - (pos.abs() * self.fee).fillna(0)
        else:
            product_nav = product_pnl - (pos.diff(1).abs() * self.fee).fillna(0)
        nav = product_nav.sum(axis=1)

        pnl.index = pd.to_datetime(pnl.index)
        nav.index = pd.to_datetime(nav.index)

        sharpe_pnl = np.round(pnl[pnl.abs() > 0].mean() / pnl[pnl.abs() > 0].std() * 16, 3)
        sharpe_nav = np.round(nav[nav.abs() > 0].mean() / nav[nav.abs() > 0].std() * 16, 3)
        max_drawdown = self._max_drawdown(product_pnl=product_nav).T.rename({0:'max_drawdown'}, axis=1)
        # profit_per_trade = pd.DataFrame(product_pnl.sum(axis=0) / np.count_nonzero(product_pnl.diff(1).fillna(0), axis=0))
        # turn_over = self._turn_over().T.rename({0:'turn_over'}, axis=1)
        # idx = ['cum_pnl', 'cum_nav', 'sharpe_pnl', 'sharpe_nav', 'alpha_sign', 'max_drawdown', 'turn_over']
        # sheet = pd.DataFrame(
        #     index=idx, 
        #     data=np.array([np.round(pnl.sum(), 4), 
        #                    np.round(nav.sum(), 4), 
        #                    sharpe_pnl, 
        #                    sharpe_nav, 
        #                    alpha_sign, 
        #                    max_drawdown.loc['total'].iloc[-1], 
        #                    turn_over['turn_over'].mean()]),
        #     columns=['stats']
        #     )
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
                             max_drawdown,
                            #  profit_per_trade,
                            #  turn_over,
                            #  sheet=sheet,
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



def sim(alpha, hold_ret, **kwargs):
    """
    单因子回测
    """
    cl_sim = Simulate(alpha, hold_ret, **kwargs)
    cl_sim.run()
    return cl_sim.report
