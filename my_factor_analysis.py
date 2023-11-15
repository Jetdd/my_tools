"""
Author: Jet Deng
Date: 2023-11-13 09:25:23
LastEditTime: 2023-11-13 10:41:16
Description: Factor Analysis Module
"""
from my_tools import my_trading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class FactorAnalysis:
    position = pd.DataFrame()
    static_df = pd.DataFrame()
    def __init__(self, alpha, hold_ret, norm_method, num_groups, **kwargs) -> None:
        self.alpha = alpha
        self.hold_ret = hold_ret
        self.method = norm_method
        self.num_groups = num_groups
        self.fee = kwargs.get('fee', 2e-4)
        self.window = kwargs.get("window", 240)
        
    def _get_positions(self):
        """标准化Alpha, 去掉量纲影响"""
        self.position = my_trading.my_normalize(
            alpha=self.alpha, method=self.method, window=self.window # 时序标准化去掉量纲影响, 可填none
        ) 
        
    def _compute_static(self):
        """计算截面分组静态平均收益"""
        position = my_trading.my_normalize(alpha=self.position, method="cross_rank")
        df = pd.DataFrame()
        df["alpha"] = position.stack().shift(1)
        df["ret"] = self.hold_ret.stack()

        # 根据因子值算截面分组
        labels = [i + 1 for i in range(self.num_groups)]
        df["groups"] = pd.cut(
            df["alpha"],
            bins=np.linspace(-1, 1, self.num_groups + 1),
            labels=labels,
            include_lowest=True,
        )
        
        self.static_df = df

    def _compute_daynamic(self):
        """计算截面分组动态收益"""
        
        dynamic_df = pd.DataFrame()
        for group in range(1, self.num_groups+1):
            sig_df = self.static_df[self.static_df['groups'] == group]['alpha'].unstack()
            pnl = sig_df * self.hold_ret # 信号已经shift(1), 此处不需要再次shift
            nav = pnl - (self.position.shift(1).diff().abs() * self.fee)
            dynamic_df[group] = nav.sum(axis=1).cumsum() # 每日分组累计收益
        
        self.dynamic_df = dynamic_df
        
    def plot(self):
        """分组平均收益图, 分组动态图, 多空最大最小两组收益图
        """
        self.run() 
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
        # 静态图
        sns.barplot(data=self.static_df, x='groups', y='ret', ax=axes[0], palette='muted')
        axes[0].set_title('Group Returns')
        axes[0].set_xlabel('Groups')
        axes[0].set_ylabel('Returns')
        
        # 动态图
        sns.lineplot(data=self.dynamic_df, ax=axes[1])
        axes[1].set_title('PnL')
        axes[1].legend(title='PnL Series', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
    def run(self) -> (pd.DataFrame, pd.DataFrame):
        self._get_positions()
        self._compute_static()
        self._compute_daynamic()
        
        return self.static_df, self.dynamic_df

        
        