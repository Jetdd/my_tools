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
        self.shift_num = kwargs.get('shift', 1) # 默认Alpha对应下一期收益
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
        df["alpha"] = position.stack().shift(self.shift_num)
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
            sig_df[~sig_df.isnull()] = 1 # 规则化信号, 将因子值转换为0/1
            sig_df = sig_df.div(self.hold_ret.count(axis=1).shift(-1), axis=0)
            pnl = sig_df * self.hold_ret # 信号已经shift(1), 此处不需要再次shift
            nav = pnl - (sig_df.fillna(0).diff().abs() * self.fee) # 扣除手续费
            dynamic_df[group] = nav.sum(axis=1).cumsum() # 每日分组累计收益
        
        self.dynamic_df = dynamic_df
        
    def plot(self, start_time='20100104'):
        """分组平均收益图, 分组动态图, 多空最大最小两组收益图
        """
        self.run() 
        
        # 从特定时间开始画图
        try:
            plot_static_df = self.static_df.loc[pd.to_datetime(start_time):]
            plot_dynamic_df = self.dynamic_df.loc[pd.to_datetime(start_time):]
        except: # 消除夜盘影响
            plot_static_df = self.static_df
            plot_dynamic_df = self.dynamic_df
        
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
        # 静态图
        sns.barplot(data=plot_static_df, x='groups', y='ret', ax=axes[0], palette='muted')
        axes[0].set_title('Group Returns')
        axes[0].set_xlabel('Groups')
        axes[0].set_ylabel('Returns')
        
        # 全组动态图
        sns.lineplot(data=plot_dynamic_df, ax=axes[1])
        axes[1].set_title('PnL')
        axes[1].legend(title='PnL Series', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 最两边组动态图
        right_group = self.num_groups
        left_group = 1
        temp = plot_dynamic_df[[left_group, right_group]].copy()
        temp['Combo'] = temp[right_group] - temp[left_group]
        temp['Combo'] = temp['Combo'] * np.sign(temp['Combo'].iloc[-1])
        
        sns.lineplot(data=temp, ax=axes[2])
        axes[2].set_title('Two Groups PnL')
        axes[2].legend(title='Two Groups PnL', bbox_to_anchor=(1.05, 1), loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    def run(self) -> (pd.DataFrame, pd.DataFrame):
        self._get_positions()
        self._compute_static()
        self._compute_daynamic()
        
        return self.static_df, self.dynamic_df

        
        