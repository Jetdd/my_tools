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
from dataclasses import dataclass


@dataclass
class Report:
    static_df: pd.DataFrame
    dynamic_df: pd.DataFrame
    ic: pd.DataFrame
    rank_ic: pd.DataFrame
    ir: pd.DataFrame
    sharpe: float
    max_drawdown: float
    stats: pd.DataFrame

class FactorAnalysis:
    position = pd.DataFrame()
    
    def __init__(self, alpha, hold_ret, norm_method, num_groups, **kwargs) -> None:
        start_time = kwargs.get('start_time', '20100104')
        self.method = norm_method
        self.num_groups = num_groups
        self.shift_num = kwargs.get('shift_num', 1) # 默认Alpha对应下一期收益
        self.fee = kwargs.get('fee', 2e-4)  # 默认手续费为万2
        self.window = kwargs.get("window", 240)
        self.ic_rolling = kwargs.get('ic_rolling', 30) # 计算IR的窗口
        
        try:
            self.alpha = alpha.loc[pd.to_datetime(start_time):]
            self.hold_ret = hold_ret.loc[pd.to_datetime(start_time):]
        except:
            self.alpha = alpha
            self.hold_ret = hold_ret
            
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
            dynamic_df[group] = nav.sum(axis=1) # 每日分组收益
        
        self.dynamic_df = dynamic_df
    
    def _compute_ic(self):
        """计算因子IC值"""
        ic_df = self.alpha.shift(self.shift_num).corrwith(self.hold_ret, axis=1, method='pearson')
        rank_ic_df = self.alpha.shift(self.shift_num).corrwith(self.hold_ret, axis=1, method='spearman')
        ir_df = ic_df.rolling(self.ic_rolling).mean() / ic_df.rolling(self.ic_rolling).std()
        
        self.ic = ic_df
        self.rank_ic = rank_ic_df
        self.ir = ir_df
    
    def _compute_stats(self):
        """计算统计指标"""
        right_group = self.num_groups
        left_group = 1
        df = self.dynamic_df[[left_group, right_group]].copy()
        df['Combo'] = df[right_group] - df[left_group]
        self.alpha_sign = np.sign(df['Combo'].cumsum().values[-1])
        
        self.sharpe = my_trading.my_sharpe(df['Combo'] * self.alpha_sign)
        self.max_drawdown = my_trading.my_max_drawdown(df['Combo'])
        
    def plot(self, suptitle='Factor Analysis'):
        """分组平均收益图, 分组动态图, 多空最大最小两组收益图
        """
        self.run() 

        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 15))
        # 1. 静态图
        sns.barplot(data=self.static_df, x='groups', y='ret', ax=axes[0], palette='muted')
        axes[0].set_title('Group Returns')
        axes[0].set_xlabel('Groups')
        axes[0].set_ylabel('Returns')
        
        # 2. 全组动态图
        sns.lineplot(data=self.dynamic_df.cumsum(), ax=axes[1]) # 累计收益图
        axes[1].set_title('PnL')
        axes[1].legend(title='PnL Series')
        
        # 3. 最两边组动态图
        right_group = self.num_groups
        left_group = 1
        temp = self.dynamic_df[[left_group, right_group]].copy()
        temp['Combo'] = temp[right_group] - temp[left_group]
        sns.lineplot(data=temp.cumsum() * np.sign(temp.cumsum().values[-1]), # alpha sign 调整方向
                     ax=axes[2]) # 累计收益图
        axes[2].set_title('Two Groups PnL')
        axes[2].legend(title='Two Groups PnL')
        
        # 4. Rolling IC图
        sns.lineplot(data=self.ic.rolling(self.ic_rolling).mean(), ax=axes[3])
        axes[3].set_title('Rolling IC')
        axes[3].set_xlabel('Dates')
        axes[3].set_ylabel('IC')
        
        # 5. IR图 
        sns.lineplot(data=self.ir, ax=axes[4])
        axes[4].set_title('IR')
        axes[4].set_xlabel('Dates')
        axes[4].set_ylabel('IR')
        
        
        plt.tight_layout()
        plt.suptitle(suptitle, x=0.5, y=1.02, fontsize=20)
        plt.show()
    
    def run(self) -> (pd.DataFrame, pd.DataFrame):
        self._get_positions()
        self._compute_static()
        self._compute_daynamic()
        self._compute_ic()
        self._compute_stats()
        stats_df = pd.DataFrame(data=[np.round(self.sharpe, 3), 
                                      f"{np.round(self.max_drawdown, 4)}%", 
                                      np.round(self.ic.mean(), 4), 
                                      int(self.alpha_sign)], 
                                index=['Combo Sharpe', 'Combo Max Drawdown', 'IC Mean', 'Alpha Sign'], 
                                columns=['Stats'])
        
        self.report = Report(static_df=self.static_df,
                        dynamic_df=self.dynamic_df,
                        ic=self.ic,
                        rank_ic=self.rank_ic,
                        ir=self.ir,
                        sharpe=self.sharpe,
                        max_drawdown=self.max_drawdown,
                        stats=stats_df)
        
        

        return self.report

        
        