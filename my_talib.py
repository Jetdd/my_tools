'''
Author: Jet Deng
Date: 2023-10-24 15:24:10
LastEditTime: 2023-11-15 15:30:58
Description: TALIB Functions Generator
'''
import pandas as pd
import numpy as np
import talib 
from talib import abstract
from . import my_operators
from . import my_tools
from joblib import Parallel, delayed

def my_talib_single(data_dict: dict, func_name: str, **kwargs) -> pd.DataFrame:
    """
    输入数据字典和需要计算的TALIB function, 输出相应的alpha
    """
    alpha = pd.DataFrame()
    func = getattr(abstract, func_name)
    for tp in data_dict:
        dd = data_dict[tp]
        alpha[tp] = func(dd, **kwargs)
    return alpha

def my_talib_vol(data_dict: dict, **kwargs) -> dict:
    '''
    计算TALIB波动率类因子
    :param data_dict: (dict) 按照品种储存的字典
    :return: (dict) 按照因子名字储存的字典
    '''
    df = my_tools.my_concat_data_dict(data_dict=data_dict)
    indicators = ['ATR', 'NATR', 'TRANGE']
    df_grouped = df.groupby('types')
    def single_ind(ind: str) -> pd.DataFrame:
            alpha = pd.DataFrame()
            func = getattr(abstract, ind)
            for name, group in df_grouped:
                res = func(group, **kwargs)
                try:
                    alpha[name] = res
                except:
                    return
            return alpha
    alpha_dict = {}
    alpha_list = Parallel(n_jobs=30)(delayed(single_ind)(ind) for ind in indicators)
    for i, ind in enumerate(indicators):
        to_add = alpha_list[i]
        if to_add is None:
            continue
        alpha_dict[ind] = alpha_list[i].groupby(alpha_list[i].index).first()
    return alpha_dict

def my_vol(data_dict: dict, n: int) -> dict:
    '''
    计算日级别parkinson, garman_klass, rogers_satchell, yang_zhang 波动率
    :param data_dict: (dict) of dataframe of 品种 OHCL
    :param n: (int) shift_back window
    '''
    close_df = my_tools.my_dataframe(data_dict=data_dict, string='close')
    high_df = my_tools.my_dataframe(data_dict=data_dict, string='high')
    low_df = my_tools.my_dataframe(data_dict=data_dict, string='low')
    open_df = my_tools.my_dataframe(data_dict=data_dict, string='open')
    res = {}
    def parkinson_vol():
        '''
        计算日级别parkinson波动率
        '''
        constant = 1 / (4 * np.log(2) * n)
        to_mul = np.log(high_df / low_df) ** 2
        res = (constant * to_mul).rolling(n).sum()
        return res
    
    def garman_klass_vol():
        '''
        计算日级别garman_klass波动率
        '''
        first_half = np.log(high_df / low_df) ** 2
        second_half = np.log(close_df / close_df.shift(1))
        res = 1/ n * (1/2 * first_half.rolling(n).sum() - (2 * np.log(2) - 1) * second_half.rolling(n).sum())
        res = np.sqrt(res)
        return res
    
    def rogers_satchell_vol():
        log_hl = np.log(high_df / close_df)
        log_ho = np.log(high_df / open_df) 
        log_lc = np.log(low_df / close_df)
        log_lo = np.log(low_df / open_df)
        res = (log_hl * log_ho + log_lc * log_lo).rolling(n).mean()
        res = np.sqrt(res)
        return res
    
    def yang_zhang_vol():
        k = 0.34 / (1.34 + (n+1)/(n-1))
        oo = (np.log(open_df / open_df.shift(1)) - np.log(open_df / open_df.shift(1)).rolling(n).mean()) ** 2
        cc = (np.log(close_df / close_df.shift(1)) - np.log(close_df / close_df.shift(1)).rolling(n).mean()) ** 2
        sigma_open = oo.rolling(n).sum() / (n-1)
        sigma_close = cc.rolling(n).sum() / (n-1)
        sigma_rs = rogers_satchell_vol()
        sigma = np.sqrt(sigma_open ** 2 + k * sigma_close ** 2 + (1-k) * sigma_rs ** 2)
        return sigma
    
    res['par_vol'] = parkinson_vol()
    res['gk_vol'] = garman_klass_vol()
    res['rs_vol'] = rogers_satchell_vol()
    res['yz_vol'] = yang_zhang_vol()
    return res

def my_talib_mom(data_dict: dict, **kwargs) -> dict:
     

     pass
if __name__ == '__main__':
    universe = ["M","RM","P","Y","OI","B","C","CS","A",
            "JD","SR","CF","AP","AL","CU","ZN","NI",
            "PB","SN","SS","RB","HC","I","J","JM",
            "SM","SF","FU","SC","BU","LU","PG","TA",
            "PF","EG","EB","L","MA","PP","SA","UR",
            "V","FG","NR","RU","AU","AG", 'SP']
    data_dict = my_tools.my_load_data_2(need=universe, freq='day', adj=True)
    res = my_talib_vol(data_dict=data_dict, **{'timeperiod': 20})