'''
Author: Jet Deng
Date: 2023-10-24 15:24:10
LastEditTime: 2024-05-21 16:17:51
Description: Tool functions, including backtesting, plotting
'''
import pandas as pd
import polars as pl
import numpy as np
import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed
from .my_operators import *
#################################

"""
LOAD DATA
"""
def my_load_data(need: list, dominant: str, freq: str, adj: bool, **kwargs) -> dict:
    '''
    Given types of futures, we load all data to a dictionary
    '''
    start_date = kwargs.get("start_date", "20100104")  # cut the data from a given date or time
    end_date = kwargs.get("end_date", "today")  # cut the data from a given date or time
    
    data_dict = {}
    if adj:
        adj = 'adj'
    else:
        adj = 'no_adj'
    if freq == 'day':
        for tp in need:
            data = pd.read_pickle('D:/projects/data/future/1d/{}/{}/{}.pkl'.format(dominant, adj,  tp)).reset_index()
            data = data[(data['date']>=start_date) & (data['date']<=end_date)]
            data_dict[tp] = data.set_index('date')
    elif freq == '30m':
        for tp in need:
            data = pd.read_pickle('D:/projects/data/future/30m/{}/{}/{}.pkl'.format(dominant, adj, tp)).reset_index()
            data = data[(data['trading_date']>=start_date) & (data['trading_date']<=end_date)]
            data_dict[tp] = data.set_index('datetime')
    elif freq == '1m':
        for tp in need:
            data = pd.read_pickle('D:/projects/data/future/1m/{}/{}/{}.pkl'.format(dominant, adj, tp)).reset_index()
            data = data[(data['trading_date']>=start_date) & (data['trading_date']<=end_date)]
            data_dict[tp] = data.set_index('datetime')
    return data_dict

def my_future_info():
    table = pd.read_csv("D:/projects/data/future/info.csv")
    return table
def get_agr_list() -> list:
    res = ['A', 'B', 'P', 'Y', 'OI', 'M', 
        'RM', 'C', 'CS', 'AP', 'CF', 
        'CJ','PK', 'LH', 'SR']  
    return res

def get_chem_list() -> list:
    chem = ['PP', 'L', 'MA', 'V', 'EB', 
        'EG', 'TA', 'SP', 'SA', 'UR',
        'PF', 'RU', 'NR', 'BU']
    return chem

def get_metal_list() -> list:
    metal = ['CU', 'AL', 'PB', 
         'ZN', 'NI', 'SN',
         'AU', 'AG', 'SS']
    return metal

def get_energy_list() -> list:
    energy = ['SC', 'LU', 'FU', 'PG']
    return energy

def get_ferrous_list() -> list:
    steel = ['RB', 'HC', 'I', 'J', 
         'JM', 'SF', 'SM', ]
    return steel

def get_oil_universe() -> list:
    return ["CF", "CY", "SR"]

def get_grain_universe() -> list:
    return ["A", "B", "C"]

def get_universe() -> list:
    universe = ["M","RM","P","Y","OI","B","C","CS","A",
            "JD","SR","CF","AP","AL","CU","ZN","NI",
            "PB","SN","SS","RB","HC","I","J","JM",
            "SM","SF","FU","SC","BU","LU","PG","TA",
            "PF","EG","EB","L","MA","PP","SA","UR",
            "V","FG","NR","RU","AU","AG", 'SP', 'CJ', 
            'PK', 'LH',] 
    universe = sorted(universe)
    return universe

def get_universe_pool() -> dict:
    res = {}
    res['universe'] = get_universe()
    res['agr'] = get_agr_list()
    res['chem'] = get_chem_list()
    res['ferrous'] = get_ferrous_list()
    res['energy'] = get_energy_list()
    res['metal'] = get_metal_list()
    return res


"""
CONVERT DATA
"""
def my_dataframe(data_dict: dict, string: str) -> pd.DataFrame:
    '''
    Get the pivot table from df_min, like close, open,...
    :param df: (pd.DataFrame) res from my_load_data
    :param string: (str) close, open, volume....
    :return: (pd.DataFrame) pivot table columns=types, index=datetime, values=string
    '''
    res_list = []
    for tp in data_dict:
        series = data_dict[tp][string].rename(tp)
        res_list.append(series[~series.index.duplicated(keep='first')])
    res = pd.concat(res_list, axis=1)
    return res

def my_concat_data_dict(data_dict: dict) -> pd.DataFrame:
    first_key = next(iter(data_dict))
    df = data_dict[first_key].rename({'underlying_symbol': 'types'}, axis=1)
    for tp in data_dict:
        if tp == 'A':
            continue
        dd = data_dict[tp].rename({'underlying_symbol': 'types'}, axis=1)
        df = pd.concat([df, dd])
    return df

def my_train_test_split(data_dict: dict, day: datetime) -> dict:
    '''
    根据日期分为训练测试段
    :return : (dict, dict)
    '''
    train_dict = {}
    test_dict = {}

    for tp in data_dict:
        train_df = data_dict[tp].loc[:day]
        test_df = data_dict[tp].loc[day:]
        train_dict[tp] = train_df
        test_dict[tp] = test_df
    
    return train_dict, test_dict

def my_df_train_test_split(df: pd.DataFrame, day: datetime) -> pd.DataFrame:
    train_df = df.loc[:day]
    test_df = df.loc[day:]
    return train_df, test_df

def my_series_to_df(seires: pd.Series, columns: list) -> pd.DataFrame:
    res = pd.DataFrame()
    for tp in columns:
        res[tp] = seires
    return res


"""
COMPUTE DATA
"""
def my_hold_ret(data_dict: dict, shift_num: int=1) -> pd.DataFrame:
    """
    计算持有收益
    shift_num (int): 未来N天的持有收益
    """
    close_hot = my_dataframe(data_dict=data_dict, string='close')
    open_hot = my_dataframe(data_dict=data_dict, string='open')
    ret = open_hot.shift(-shift_num) / open_hot - 1
    ret.iloc[-1, :] = close_hot.iloc[-1, :] / open_hot.iloc[-shift_num, :] - 1
    return ret.replace(np.inf, np.nan)

def my_ret(data_dict: dict) -> pd.DataFrame:
    '''
    计算到期收益，用来波动率调整
    '''
    close_hot = my_dataframe(data_dict=data_dict, string='close')
    res = close_hot / close_hot.shift(1) - 1
    return res


def my_split_list_of_dfs(dfs, n):
    """
    Split a list of DataFrames into n approximately equal parts.

    Parameters:
    - dfs: list of pd.DataFrame
    - n: int

    Returns:
    - List of lists of DataFrames
    """

    total_dfs = len(dfs)
    sizes = []

    base_size = total_dfs // n
    remainder = total_dfs % n

    for i in range(n):
        if remainder > 0:
            sizes.append(base_size + 1)
            remainder -= 1
        else:
            sizes.append(base_size)

    splits = []
    idx = 0
    for size in sizes:
        splits.append(dfs[idx:idx+size])
        idx += size

    return splits

