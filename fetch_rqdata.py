'''
Author: Jet Deng
Date: 2023-10-24 15:24:10
LastEditTime: 2023-11-15 15:22:15
Description: 
'''

import rqdatac as rq
import pandas as pd
import numpy as np
from datetime import *
import re
import warnings
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')
rq.init('15882778060', 'Wenwen200671')
# rq.init('15959259161', '1997abc11221')

def fetch_data(path: str, freq: str, adj: bool) -> None:
    '''
    Given the folder path, fetch the corresponding freq data and store
    :param path: (str) folder path for storing data
    :param freq: (str) '1m', '30m', or '1d'...
    :return: None
    '''
    try:
        os.makedirs(path)
    except:
        pass
    
    all_futures =  rq.all_instruments(type="Future", date=pd.to_datetime('today'))["underlying_symbol"].unique()
    info_df = rq.all_instruments('Future').set_index('order_book_id')
    if adj:
        for i, tp in enumerate(all_futures):
            df = rq.futures.get_dominant_price(underlying_symbols=tp,
                                               start_date=pd.to_datetime('20100104'),
                                               end_date=pd.to_datetime('20230925'),
                                               frequency=freq,
                                               adjust_type='pre',
                                               adjust_method='prev_close_ratio')
            df.to_pickle(path + '/' + str(tp) + '.pkl')
            print(f'{tp} download complete {np.round((i+1)/len(all_futures),2)*100}% complete')
    else:
        for i, tp in enumerate(all_futures):
            main_contracts = rq.futures.get_dominant(tp, start_date=None, end_date=None, rule=0, rank=1).tolist()
            submain_contracts = rq.futures.get_dominant(tp, start_date=None, end_date=None, rule=0, rank=2).tolist()
            contracts = main_contracts + submain_contracts 
            df = rq.get_price(contracts,
                            start_date=pd.to_datetime('20100104'),
                            end_date=pd.to_datetime('20230925'),
                            frequency=freq,
                            fields=None,
                            adjust_type='none', 
                            )
            df = df.reset_index().set_index('order_book_id')
            df['maturity_date'] = info_df['maturity_date']
            df.to_pickle(path + '/' + str(tp) +'.pkl')
            print(f'{tp} download complete {np.round((i+1)/len(all_futures)*100, 2)}% complete')
    return 

def fetch_contracts(path: str, rank: int) -> None:
    '''
    rank: 1 -> 主力, 2 -> 次主力
    '''
    try:
        os.makedirs(path)
    except:
        pass
    all_futures =  rq.all_instruments(type="Future", date=pd.to_datetime('today'))["underlying_symbol"].unique()
    contracts_df = pd.DataFrame()
    for tp in all_futures:
        contracts = rq.futures.get_dominant(tp, start_date=None, end_date=None, rule=0, rank=rank)
        contracts_df[tp] = contracts
    contracts_df.to_csv(path + '/contracts.csv')
    return 

        
# def fetch_submain_data(path: str, freq: str) -> None:
#     '''
#     下载次主力合约
#     '''
#     try:
#         os.makedirs(path)
#     except:
#         pass
#     all_futures =  rq.all_instruments(type="Future", date=pd.to_datetime('today'))["underlying_symbol"].unique()
#     info_df = info_df = rq.all_instruments('Future').set_index('order_book_id')
#     for tp in tqdm(all_futures):
#         submain_contracts = rq.futures.get_dominant(tp, start_date='20100104', end_date='20230925', rule=0, rank=2)
#         df = rq.get_price(order_book_ids=submain_contracts,
#                           start_date='20100104',
#                           end_date='20230925',
#                           frequency='1d',
#                           )
#         df = df.reset_index().set_index('order_book_id')
#         df['maturity_date'] = info_df['maturity_date']
#         df.to_pickle(path + '/' + str(tp) + '.pkl')
#         print(f'{tp} download complete')
#     return 

def prepare_data(types: list, file_path: str) -> pd.DataFrame:
    '''
    Concat the dataframes of needed types
    :param types: (list) needed types
    :param file_path: (str) reading pickle files
    :return: (pd.DataFrame) of the final concatnated dataframe
    '''
    df = pd.read_pickle(f'{file_path}/A.pkl')
    for i, tp in enumerate(types):
        if i == 0:
            continue
        path = f'{file_path}/{tp}.pkl'
        to_concat = pd.read_pickle(path)
        df = pd.concat([df, to_concat])
        print(f'{tp} complete --{i}')
    df.to_pickle('future_minbar.pkl')
    print('Complete')
    return 

if __name__ == '__main__':
    folder_path_1 = "D://projects/data/minbar/1min/unadj"
    folder_path_2 = "D://projects/data/basis"
    freq = '1m'
    adj = False
    # fetch_data(path=folder_path_1, freq=freq, adj=adj)
    need = ["M","RM","P","Y","OI","B","C","CS","A",
            "JD","SR","CF","AP","AL","CU","ZN","NI",
            "PB","SN","SS","RB","HC","I","J","JM",
            "SM","SF","FU","SC","BU","LU","PG","TA",
            "PF","EG","EB","L","MA","PP","SA","UR",
            "V","FG","NR","RU","AU","AG", 'SP']
    
    # prepare_data(types=need, file_path=folder_path)


