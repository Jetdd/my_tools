"""
Author: Jet Deng
Date: 2023-11-09 10:33:12
LastEditTime: 2023-11-09 10:33:24
Description: Some Useful RiceQuant Functions
"""
import rqdatac as rq
import pandas as pd
import numpy as np
from pathlib import Path
rq.init("15882778060", "Wenwen200671")


def get_trading_date(start_date: str, end_date: str) -> list:
    """得到起始和结束日期中间的交易日List

    Args:
        start_date (str):
        end_date (str):

    Returns:
        list: (datetime.date) of trading_dates
    """
    trading_date_list = rq.get_trading_dates(
        start_date=start_date, end_date=end_date, market="cn"
    )
    return trading_date_list

def get_data(path: str, freq: str, adj: bool) -> None:
    '''
    Given the folder path, fetch the corresponding freq data and store
    :param path: (str) folder path for storing data
    :param freq: (str) '1m', '30m', or '1d'...
    :return: None
    '''
    path = Path(path).mkdir(parents=True, exist_ok=True)
    
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

if __name__ == '__main__':
    # 1. 下载交易日列表并储存
    trading_date_list = get_trading_date(start_date='20210104', end_date='20241231')