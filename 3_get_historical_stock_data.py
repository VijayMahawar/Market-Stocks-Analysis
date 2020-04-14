import APIs_StockAnalysis as api_s

## Get Historical Stock Data
from multiprocessing import Pool
import time
import pandas as pd
import numpy as np
import stock_data


df_stock_map = pd.read_pickle(r'StockList_ScID.pk')
df_stock_map['ind'] = np.arange(1, df_stock_map.shape[0]+1)

ind_check = df_stock_map.loc[df_stock_map['sc_id_gp'].isnull(), 'sc_id_gp'].index
df_stock_map.loc[df_stock_map['sc_id_gp'].isnull(), 'sc_id_gp'] = ['null - '+str(x) for x in  range(1, df_stock_map.loc[df_stock_map['sc_id_gp'].isnull()].shape[0]+1)]
df_stock_map.drop_duplicates('sc_id_gp', inplace=True)


total_shares = df_stock_map.shape[0]

p = Pool(processes = 100)

start = time.perf_counter()

data_pool = p.starmap(api_s.get_data_using_csv_file, [[x[0], x[1], total_shares] for x in df_stock_map[['sc_id_gp', 'ind']].values])

p.close()
finish = time.perf_counter()
print('Total Time {}'.format(round(finish-start, 2)))


df_bse = pd.concat(data_pool)

df_bse = df_bse.reset_index().rename(columns={'index':'sr.no'})
df_bse['stock'] = df_bse['sc_id_gp'].map(df_stock_map.set_index('sc_id_gp')['title_gp'])
df_bse = df_bse[['sr.no', 'date', 'high', 'low', 'open', 'close', 'volume', 'sc_id_gp', 'stock']]

df_bse.to_pickle(r'DataBase_Stocks_BSE.pk')
