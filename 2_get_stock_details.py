import APIs_StockAnalysis as api_s

## Get Stock Details
from multiprocessing import Pool
import time
import pandas as pd
import numpy as np


## Get all financial details of stocks
df_stock_map = pd.read_pickle(r'StockList_ScID.pk')
df_stock_map['ind'] = np.arange(1, df_stock_map.shape[0]+1)

ind_check = df_stock_map.loc[df_stock_map['sc_id_gp'].isnull(), 'sc_id_gp'].index
df_stock_map.loc[df_stock_map['sc_id_gp'].isnull(), 'sc_id_gp'] = ['null - '+str(x) for x in  range(1, df_stock_map.loc[df_stock_map['sc_id_gp'].isnull()].shape[0]+1)]
df_stock_map.drop_duplicates('sc_id_gp', inplace=True)

df_stock_map = df_stock_map.drop(['info', 'disp_gp'], axis=1)

	
df_stock_map['ind'] = np.arange(1, df_stock_map.shape[0]+1)
total_shares = df_stock_map.shape[0]
p = Pool(processes = 100)

start = time.perf_counter()
data_pool = p.starmap(api_s.get_stocks_details_fr_pl, [[x[0], x[1], x[2], total_shares] for x in df_stock_map[['link', 'sc_id_sp', 'ind']].values])
p.close()
finish = time.perf_counter()
print('Total Time {}'.format(round(finish-start, 2)))

# to load stocks which might not get loaded...
df_stock_map['index_check'] = df_stock_map['ind'].map(pd.Series({x['index_to_mapback']:x['index_to_mapback'] for x in data_pool}).reset_index().dropna(subset=['index']).set_index('index')[0])
unloaded_stocks = [stock_data.get_stocks_details_fr_pl(*x, total_shares) for x in df_stock_map[df_stock_map['index_check'].isnull()][['link', 'sc_id_sp', 'ind']].values]
data_pool = data_pool + unloaded_stocks

df_stock_map['index_check'] = df_stock_map['ind'].map(pd.Series({x['index_to_mapback']:x['index_to_mapback'] for x in data_pool}).reset_index().dropna(subset=['index']).set_index('index')[0])
df_stock_map[df_stock_map['index_check'].isnull()]

# pd.Series(data_pool).to_pickle(r'BackUp_RawStocksInfo.pk')


valuation_table = pd.Series({x['index_to_mapback']:x['valuation'] for x in data_pool}).apply(pd.Series).stack().apply(pd.Series).stack().apply(pd.Series).reset_index().set_index(['level_0', 0])[1].unstack()

basic_info = pd.Series([{y+'_':x[y] for y in ['sc_id_gp', 'title_gp', 'bse_status', 'nse_status']} for x in  data_pool],
                      index=[x['index_to_mapback'] for x in data_pool]).apply(pd.Series)
basic_info['inBSE?'] = basic_info['bse_status_'].apply(lambda x: ('No' if 'not listed' in x else 'Yes') if str(x)!='nan' else 'Unknown')
basic_info['inNSE?'] = basic_info['nse_status_'].apply(lambda x: ('No' if 'not listed' in x else 'Yes') if str(x)!='nan' else 'Unknown')
basic_info['sector'] = pd.Series({x['index_to_mapback']:x['nse_bse_info'] for x in data_pool}).dropna().apply(lambda x: x[-1] if len(x)!=0 else np.nan)

swot_details = pd.Series([x['swot_info'] for x in data_pool], index=[x['index_to_mapback'] for x in data_pool]).dropna().apply(pd.Series).drop(np.nan, axis=1)

fr_df_dict = {}
for fr in data_pool:
    if str(fr['fr_ratio'])!='nan':
        fr_df = pd.concat([pd.DataFrame(x).set_index(0).T for x in fr['fr_ratio']])
        fr_df = fr_df.replace('\xa0', np.nan).dropna(how='all').replace('',np.nan).T
        fr_df.index = ['Date'] + list(fr_df.index[1:])
        fr_df = fr_df.T.set_index('Date').T
        grp_info = fr_df.apply(lambda x: x.isnull().all(), axis=1).replace(False, np.nan).dropna()
        fr_df['grp'] = pd.Series(grp_info.index, grp_info.index)
        fr_df['grp'] = fr_df['grp'].fillna(method='ffill')
        fr_df = fr_df.dropna(subset=fr_df.drop('grp', axis=1).columns, how='all').reset_index().set_index(['grp', 'index']).T
        
        fr_df['index_to_mapback'] = fr['index_to_mapback']
    else:
        #fr_df = pd.DataFrame([fr['index_to_mapback']], columns=['index_to_mapback'], index=['']).unstack().to_frame().T
        fr_df = np.nan
    fr_df_dict[fr['index_to_mapback']] = fr_df
	
	
	
total_pl = len(data_pool)
pl_df_dict = {}
for fr in data_pool:
    print('Done.................{}%'.format(round(fr['index_to_mapback']*100/total_pl, 2)), end='\r')
    if str(fr['profit_loss'])!='nan':
        fr_df = pd.concat([pd.DataFrame(x).set_index(0).T for x in fr['profit_loss']])
        fr_df = fr_df.replace('\xa0', np.nan).dropna(how='all').replace('',np.nan).T
        fr_df.index = ['Date'] + list(fr_df.index[1:])
        fr_df = fr_df.T.set_index('Date').T
        grp_info = fr_df.apply(lambda x: x.isnull().all(), axis=1).replace(False, np.nan).dropna()
        fr_df['grp'] = pd.Series(grp_info.index, grp_info.index)
        fr_df['grp'] = fr_df['grp'].fillna(method='ffill')
        fr_df = fr_df.dropna(subset=fr_df.drop('grp', axis=1).columns, how='all').reset_index().set_index(['grp', 'index']).T
        
        fr_df['index_to_mapback'] = fr['index_to_mapback']
    else:
        #fr_df = pd.DataFrame([fr['index_to_mapback']], columns=['index_to_mapback'], index=['']).unstack().to_frame().T
        fr_df = np.nan
    pl_df_dict[fr['index_to_mapback']] = fr_df
	
	
df_stock_map[basic_info.columns] = basic_info[~basic_info.index.isnull()].apply(lambda x: df_stock_map['ind'].map(x))
df_stock_map[valuation_table.columns] = valuation_table.apply(lambda x: df_stock_map['ind'].map(x)).astype('str').apply(lambda x: x.apply(lambda y: y.replace(',',''))).replace(['-', 'nan'], [np.nan, np.nan]).astype(float)
df_stock_map[swot_details.columns] = swot_details.apply(lambda x: df_stock_map['ind'].map(x))


fr_df_dict = pd.Series(fr_df_dict)
df_stock_map['fr_ratio'] = df_stock_map['ind'].map(fr_df_dict[~fr_df_dict.index.isnull()])

pl_df_dict = pd.Series(pl_df_dict)
df_stock_map['pl_ratio'] = df_stock_map['ind'].map(pl_df_dict[~pl_df_dict.index.isnull()])

df_stock_map.to_pickle(r'Stocks_GeneralInfo_MCap_SWOT_FR_PL.pk')

