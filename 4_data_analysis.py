import APIs_StockAnalysis as api_s

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_bse = pd.read_pickle(r'DataBase_Stocks_BSE.pk')
df_bse.head()

df = df_bse[~df_bse['open'].isnull()]

tstock = df_bse['stock'].nunique()
astock = df['stock'].nunique()
print('Number of stocks having data = {}/{}'.format(astock, tstock))
print('Number of Stocks not having data = {}/{}'.format(tstock-astock, tstock))

df = df.reset_index(drop = True).set_index(['sc_id_gp', 'date'])
df = df['close'].unstack(0)
df.head()

# an example of historical stock price evaluation
%matplotlib notebook
df[['Bajaj Auto Ltd.']].plot(style='o-', ms=1, lw=0.5, figsize=(9.5,3))
plt.tight_layout()

# a quick check on one to one relationship among stocks
# to avoid crashing machine, top 100 stocks have been selected.
check = df.reset_index()
check1 = check[check['date']>'2018-01-01'].set_index('date')
check_corr1 = check1[check1.columns[0:100]].corr()

f = plt.figure(figsize=(10, 8))
plt.matshow(check_corr1, fignum=f.number, cmap = 'RdYlGn')
plt.xticks(range(check1[check1.columns[0:100]].shape[1]), check1[check1.columns[0:100]].columns, fontsize=5, rotation=90)
plt.yticks(range(check1[check1.columns[0:100]].shape[1]), check1[check1.columns[0:100]].columns, fontsize=5)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);


## Few criterion defined to get the right stocks to invest in
#Established - More than 5years and exist till today
#    Positive Linear Trend (m>0)
#    Max Return (Max-Starting)
#    Till today return (price at Jan2020 - Starting Price)
#    Market Cap
#    Financial Ratios...


from tqdm import tqdm
tqdm.pandas()
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress
scaler = MinMaxScaler(feature_range=(0,1))

stocks_exist_more5yrs = df.T.apply(lambda x: 1 if (max(x.dropna().index)>pd.to_datetime('2020-01-01')) and (len(x.dropna())>1*300) else np.nan, axis=1).dropna().index
df_stocks_exit_more5yrs = df[stocks_exist_more5yrs]


linear_trend = df_stocks_exit_more5yrs.progress_apply(lambda x: linregress(x.reset_index().dropna().index,
                                                                           x.dropna().values))
linear_trend = linear_trend.apply(pd.Series)
linear_trend.columns = ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']

return_max_start = df_stocks_exit_more5yrs.apply(lambda x: (x.dropna().max() - x.dropna().iloc[0])/(x.dropna().iloc[0]))
return_max_min = df_stocks_exit_more5yrs.apply(lambda x: (x.dropna().max() - x.dropna().min())/(x.dropna().min()))

min_max_nodays = df_stocks_exit_more5yrs.apply(lambda x: {'min_date':min(x.dropna().index),
                                                          'max_date':max(x.dropna().index),
                                                          'total_days':len(x.dropna())}).apply(pd.Series).apply(pd.Series)
last_closing_price = df_stocks_exit_more5yrs.apply(lambda x: x.dropna().iloc[-1])
start_date = df_stocks_exit_more5yrs.apply(lambda x: x.dropna().index[0])
end_date = df_stocks_exit_more5yrs.apply(lambda x: x.dropna().index[-1])

# load detailed stocks data to mapped with the enrich filter API
df_gn_mcap_swot_fr_pl = pd.read_pickle(r'Stocks_GeneralInfo_MCap_SWOT_FR_PL.pk')

cols_to_add = ['title_gp', 'inBSE?', 'inNSE?', 'sector',
       'Book Value (Rs)', 'Deliverables (%)', 'Dividend (%)',
       'Dividend Yield.(%)', 'EPS (TTM)', 'Face Value (RS)', 'Industry P/E',
       'Market Cap (Rs Cr.)', 'Market Lot', 'P/C', 'P/E', 'Price/Book', 'link']
	   
linear_trend['return_max_min'] = return_max_min
linear_trend['return_max_start'] = return_max_start
linear_trend['CP'] = last_closing_price
linear_trend['start_date'] = start_date
linear_trend['end_date'] = end_date

linear_trend[min_max_nodays.columns] = min_max_nodays

linear_trend[cols_to_add] = df_gn_mcap_swot_fr_pl.set_index('sc_id_gp')[cols_to_add]
linear_trend['prime_sector'] = linear_trend['sector'].apply(lambda x: x.split('-')[0])
linear_trend['trend_type'] = linear_trend['slope'].apply(lambda x: 'positive' if x>=0 else 'negative')

## applying few filters to identify the interested stock to invest in 
check = api_s.df_filtering(linear_trend, filters_dict={
                                         'start_date':['>', pd.to_datetime('2015-01-01')],
                                         'CP':['<',1]})
check.sort_values('Market Cap (Rs Cr.)')[['slope', 'CP', 'Price/Book', 'title_gp', 'sector', 'start_date', 'end_date', 'Market Cap (Rs Cr.)', 'link']]

## Quick check on the trend of historical closing price of the interested stock
ticker = 'GB06'
coeeff, pcov = curve_fit(fit_linear,
                         list(range(df[ticker].dropna().shape[0])),
                         df[ticker].dropna(), method='trf')
print(coeeff)

ax = df[[ticker]].dropna().plot(style='o-', ms=1, lw=0.5, figsize=(9.5,3))
pd.Series(fit_linear(np.array(range(df[ticker].dropna().shape[0])), *coeeff), index = df[ticker].dropna().index).plot(ax=ax)
# df[[ticker]].dropna().rolling(int(365/12)).mean().plot(style='o-', ms=1, lw=0.5, c='r', ax=ax,alpha=0.05)
# ax.set_ylim(None, 1000)
plt.tight_layout()




 
