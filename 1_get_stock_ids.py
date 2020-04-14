import APIs_StockAnalysis as api_s

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import bs4 as bs
import sys
sys.setrecursionlimit(10000000)

import urllib3
from bs4 import BeautifulSoup
import urllib3.contrib.pyopenssl
urllib3.contrib.pyopenssl.inject_into_urllib3()

import certifi
http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 50)
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
tqdm.pandas()

## Get all list of stocks and Ids
# hp:hist page, sp:stock page, gp:graph page
					   					   
ab = 'A a B b C c D d E e F f G g H h I i J j K k L l M m N n O o P p Q q R r S s T t U u V v W w X x Y y Z z'
ab = ab.split(' ')[::2] + ['others']

list_stock_map = []
for a in ab:
    url = r'https://www.moneycontrol.com/india/stockpricequote/{}'.format(a)
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data)
    table = soup.find_all('table')[1]
    
    data = [[{cell.text:cell.find_all('a')[0].get('href')} for cell in row.find_all('td') if cell.find_all('a')[0].get('href').split('/')[-1]!=''] for row in table.find_all('tr') if len(row.find_all('td'))>0]
    data = pd.Series(data).apply(pd.Series).stack().apply(pd.Series).stack().reset_index()
    data['ind'] = np.arange(1,data.shape[0]+1)
    data['alpha_cat'] = a
    total_stock = data.shape[0]
    data['info'] = data.apply(lambda x: api_s.get_detail_info_of_sc_id(x[0], x['level_2'], x['ind'], total_stock, x['alpha_cat']), axis=1)
    list_stock_map.append(data)
    
df_stock_map = pd.concat(list_stock_map)


df_stock_map = df_stock_map.reset_index().drop(['index', 'level_0', 'level_1', 'ind'], axis=1)
df_stock_map[['sc_id_gp' , 'title_gp' , 'disp_gp' , 'title_hp' , 'sc_id_sp' , 'title_sp' , 'cat']] = df_stock_map['info'].apply(pd.Series)
df_stock_map.rename(columns = {'level_2':'ListedStock', 0:'link'}, inplace=True)

df_stock_map.to_pickle(r'StockList_ScID.pk')


