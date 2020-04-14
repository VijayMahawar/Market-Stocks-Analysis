import warnings
warnings.filterwarnings('ignore')

from requests import request

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


import numpy as np
import pandas as pd


def get_data(url):
    try:
        response = http.request('GET', url)
        soup = BeautifulSoup(response.data)

        table = soup.find_all('table')[2]

        data = [[cell.text for cell in row.find_all('th') ] for row in table.find_all('tr')[0:2]] +\
               [[cell.text for cell in row.find_all('td') ]for row in table.find_all('tr')] 

        df = pd.DataFrame(data).dropna(how='all').drop([6,7], axis=1).drop(1).T.set_index(0).T.reset_index().drop('index', axis=1)
    except:
        df = pd.DataFrame()
    return df
	
def get_page_numbers(share_id, date_start, date_end):
    url = r'http://www.moneycontrol.com/stocks/hist_stock_result.php?sc_id={}&pno=1&hdn=daily&fdt={}&todt={}'.format(share_id, date_start, date_end)
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data)
    try:
        pages = '1' + soup.getText().split('Next')[0].split('1')[-1]
        pages = [int(x) for x in pages.split(' ')[0:-1]]
        page_to_add = pages[-1] + 1

        more_pages = True
        while more_pages:
            try:
                url = r'http://www.moneycontrol.com/stocks/hist_stock_result.php?sc_id={}&pno={}&hdn=daily&fdt={}&todt={}'.format(share_id, page_to_add, date_start, date_end)
                response = http.request('GET', url)
                soup = BeautifulSoup(response.data)
                pages = pages +[int(x) for x in soup.getText().split('Previous')[-1].split('Next')[0].split(' ')[1:-1]]
                page_to_add = pages[-1] + 1
                #print('worked')
            except:
                more_pages=False
        return pages
    except:
        pages = [1]
        return pages
		
def get_stock_details(share_id, date_start, date_end, ind, total):
    print('Start.....................{} | {}/{}...{}%'.format(share_id, ind, total, round(ind*100/total, 0)))
    pages = get_page_numbers(share_id, date_start, date_end)
    stock_data_list = []
    for p in pages:
        print('Loading...{}/{} | {}% | {} ----------------------------------PageNO {}/{}'.format(ind, total, round(ind*100/total, 0), share_id, p, len(pages)))
        url = r'http://www.moneycontrol.com/stocks/hist_stock_result.php?sc_id={}&pno={}&hdn=daily&fdt={}&todt={}'.format(share_id, p, date_start, date_end)
        df = get_data(url)
        df['PageNo'] = p
        df['Share_ID'] = share_id
        stock_data_list.append(df)
    return pd.concat(stock_data_list).reset_index().drop('index', axis=1)
	
def get_data_from_graph_on_web(sc_id_sp, ind, total):
    try:
        print('Loading.................................{} | {}/{}...{}%'.format(sc_id_sp, ind, total, round(ind*100/total,2)))
        url = r'https://www.moneycontrol.com/mc/widget/basicchart/get_chart_value?classic=true&sc_did={}&dur=max'.format(sc_id_sp)
        response=request(url=url, method='get')
        response_json = response.json()

        data = pd.DataFrame(response_json['g1'])
        float_cols = list(data.columns)
        float_cols.remove('date')
        data[float_cols] = data[float_cols].astype(float)
        data['date'] = pd.to_datetime(data['date'], dayfirst=False)    
        data['sc_id_sp'] = sc_id_sp

        return data
    except:
        data = pd.DataFrame([sc_id_sp], columns=['sc_id_sp'])
        return data
	
	
def get_data_using_csv_file(sc_id_gp, ind, total):
    try:
        print('Loading.................................{} | {}/{}...{}%'.format(sc_id_gp, ind, total, round(ind*100/total,2)))
        url = r'https://www.moneycontrol.com/tech_charts/bse/his/{}.csv'.format(sc_id_gp.lower())
        response = http.request('GET', url)
        soup = BeautifulSoup(response.data)
        data = pd.Series(soup.text.split('\n')).apply(lambda x: x.split(',,')[0].split(',')).apply(pd.Series).dropna(axis=1)
        data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        float_cols = list(data.columns)
        float_cols.remove('date')
        data[float_cols] = data[float_cols].astype(float)
        data['date'] = pd.to_datetime(data['date'], dayfirst=False)
        data['sc_id_gp'] = sc_id_gp
        return data
    except:
        data = pd.DataFrame([sc_id_gp], columns=['sc_id_gp'])
        return data	
	
	
def get_stocks_details_fr_pl(url, sc_id_sp, curr_ind, total_ind):
    print('Loading.....{}%_____{}/{}..................{}'.format(round(curr_ind*100/total_ind, 2), curr_ind, total_ind, sc_id_sp))
		
    try:
        response = http.request('GET', url)
        soup = BeautifulSoup(response.data)
        web_page_load = True
    except:
        web_page_load = False

    if web_page_load:
        #Getting sc_id and title
        try:
            sc_id_graph_page = soup.find_all(class_='techany smaD')[0].find_all(class_='viewmore')[0].find_all('a')[0].get('href').split('=')[-1]
        except:
            sc_id_graph_page = np.nan
        try:
            title_graph_page = soup.find_all(class_='pcstname')[0].text
        except:
            title_graph_page = np.nan

        try:
            in_bse = soup.find_all(class_='bselft')[0].find_all(class_='not_tradedbx')[0].text
        except:
            in_bse = np.nan
        try:
            in_nse = soup.find_all(class_='nsert')[0].find_all(class_='not_tradedbx')[0].text
        except:
            in_nse = np.nan

        #Company NSE BSE codes
        try:
            nse_bse_info = [x.text for x in soup.find_all(class_ = 'clearfix MT10')[0].find_all('span')][0:8]
        except:
            nse_bse_info = np.nan

        #Getting SWOT details
        try:
            url_swot = [li.find_all('a')[0].get('href') for li in soup.find_all(class_='swot_feature')[0].find_all('li')][0]
            response_swot = http.request('GET', url_swot)
            soup_swot = BeautifulSoup(response_swot.data)
            swot_details = {div.get('id'):[li.text for li in div.find_all('li')] for div in soup_swot.find_all(class_='tab-content')[1].find_all('div')[::2]}
        except:
            swot_details = np.nan

        #Getting valuation table
        try:
            tabs = soup.find_all(class_='valuation_lft')[0].find_all(class_='tab-content MT10')[0].find_all('div')[0].find_all('ul')[1:]
            valuation_table = [[[di.text if len(di.find_all('a'))==0 else di.find_all('a')[0].text for di in li.find_all('div')][0:2] for li in a.find_all('li')] for a in tabs]
        except:
            valuation_table = np.nan

        ## Getting financial ratio info
        try:
            ##.find_all(class_='tab-content')[0].find_all('div')[---0---] and [--1---], --0-- for standalone data and --1-- for consolidated data 
            url_fr = [x.get('href') for x in soup.find_all(class_='finance_lft')[0].find_all(class_='tab-content')[0].find_all('div')[0].find_all('a') if x.text=='Financial Ratios'][0]
            response_fr = http.request('GET', url_fr)
            soup_fr = BeautifulSoup(response_fr.data)
            table_fr = soup_fr.find_all(class_='financial-section')[0].find_all(class_='tab-content')[0].find_all('table')[0]
            fr_data = [[td.text for td in tr.find_all('td')] for tr in table_fr.find_all('tr')]

            fr_data_list = [fr_data]
            url_fr_np = soup_fr.find_all(class_='financial-section')[0].find_all('ul')[0].find_all('li')[1].find_all('a')[0].get('href')
            while url_fr_np.startswith('https'):
                response_fr_np = http.request('GET', url_fr_np)
                soup_fr_np = BeautifulSoup(response_fr_np.data)
                table_fr = soup_fr_np.find_all(class_='financial-section')[0].find_all(class_='tab-content')[0].find_all('table')[0]
                fr_data = [[td.text for td in tr.find_all('td')] for tr in table_fr.find_all('tr')]
                fr_data_list.append(fr_data)
                url_fr_np = soup_fr_np.find_all(class_='financial-section')[0].find_all('ul')[0].find_all('li')[1].find_all('a')[0].get('href')
        except:
            fr_data_list=np.nan

        ## Getting Profit Loss info
        try:
            ##.find_all(class_='tab-content')[0].find_all('div')[---0---] and [--1---], --0-- for standalone data and --1-- for consolidated data 
            url_fr = [x.get('href') for x in soup.find_all(class_='finance_lft')[0].find_all(class_='tab-content')[0].find_all('div')[0].find_all('a') if x.text=='Profit & Loss'][0]
            response_fr = http.request('GET', url_fr)
            soup_fr = BeautifulSoup(response_fr.data)
            table_fr = soup_fr.find_all(class_='financial-section')[0].find_all(class_='tab-content')[0].find_all('table')[0]
            fr_data = [[td.text for td in tr.find_all('td')] for tr in table_fr.find_all('tr')]

            pl_data_list = [fr_data]
            url_fr_np = soup_fr.find_all(class_='financial-section')[0].find_all('ul')[0].find_all('li')[1].find_all('a')[0].get('href')
            while url_fr_np.startswith('https'):
                response_fr_np = http.request('GET', url_fr_np)
                soup_fr_np = BeautifulSoup(response_fr_np.data)
                table_fr = soup_fr_np.find_all(class_='financial-section')[0].find_all(class_='tab-content')[0].find_all('table')[0]
                fr_data = [[td.text for td in tr.find_all('td')] for tr in table_fr.find_all('tr')]
                pl_data_list.append(fr_data)
                url_fr_np = soup_fr_np.find_all(class_='financial-section')[0].find_all('ul')[0].find_all('li')[1].find_all('a')[0].get('href')
        except:
            pl_data_list=np.nan

        #Getting company details
        try:
            comp_info = [x.text for x in soup.find_all(class_='abt_cntExpand')[0].find_all('p')]
        except:
            comp_info = np.nan

        # {'sc_id_gp':sc_id_graph_page,
        # 'title_gp':}

        return_dict = {'sc_id_gp':sc_id_graph_page,
                       'title_gp':title_graph_page,
                       'bse_status':in_bse,
                       'nse_status':in_nse,
                       'nse_bse_info':nse_bse_info,
                       'swot_info':swot_details,
                       'valuation':valuation_table,
                       'fr_ratio':fr_data_list,
                       'profit_loss':pl_data_list,
                       'comp_info':comp_info,
                       'index_to_mapback':curr_ind}
    else:
        return_dict = {'sc_id_gp':np.nan,
                       'title_gp':np.nan,
                       'bse_status':np.nan,
                       'nse_status':np.nan,
                       'nse_bse_info':np.nan,
                       'swot_info':np.nan,
                       'valuation':np.nan,
                       'fr_ratio':np.nan,
                       'profit_loss':np.nan,
                       'comp_info':np.nan,
                       'index_to_mapback':curr_ind}
    return return_dict
	
def df_filtering(df, filters_dict, return_other=False):
    """
    filtering any df from filter defined by a filters dictionnary

    Parameters
    ----------
    df : pandas dataframe
        input data
    filters_dict : dictionnary defining filter
      take this form {'column1': ['==', 'a'],
                      'column2': ['!=', ['a', 'b']],
                      'column3': ['>', 2.3],
                      'column4': ['<=', 2.3]}
                    or list of filters_dict
    Returns
    -------
    df : pandas dataframe
        the corresponding filtered dataframe

    """
    # TODO consistency of dict keys.
    if filters_dict is None:
        return df
    else:
        ok = np.ones((len(df)), dtype=bool)

        if not isinstance(filters_dict, list):
            filters_dict_list = [filters_dict]
        else:
            filters_dict_list = filters_dict

        for filters_dict in filters_dict_list:

            for i, key in enumerate(filters_dict.keys()):

                # force type list..
                if not hasattr(filters_dict[key][1], '__iter__'):
                    filters_dict[key][1] = [filters_dict[key][1]]

                if filters_dict[key][0] == '==':
                    ok_ = np.zeros((len(df)), dtype=bool)
                    for val in filters_dict[key][1]:
                        ok_ += df[key] == val
                    ok *= ok_

                elif filters_dict[key][0] == '!=':
                    for val in filters_dict[key][1]:
                        ok *= df[key] != val

                elif filters_dict[key][0] == '>':
                    for val in filters_dict[key][1]:
                        ok *= df[key] > val

                elif filters_dict[key][0] == '<':
                    for val in filters_dict[key][1]:
                        ok *= df[key] < val

                elif filters_dict[key][0] == '>=':
                    for val in filters_dict[key][1]:
                        ok *= df[key] >= val

                elif filters_dict[key][0] == '<=':
                    for val in filters_dict[key][1]:
                        ok *= df[key] <= val

                elif filters_dict[key][0] == 'in':
                    ok *= df[key].isin(filters_dict[key][1])
                
                elif filters_dict[key][0] == 'notin':
                    ok *= (-df[key].isin(filters_dict[key][1]))

                else:
                    warnings.warn('{} is not a known comparison symbol'.format(filters_dict[key][0]))

        if return_other:
            return df.ix[ok, :], df.ix[-ok, :]
        else:
            return df.ix[ok, :]
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	