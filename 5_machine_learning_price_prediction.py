from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

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


#Getting top companies list
url = r'https://www.moneycontrol.com/stocks/marketstats/bse-gainer/bse-500_12/'
# url = r'https://www.moneycontrol.com/stocks/marketstats/nse-gainer/nifty-500_7/'
response = http.request('GET', url)
soup = BeautifulSoup(response.data)
table = soup.find_all('table')[1]
row = table.find_all('tr')
all_cells = [x.find_all('td')[0].find_all('a')[0] for x in row[1:][::7]]
bse_500 = pd.Series({x.text:x.get('href').split('/')[-1] for x in all_cells})
company_name_bse500 = df_stock_map.set_index('sc_id_sp').ix[bse_500.values]['title_gp'].values


# Load historical stock closing price data
df_bse = pd.read_pickle(r'DataBase_Stocks_BSE.pk')
df_bse.head()

df = df_bse[~df_bse['open'].isnull()]

tstock = df_bse['stock'].nunique()
astock = df['stock'].nunique()
print('Number of stocks having data = {}/{}'.format(astock, tstock))
print('Number of Stocks not having data = {}/{}'.format(tstock-astock, tstock))

df = df.reset_index(drop = True).set_index(['sc_id_gp', 'date'])
df = df['close'].unstack(0)

df = df[company_name_bse500]

# let's predict 'Bajaj Auto Ltd.' to buy, sell or hold..
tickers = df.columns
ticker = 'Bajaj Auto Ltd.'

df = df.dropna(subset = [ticker]).fillna(0)
for x in range(1,8):
    df['{}_{}d'.format(ticker, x)] = (df[ticker].shift(-x) - df[ticker])/df[ticker]
	
	
df = df.replace([np.inf, -np.inf], [np.nan, np.nan])

def buy_sell_hold(cols):
    req = 0.02
    for col in cols:
        if col>req:
            return 1
        if col<-req:
            return -1
    return 0
	
df['{}_target'.format(ticker)] = df.apply(lambda x: buy_sell_hold(x[['{}_{}d'.format(ticker, i) for i in range(1,8)]].values), axis=1)

df['{}_target'.format(ticker)].value_counts()

df_vals = df[tickers].pct_change()
df_vals = df_vals.replace([np.inf, -np.inf], [0, 0])
df_vals.fillna(0, inplace=True)


X = df_vals.values
y = df['{}_target'.format(ticker)].values

# Implementing ML models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
clf = neighbors.KNeighborsClassifier()
clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                        ('knn', neighbors.KNeighborsClassifier()),
                        ('rfor', RandomForestClassifier())])

#X_train, X_test, y_train, y_test = X[0:2195], X[2195:], y[0:2195], y[2195:]

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
predictions = clf.predict(X_test)

print(confidence)
print(pd.Series(predictions).value_counts())
print(pd.DataFrame([predictions, y_test]).T.groupby(1)[0].value_counts())


	
	
	