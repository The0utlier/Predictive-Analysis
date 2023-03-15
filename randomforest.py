from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import yfinance as yf
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler



df = yf.download(tickers='BTC-USD', period='10d', interval='30m')
df['TargetNextClose'] = df['Adj Close'].shift(-1)

'''
df['EMA_12'] = df['Adj Close'].ewm(span=12).mean()
df['EMA_26'] = df['Adj Close'].ewm(span=26).mean()
df['Divergence'] = df['EMA_12'] - df['EMA_26']
df['OBV'] = ta.obv(df.Close, df.Volume)
df['EMAF'] = ta.ema(df.Close, length=20)
#df['EMAM'] = ta.ema(df.Close, length=90)
#df['EMAS'] = ta.ema(df.Close, length=150)
df['SMA_5'] = ta.sma(df.Close, 5)
df['EMA_10'] = ta.ema(df.Close, 10)
df['RSI_14'] = ta.rsi(df.Close, 14)
df['ATR_14'] = ta.atr(df.High, df.Low, df.Close, 14)
df['CCI_14'] = ta.cci(df.High, df.Low, df.Close, 14)
#df['BBANDS_20'] = ta.bbands(df.Close, 20)
#df['MACD_12_26'] = ta.macd(df.Close, 12, 26)
#df['STOCH_14_3_3'] = ta.stoch(df.High, df.Low, df.Close, 14, 3, 3)
df['WILLR_14'] = ta.willr(df.High, df.Low, df.Close, 14)
#df['ADX_14'] = ta.adx(df.High, df.Low, df.Close, 14)
df['Fib_0.236'] = (df['High'] - df['Low']) * 0.236 + df['Low']
df['Fib_0.382'] = (df['High'] - df['Low']) * 0.382 + df['Low']
df['Fib_0.5'] = (df['High'] - df['Low']) * 0.5 + df['Low']
df['Fib_0.618'] = (df['High'] - df['Low']) * 0.618 + df['Low']
df['Fib_0.786'] = (df['High'] - df['Low']) * 0.786 + df['Low']
'''

df = df.drop(columns=['Adj Close'])
df.dropna(inplace=True) #false
# Remove outliers using the IQR method
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

independant = [i for i in df.columns if i != 'TargetNextClose']
X = df[independant]
y = df['TargetNextClose']

# Normalize the data using StandardScaler
scaler = StandardScaler()
df[independant] = scaler.fit_transform(df[independant])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=500)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(mae)# create a new dataframe for test data

y_test_targets = rf.predict(X_test)
# create a new dataframe for test data

test_df = pd.concat([X_test, y_test], axis=1)

test_df['Direction'] = np.where(test_df['TargetNextClose'] > test_df['Close'], 1, 0)


