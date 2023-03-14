
from sklearn.ensemble import RandomForestRegressor
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
from pytrends.request import TrendReq
# set the seed to 42

df = yf.download(tickers='BTC-USD', period='90d', interval='1d')#cv


#noramlize and remove outliers

#df = df[np.abs(data-data.mean()) <= (3*data.std())] 
#print(df.tail(1))
#df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.macd(df.Close)
df['TargetNextClose'] = df['Adj Close'].shift(-1)
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

#print(df.tail(10))
#fibonnacci

#print(df.tail(10))
columns = [i for i in df.columns if i != "Close"]
print(columns)
df = df[columns]
df = df.dropna()
#df = df.drop(columns=['EMA_12', 'EMA_26'], axis=1)



independant = [i for i in columns if i != 'TargetNextClose']
X = df[independant]
y = df['TargetNextClose']
#X = X[corr[corr[0] > 0.05].index]#0.75
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#perfect = cross_validation()['n_estimators']
#print(perfect)
#np.random.seed(154)#randint
# Instantiate the model with 100 decision trees

rf = RandomForestRegressor(n_estimators=500)#400)

rf.fit(X_train, y_train)

# Make predictions on the test data
#y_pred = rf.predict(X[-1:])

#print(f"Prediction for tomorrow: {y_pred}")

y_pred = rf.predict(X_test)

print(f"Tomorrow's prediction {y_pred[-1]}")

mae = mean_absolute_error(y_test, y_pred)
print(mae)
