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
from pytrends.request import TrendReq
# set the seed to 42

df = yf.download(tickers='BTC-USD', period='90d', interval='1d')#cv


#df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = ta.macd(df.Close)
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


df['TargetNextClose'] = df['Adj Close'].shift(-1)


df['Target'] = np.where(df['TargetNextClose'] > df['Close'], 1, 0)

df = df.dropna()

y = df['Target']

df = df.drop(['Close', 'Adj Close', 'TargetNextClose', 'Target'], axis=1)

independant = [i for i in df.columns]

X = df[independant]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 36)

# Define the model and hyperparameters to tune
model = RandomForestRegressor()
params = {'n_estimators': [10, 50, 100],
          'max_depth': [5, 10, 20]}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = grid_search.predict(X_test)

# Convert the predicted values to binary using a threshold of 0.5
y_pred_binary = np.where(y_pred > 0.5, 1, 0)

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)
