
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

# set the seed to 42

df = yf.download(tickers='BTC-USD', period='10d', interval='30m')#cv


df['TargetNextClose'] = df['Adj Close'].shift(-1)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 36)


rf = RandomForestRegressor(n_estimators=1500)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f"Tomorrow's prediction {y_pred[-1]}")

mae = mean_absolute_error(y_test, y_pred)

print(mae)
