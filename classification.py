from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import yfinance as yf
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn import svm
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = yf.download(tickers='BTC-USD', period='10d', interval='90m')#cv

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
df['TargetNextClose'] = df['Adj Close'].shift(-1)

df['Target'] = np.where(df['TargetNextClose'] > df['Close'], 1, 0)

df.dropna(inplace=True) #false
# Remove outliers using the IQR method
'''
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
'''


# Normalize the data using StandardScaler
'''
scaler = StandardScaler()
'''

y = df['Target']

df = df.drop(['Adj Close', 'TargetNextClose', 'Target'], axis=1)

X = df
'''
X = scaler.fit_transform(X)
'''



# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 36)

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rfc.fit(X_train, y_train)

# Predict the values of the test set
y_pred = rfc.predict(X_test)

# Convert the predicted values to binary using a threshold of 0.5
print(len(y_pred))

print(y_pred)


# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

svc = svm.SVC() #RBF

# Fit the model to the training data
svc.fit(X_train, y_train)

# Predict the values of the test set
y_pred = svc.predict(X_test)

print(y_pred)

# Convert the predicted values to binary using a threshold of 0.5

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print(y_pred)

# Convert the predicted values to binary using a threshold of 0.5

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(y_pred)

# Convert the predicted values to binary using a threshold of 0.5

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='adam', alpha=1e-5, max_iter=10000,
                     hidden_layer_sizes=(10,5,2), random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(y_pred)

# Convert the predicted values to binary using a threshold of 0.5

# Evaluate the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
