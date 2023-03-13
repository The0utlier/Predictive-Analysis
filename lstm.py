import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas_ta as ta
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Download the historical data for Bitcoin
df = yf.download(tickers='BTC-USD', period='90d', interval='1d')

# Add technical indicators to the data frame
df['EMA_12'] = df['Adj Close'].ewm(span=12).mean()
df['EMA_26'] = df['Adj Close'].ewm(span=26).mean()
df['Divergence'] = df['EMA_12'] - df['EMA_26']
df['OBV'] = ta.obv(df.Close, df.Volume)
df['EMAF'] = ta.ema(df.Close, length=20)
df['SMA_5'] = ta.sma(df.Close, 5)
df['EMA_10'] = ta.ema(df.Close, 10)
df['RSI_14'] = ta.rsi(df.Close, 14)
df['ATR_14'] = ta.atr(df.High, df.Low, df.Close, 14)
df['CCI_14'] = ta.cci(df.High, df.Low, df.Close, 14)
df['WILLR_14'] = ta.willr(df.High, df.Low, df.Close, 14)
df['Fib_0.236'] = (df['High'] - df['Low']) * 0.236 + df['Low']
df['Fib_0.382'] = (df['High'] - df['Low']) * 0.382 + df['Low']
df['Fib_0.5'] = (df['High'] - df['Low']) * 0.5 + df['Low']
df['Fib_0.618'] = (df['High'] - df['Low']) * 0.618 + df['Low']
df['Fib_0.786'] = (df['High'] - df['Low']) * 0.786 + df['Low']

df['Target'] = df['Adj Close'].shift(-1)

returns = df['Adj Close'].pct_change()
df = df.dropna()
min_max_scaler = MinMaxScaler()

x = df.values
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Flatten this matrix down.
npa = returns.values[1:].reshape(-1,1)
scale = MinMaxScaler(feature_range=(0,1))
npa = scale.fit_transform(npa)

# Need the data to be in the form [sample, time steps, features (dimension of each element)]
samples = 3 # Number of samples (in past)
steps = 1 # Number of steps (in future)
X = [] # X array
Y = [] # Y array
for i in range((df).shape[0] - samples):
    X.append(df.iloc[i:i+samples, :-1].values) # Independent Samples (all columns except target)
    Y.append(df.iloc[i+samples, -1:].values) # Dependent Samples (only target column)
print('Training Data: Length is ',len(X[0:1][0]),': ', X[0:1])
print('Testing Data: Length is ', len(Y[0:1]),': ', Y[0:1])

# Reshape the data so that the inputs will be acceptable to the model.
X = np.array(X)
Y = np.array(Y)
X = np.reshape(X, (X.shape[0], samples, X.shape[2])) # Reshape X
Y = np.reshape(Y, (Y.shape[0], Y.shape[1])) # Reshape Y

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# Build the model
model = keras.Sequential()
model.add(layers.LSTM(64, input_shape=(samples, X.shape[2]), return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(Y.shape[1]))

model.compile(loss='mse', optimizer='adam')

# Train the model
history = model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Make predictions
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

print(testPredict)

# Invert predictions back to actual numbers
trainPredict = scale.inverse_transform(trainPredict)
trainY = scale.inverse_transform(Y_train)
testPredict = scale.inverse_transform(testPredict)
testY = scale.inverse_transform(Y_test)
print(testPredict)


# Plot the predicted values against the actual values
plt.plot(trainY[:, 0])
plt.plot(trainPredict[:, 0])
plt.title('Train Predictions')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Actual', 'Predicted'], loc='upper left')
plt.show()

plt.plot(testY[:, 0])
plt.plot(testPredict[:, 0])
plt.title('Test Predictions')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Actual', 'Predicted'], loc='upper left')
plt.show()


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

# Calculate mean absolute error (MAE) for training and testing data
trainScore_mae = mean_absolute_error(trainY[:, 0], trainPredict[:, 0])
print('Train Score (MAE): %.2f' % (trainScore_mae))
testScore_mae = mean_absolute_error(testY[:, 0], testPredict[:, 0])
print('Test Score (MAE): %.2f' % (testScore_mae))

print(testPredict)

# Calculate mean absolute percentage error (MAPE) for training and testing data
trainScore_mape = mean_absolute_percentage_error(trainY[:, 0], trainPredict[:, 0])
print('Train Score (MAPE): %.2f%%' % (trainScore_mape*100))
testScore_mape = mean_absolute_percentage_error(testY[:, 0], testPredict[:, 0])
print('Test Score (MAPE): %.2f%%' % (testScore_mape*100))
