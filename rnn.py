#source https://github.com/SpencerPao/Data_Science/blob/main/RNN/RNN.ipynb
#https://www.youtube.com/watch?v=FBlPZJrJt9g

import yfinance as yf
# Standard Math / Data libraries
import numpy as np
import pandas as pd

# Plotting package
import matplotlib.pyplot as plt
# Scaling Package
from sklearn.preprocessing import MinMaxScaler

# Keras Network @ https://www.tensorflow.org/guide/keras/rnn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set Random seed
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas_ta as ta
import numpy as np
import pandas as pd

# Set the random seed to ensure reproducibility
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

#plt.plot(returns)
#returns.hist()

# Flatten this matrix down.
npa = returns.values[1:].reshape(-1,1) # Python is smart to recognize whatever dimension you need by using this parameter
print(len(npa))
# # Let's scale the data -- this helps avoid the exploding gradient issue
scale = MinMaxScaler(feature_range=(0,1)) # This is by default.
npa = scale.fit_transform(npa)
#print(len(npa))


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


print(X, Y)
#apply featrues


#Reshape the data so that the inputs will be acceptable to the model.
X = np.array(X)
Y = np.array(Y)
print('Dimensions of X', X.shape, 'Dimensions of Y', Y.shape)



# # Get the training and testing set
threshold = round(0.9 * X.shape[0])
trainX, trainY = X[:threshold], Y[:threshold]
testX, testY =  X[threshold:], Y[threshold:]
print('Training Length',trainX.shape, trainY.shape,'Testing Length:',testX.shape, testY.shape)

# Let's build the RNN
model = keras.Sequential()

# Add a RNN layer with 30 internal units.
model.add(layers.SimpleRNN(30,
                           activation = 'tanh',
                           use_bias=True,
                           input_shape=(trainX.shape[1], trainX.shape[2])))
# Add a dropout layer (penalizing more complex models) -- prevents overfitting
model.add(layers.Dropout(rate=0.2))


# Add a Dense layer with 1 units (Since we are doing a regression task.
model.add(layers.Dense(1))

# Evaluating loss function of MSE using the adam optimizer.
model.compile(loss='mean_squared_error', optimizer = 'adam')

# Print out architecture.
model.summary()

# Fitting the data
history = model.fit(trainX,
                    trainY,
                    shuffle = False, # Since this is time series data
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1) # Verbose outputs data

# Plotting the loss iteration
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label ='validation loss')
plt.legend()

plt.show()
# Note:
# if training loss >> validation loss -> Underfitting
# if training loss << validation loss -> Overfitting (i.e model is smart enough to have mapped the entire dataset..)
# Several ways to address overfitting:
# Reduce complexity of model (hidden layers, neurons, parameters input etc)
# Add dropout and tune rate
# More data :)

# This is a one step forecast (based on how we constructed our model)
# Get the predicted values

y_pred = model.predict(testX)
y_pred = scale.inverse_transform(y_pred)

testY = scale.inverse_transform(testY)

# Evaluate the performance of the model
mse = np.mean(np.square(testY - y_pred))
print("Mean Squared Error:", mse)

print(y_pred)


# This is a one step forecast (based on how we constructed our model)
plt.plot(testY, label = 'True Value')
plt.plot(y_pred, label = 'Forecasted Value')
plt.legend()
plt.show()
