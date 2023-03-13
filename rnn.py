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

returns = df['Adj Close'].pct_change() # Used for univariate example.

column_names = df.columns
x = df.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Flatten this matrix down.
npa = returns.values[1:].reshape(-1,1) # Python is smart to recognize whatever dimension you need by using this parameter
print(len(npa))
# # Let's scale the data -- this helps avoid the exploding gradient issue
scale = MinMaxScaler(feature_range=(0,1)) # This is by default.
npa = scale.fit_transform(npa)
print(len(npa))

# Need the data to be in the form [sample, time steps, features (dimension of each element)]
# Need the data to be in the form [sample, time steps, features (dimension of each element)]
samples = 10 # Number of samples (in past)
steps = 1 # Number of steps (in future)
X = [] # X array
Y = [] # Y array
for i in range(df.shape[0] - samples):
    X.append(df.iloc[i:i+samples, 0:5].values) # Independent Samples
    Y.append(df.iloc[i+samples, 5:].values) # Dependent Samples
print('Training Data: Length is ',len(X[0:1][0]),': ', X[0:1])
print('Testing Data: Length is ', len(Y[0:1]),': ', Y[0:1])

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
# Fitting the data
history = model.fit(trainX,
                    trainY,
                    shuffle = False, # Since this is time series data
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1) # Verbose outputs data

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

'''
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

'''
