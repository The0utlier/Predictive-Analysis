from sklearn.neural_network import MLPRegressor
# univariate data preparation
from numpy import array
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.layers import Dense
import pandas as pd
import time 
import yfinance as yf
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
# ... previous code ...

# define the scoring metric
#Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

raw_seq = yf.download(tickers='BTC-USD', period='10d', interval='90m')
# define input sequence
raw_seq = pd.Series(raw_seq['Close'])

start = time.time()
def sequential_prediction():
    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)

    model = MLPRegressor(hidden_layer_sizes=(25, ), max_iter=5000)

    # fit model
    model.fit(X, y) #2000
    #print(raw_seq[-n_steps:])
    x_input = array(raw_seq[-n_steps:])
    #print(x_input)
    x_input = x_input.reshape((1, n_steps))
    future = model.predict(x_input)
    y_pred = model.predict(X)
    #print(future)
    mae = mean_absolute_error(y.flatten(), y_pred.flatten())
    print(f'mae of mlp {mae}')

sequential_prediction()



# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
# choose a number of time steps
n_steps_in, n_steps_out = 3, 1
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=1000, verbose=0)
# demonstrate prediction
x_input = array(raw_seq[-n_steps_in:])
x_input = x_input.reshape((1, n_steps_in, n_features))
future = model.predict(x_input, verbose=0)
y_pred = model.predict(X)
mae = mean_absolute_error(y.flatten(), y_pred.flatten())
print(mae)

end = time.time()
print(end-start)
#print(future)
