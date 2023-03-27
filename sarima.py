import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# Download data and create dataframe
df = yf.download(tickers='BTC-USD', period='120d', interval='1d')
data = df.iloc[:-3]

# Define the parameter grid
p = range(0, 3)
d = range(1, 3)
q = range(0, 3)
P = range(0, 3)
D = range(1, 3)
Q = range(0, 3)
s = 7
param_grid = product(p, d, q, P, D, Q)

# Perform grid search
best_mae = np.inf
best_params = None
for params in param_grid:
    try:
        model = SARIMAX(data['Close'], order=(params[0], params[1], params[2]), 
                        seasonal_order=(params[3], params[4], params[5], s))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=3)
        actual = df['Close'][-3:]
        mae = mean_absolute_error(actual, forecast)
        if mae < best_mae:
            best_mae = mae
            best_params = params
    except:
        continue

# Print the best parameters and MAE
print("Best parameters:", best_params)
print("Best MAE:", best_mae)

# Plot results with the best model
model = SARIMAX(data['Close'], order=(best_params[0], best_params[1], best_params[2]), 
                seasonal_order=(best_params[3], best_params[4], best_params[5], s))
model_fit = model.fit()
forecast = model_fit.forecast(steps=3)
plt.plot(df.index[-100:], df['Close'][-100:], label='Actual')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()
