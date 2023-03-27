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
data = df.iloc[:-7]

# Define parameter grid
p = range(0, 5)
d = range(0, 5)
q = range(0, 5)
P = range(0, 3)
D = range(0, 3)
Q = range(0, 3)
s = 7
pdq = list(product(p, d, q))
seasonal_pdq = list(product(P, D, Q, [s]))

# Fit and evaluate models for each parameter combination
mae_dict = {}
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(data['Close'], order=param, seasonal_order=param_seasonal)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=7)
            actual = df['Close'][-7:]
            mae = mean_absolute_error(actual, forecast)
            mae_dict[(param, param_seasonal)] = mae
            print('SARIMA{}x{} - MAE: {}'.format(param, param_seasonal, mae))
        except:
            continue

# Find parameter combination with lowest MAE
best_params = min(mae_dict, key=mae_dict.get)
print('Best parameters: ', best_params)

# Split data
train_data = df.iloc[:-7]
test_data = df.iloc[-7:]

model = SARIMAX(data['Close'], order=best_params[0], seasonal_order=best_params[1])
model_fit = model.fit()

# forecast next 7 days
forecast = model_fit.forecast(steps=7)

actual = df['Close'][-7:]

# calculate MAE
mae = mean_absolute_error(actual, forecast)

print("MAE:", mae)

# Plot results
import matplotlib.pyplot as plt

plt.plot(df.index[-10:], df['Close'][-10:], label='Actual')
plt.plot(pd.date_range(start=df.index[-8], periods=8, freq='D'), forecast, label='Predicted')
plt.legend()
plt.show()
