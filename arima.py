import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from itertools import product
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Download data and create dataframe
df = yf.download(tickers='BTC-USD', period='120d', interval='1d')
data = df.iloc[:-7]

# Define parameter grid
p = range(1, 25)
d = range(1,3)
q = range(1, 25)
pdq = list(product(p, d, q))

# Fit and evaluate models for each parameter combination
mae_dict = {}
for param in pdq:
    try:
        model = ARIMA(data['Close'], order=param)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        actual = df['Close'][-7:]
        mae = mean_absolute_error(actual, forecast)
        mae_dict[param] = mae
        print('ARIMA{} - MAE: {}'.format(param, mae))
    except:
        continue

# Find parameter combination with lowest MAE
best_params = min(mae_dict, key=mae_dict.get)
print('Best parameters: ', best_params)

# Split data
train_data = df.iloc[:-7]
test_data = df.iloc[-7:]

model = ARIMA(data['Close'], order=(best_params))#exog=data.drop(columns=['Close']))
model_fit = model.fit()

# forecast next 7 days
forecast = model_fit.forecast(steps=7)#,exog=df.iloc[-7:].drop(columns=['Close']))

print(forecast)

actual = df['Close'][-7:]

# calculate MAE
mae = mean_absolute_error(actual, forecast)

print("MAE:", mae)

actual = df['Close'][-30:]

# Plot data and predictions
plt.plot(actual.index, actual, label='Actual')
plt.plot(forecast.index, forecast, label='Predicted')
plt.legend()
plt.show()
