import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Download data and create dataframe
df = yf.download(tickers='BTC-USD', period='30d', interval='1d')
data = df.iloc[:-3]

# Fit model
model = ExponentialSmoothing(data['Close'], trend='add', seasonal='add', seasonal_periods=3).fit()

# forecast next 7 days
forecast = model.forecast(steps=3)

actual = df['Close'][-3:]

# calculate MAE
mae = mean_absolute_error(actual, forecast)

print("MAE:", mae)

# Plot results
import matplotlib.pyplot as plt

plt.plot(df.index[-100:], df['Close'][-100:], label='Actual')
plt.plot(forecast)
plt.legend()
plt.show()
