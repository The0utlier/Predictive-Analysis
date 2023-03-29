import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import time
import pandas_ta as ta
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer

# Load data
import warnings
warnings.filterwarnings('ignore')
#Split your data into training and testing sets before doing any feature scaling. This will prevent data leakage from the testing set into the training set.

df = yf.download(tickers='BTC-USD', period='10d', interval='90m')

df['abs_change'] = abs(df['Close'].diff())

# compute the mean of the absolute price change
mean_abs_change = df['abs_change'].mean()

print(f"The mean of the absolute change of the closing price is: {mean_abs_change:.2f}")


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

df['target'] = df['Close'].shift(-1)


df.dropna(inplace=True)


y = df['target']

X = df.drop(columns=['target','Adj Close'])


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Apply the same scaler to the testing set
X_test = scaler.transform(X_test)

# Define scorer functions for MAE and MAPE
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
def mape_scorer(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start_time = time.time()

# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create Random Forest model and GridSearchCV object
rf = RandomForestRegressor(random_state=10)
rf_grid = GridSearchCV(rf, param_grid, cv=5, refit='mae', n_jobs=-1)

# Fit GridSearchCV object to training data
rf_grid.fit(X_train, y_train)

# Get best hyperparameters and evaluate on test set
best_params = rf_grid.best_params_
rf_best = RandomForestRegressor(**best_params, random_state=10)

rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mape_scorer(y_test, y_pred)

print(f"Best hyperparameters: {best_params}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)


# Import XGBoost library
from xgboost import XGBRegressor

start_time = time.time()

# Define parameter grid for XGBoost hyperparameter tuning
xgb_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #'subsample': [0.5, 0.7, 0.9],
    #'colsample_bytree': [0.5, 0.7, 0.9],
    #'reg_alpha': [0.01, 0.1, 1],
    #'reg_lambda': [0.01, 0.1, 1],
    'gamma': [0, 0.1, 0.5, 1]
}

# Create XGBoost model and GridSearchCV object
xgb = XGBRegressor(random_state=10)
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, refit='mae', n_jobs=-1)

# Fit GridSearchCV object to training data
xgb_grid.fit(X_train, y_train)

# Get best hyperparameters and evaluate on test set
xgb_best_params = xgb_grid.best_params_
xgb_best = XGBRegressor(**xgb_best_params, random_state=10)
xgb_best.fit(X_train, y_train)

y_pred_xgb = xgb_best.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mape_xgb = mape_scorer(y_test, y_pred_xgb)

print(f"Best XGBoost hyperparameters: {xgb_best_params}")
print(f"XGBoost MAE: {mae_xgb}")
print(f"XGBoost MAPE: {mape_xgb}")

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)
