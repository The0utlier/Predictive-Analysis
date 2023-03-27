import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time
import pandas_ta as ta
from sklearn.metrics import f1_score, r2_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import tree
from sklearn import metrics
import yfinance as yf
 
import warnings
warnings.filterwarnings('ignore')
#Split your data into training and testing sets before doing any feature scaling. This will prevent data leakage from the testing set into the training set.

df = yf.download(tickers='BTC-USD', period='10d', interval='90m')#cv

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

df.dropna(inplace=True)

df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# Initialize lists to store input and output samples

target = df['target']
features = df.drop(columns=['target','Adj Close'])


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 36)

print(len(y_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Apply the same scaler to the testing set
X_test = scaler.transform(X_test)


'''

models = [LogisticRegression(penalty='l2', solver='liblinear',max_iter=10000), SVC(kernel='poly', probability=True,C=0.1), XGBClassifier(reg_alpha=0.1, reg_lambda=0.1), RandomForestClassifier(n_estimators=1500, random_state=25,max_depth=10, min_samples_split=5),tree.DecisionTreeClassifier(max_depth=5),MLPClassifier(solver='adam', alpha=0.0001, max_iter=10000, 
              hidden_layer_sizes=(10,5,2), random_state=1)]
 
for i in range(6):
  models[i].fit(X_train, y_train)
 
  print(f'{models[i]} : ')
  print('Validation Accuracy : ', metrics.roc_auc_score(y_test, models[i].predict_proba(X_test)[:,1]))
  print()

print(len(X_test))
'''

from sklearn.model_selection import GridSearchCV

start_time = time.time()

lr_model = LogisticRegression(penalty='l2', max_iter=10000)

# Define the hyperparameters to search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Create the GridSearchCV object
lr_grid = GridSearchCV(lr_model, param_grid, cv=3, scoring='roc_auc')

# Fit the GridSearchCV object to the training data
lr_grid.fit(X_train, y_train)
# Print the best hyperparameters and the validation score
# your code here

best_lr_model = LogisticRegression(penalty='l2', C=lr_grid.best_params_['C'], solver=lr_grid.best_params_['solver'], max_iter=10000)

# Fit the model to the training data
best_lr_model.fit(X_train, y_train)

lr_model.fit(X_train, y_train)

# Use the model to make predictions on new data
y_pred = lr_model.predict(X_test)

from sklearn.metrics import accuracy_score


y_pred = best_lr_model.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred)


print(f'Accuracy {accuracy2}')

# Calculate the F1 score
f1_lr_score = f1_score(y_test, y_pred)
print(f'f1 score {f1_lr_score}')

end_time = time.time()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)
#print(cf_matrix)

import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

plt.show()
labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.show()

print("Logistic Regression")
print(f"Time taken: {end_time - start_time} seconds")
print("Best hyperparameters: ", lr_grid.best_params_)




start_time = time.time()

svm_model = SVC(kernel='poly', probability=True)

# Define the hyperparameters to search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'degree': [2, 3, 4, 5],
}

# Create the GridSearchCV object
svm_grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='roc_auc')

# Fit the GridSearchCV object to the training data
svm_grid.fit(X_train, y_train)

best_svm_model = SVC(kernel='poly', probability=True, C=svm_grid.best_params_['C'], gamma=svm_grid.best_params_['gamma'], degree = svm_grid.best_params_['degree'], max_iter=10000)

# Fit the model to the training data
best_svm_model.fit(X_train, y_train)

# Use the model to make predictions on new data
y_pred = best_svm_model.predict(X_test)

end_time = time.time()

accuracy = accuracy_score(y_pred, y_test)

print(f'Accuracy {accuracy}')

f1_score_svc = f1_score(y_test, y_pred)
print(f'f1 score {f1_score_svc}')

# Print the best hyperparameters and the validation score
print("Support Vector Machine")

print(f"Time taken: {end_time - start_time} seconds")
print("Best hyperparameters: ", svm_grid.best_params_)

cf_matrix = confusion_matrix(y_test, y_pred)
import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

plt.show()



start_time = time.time()

xgb_model = XGBClassifier()

# Define the hyperparameters to search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 0.8],
    'max_depth': [5, 7, 9, 12],
    'n_estimators': [100, 300,500,900,1200]
}

# Create the GridSearchCV object
xgb_grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc')

# Fit the GridSearchCV object to the training data
xgb_grid.fit(X_train, y_train)

best_xgb_model = XGBClassifier(learning_rate=xgb_grid.best_params_['learning_rate'], max_depth=xgb_grid.best_params_['max_depth'], n_estimators = xgb_grid.best_params_['n_estimators'])

# Fit the model to the training data
best_xgb_model.fit(X_train, y_train)

# Use the model to make predictions on new data
y_pred = best_xgb_model.predict(X_test)

end_time = time.time()

accuracy = accuracy_score(y_pred, y_test)

print(f'Accuracy {accuracy}')

f1_xgb_score = f1_score(y_test, y_pred)
print(f'f1 score {f1_xgb_score}')

# Print the best hyperparameters and the validation score
print("XGBoost")

print(f"Time taken: {end_time - start_time} seconds")
print("Best hyperparameters: ", xgb_grid.best_params_)

cf_matrix = confusion_matrix(y_test, y_pred)
#print(cf_matrix)

import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

plt.show()

start_time = time.time()

rf_model = RandomForestClassifier(random_state=2022)

# Define the hyperparameters to search
param_grid = {
    'n_estimators': [100, 300, 600, 900, 1200],
    'max_features': ['sqrt', 'log2', 0.5]
    #'max_depth': [5, 7, 9, 15],
}

# Create the GridSearchCV object
rf_grid = GridSearchCV(rf_model, param_grid, cv=3, scoring='roc_auc')

# Fit the GridSearchCV object to the training data
rf_grid.fit(X_train, y_train)

best_rf_model = RandomForestClassifier(n_estimators=rf_grid.best_params_['n_estimators'], max_features=rf_grid.best_params_['max_features'])#,max_depth=rf_grid.best_params_['math_depth'], max_iter=10000)

# Fit the model to the training data
best_rf_model.fit(X_train, y_train)

# Use the model to make predictions on new data
y_pred = best_rf_model.predict(X_test)

end_time = time.time()
f1_rf_score = f1_score(y_test, y_pred)
print(f'f1 score {f1_rf_score}')
accuracy = accuracy_score(y_pred, y_test)

print(f'Accuracy {accuracy}')


# Print the best hyperparameters and the validation score
print("Random Forest")

print(f"Time taken: {end_time - start_time} seconds")
print("Best hyperparameters: ", rf_grid.best_params_)
print("Validation score: ", rf_grid.best_score_)

cf_matrix = confusion_matrix(y_test, y_pred)
#print(cf_matrix)

import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

plt.show()

start_time = time.time()

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}


# Create a decision tree classifier
dtc = tree.DecisionTreeClassifier()

# Create a grid search object
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=3)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and its score
best_dtc = grid_search.best_estimator_
best_score = grid_search.best_score_

# Evaluate the best estimator on the test data
test_score = best_dtc.score(X_test, y_test)

end_time = time.time()


# Print the best hyperparameters and scores
print(f"Decision tree")
print(f"Time taken: {end_time - start_time} seconds")
print("Best parameters: ", grid_search.best_params_)
print("Training score: ", best_score)
print("Test score: ", test_score)

start_time = time.time()
# Define the hyperparameters to tune
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (40,)],
    'alpha': [1e-5, 1e-4, 1e-3],
    'max_iter': [1000, 5000, 10000]
}

# Create an MLP classifier
mlp = MLPClassifier(solver='adam', random_state=1)

# Create a grid search object
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and its score
best_mlp = grid_search.best_estimator_
best_score = grid_search.best_score_

# Evaluate the best estimator on the test data
test_score = best_mlp.score(X_test, y_test)

end_time = time.time()


# Print the best hyperparameters and scores
print(f"MLP")
print(f"Time taken: {end_time - start_time} seconds")
print("Best parameters: ", grid_search.best_params_)
print("Training score: ", best_score)
print("Test score: ", test_score)

