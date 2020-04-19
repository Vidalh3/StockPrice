import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

print("To Exit Press (Q)")

# Get Stock with input
dt = input("What Stock Do You Want:").upper()
df = yf.Ticker(dt)

# Download Data for 5 years
df = yf.download(dt, period='3mo')
# print(df)


# Save Stock Data
df_save = pd.DataFrame(df)
df_save.to_csv("StockData.csv")
df_save = pd.read_csv("StockData.csv")
print(df_save.dtypes)
print("##############################################\n")

# Create List X and Y
future_dates = 30
#close_data = []

# Get all rows from date columns
df_close = df_save.loc[:, 'Adj Close']
print(df_close, '\n')

# Create the dependent data set 'y' as prices
# for close_price in df_close:
#    close_data.append(close_price)

# Save Adj Close Data
save_adj = pd.DataFrame(df_close)
save_adj.to_csv("AdjCloseData.csv")
save_adj = pd.read_csv("AdjCloseData.csv")

# Column 'the target or dependent variable' shift 'n' units up
df_close['Prediction'] = save_adj.shift(-future_dates)
# Save Adj Close Data
save_adj = pd.DataFrame(df_close)
save_adj.to_csv("AdjCloseData.csv")
print(df_close, '\n')

# Create the independent data set (X)
# Convert dataframe to numpy array
X = np.array(df_close.drop(['Prediction']))
# Remove the last 'n' rows
X = X[:-future_dates]
X.reshape(-1, 1)
print(X, '\n')

# Crate dependent data set (y)
# Convert dataframe to numpy array (All values including NAN's)
y = np.array(df_close['Prediction'])
# Get All data except for last row
y = y[:-future_dates]
y.reshape(-1, 1)
print(y)

print('#############################################################\n')

# Split data into 80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the 3 support vector models
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_lin.fit(x_train, y_train)
svr_poly.fit(x_train, y_train)
svr_rbf.fit(x_train, y_train)


# Testing the Models : Score Returns
# The Best Score is 1.0
svm_confidence_l = svr_lin.score(x_test, y_test)
svm_confidence_p = svr_poly.score(x_test, y_test)
svm_confidence_r = svr_rbf.score(x_test, y_test)
print('SVM Confidence: ', svm_confidence_l, '\n')
print('SVM Confidence: ', svm_confidence_p, '\n')
print('SVM Confidence: ', svm_confidence_r)
