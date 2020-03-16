# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:09:47 2020


"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import scipy.signal
import scipy.stats
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from numpy import mean, sqrt, square, arange
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler

def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    strain.append(np.quantile(X,0.01))
    strain.append(np.quantile(X,0.05))
    strain.append(np.quantile(X,0.95))
    strain.append(np.quantile(X,0.99))
    strain.append(np.abs(X).max())
    strain.append(np.abs(X).mean())
    strain.append(np.abs(X).std())
    return pd.Series(strain)

def catbostregtest(X_train, y_train):   
    # submission format
    submission = pd.read_csv('submission.csv', index_col='seg_id')
    X_test = pd.DataFrame()
    # prepare test data
    for seg_id in submission.index:
        seg = pd.read_csv('test/' + seg_id + '.csv')
        ch = gen_features(seg['acoustic_data'])
        X_test = X_test.append(ch, ignore_index=True)
    # model of choice here
    model = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)    #error line
    # write submission file LSTM
    submission['time_to_failure'] = y_hat
    submission.to_csv('submission.csv')
    print(model.best_score_)
def main(): 
      train1 = pd.read_csv('train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
      X_train = pd.DataFrame()
      y_train = pd.Series()
      for df in train1:
          ch = gen_features(df['acoustic_data'])
          X_train = X_train.append(ch, ignore_index=True)
          y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
      print(X_train.describe())
      catbostregtest(X_train, y_train)

if __name__ == '__main__':
    main()        
    