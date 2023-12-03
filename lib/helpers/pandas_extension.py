import pandas as pd
import os
from sklearn.model_selection import train_test_split
from lib.data.rates.rate_loader import RateLoader
import numpy as np


@pd.api.extensions.register_dataframe_accessor("libs")
class LibsAccessor:
  def __init__(self, pandas_obj):
    self._obj = pandas_obj
  
  def load(self, pair, tf='m'):
    return RateLoader(pair).minute_candles_data_frame().df if tf == 'm' else RateLoader(pair).hour_candles_data_frame().df if tf == 'h' else pd.DataFrame()

  def insert_previous(self, column_name, previous=1):
    idx = np.arange(len(self._obj))[:, None] + np.arange(1, previous + 1)[None, :]
    idx = np.clip(idx, 0, len(self._obj) - 1)
    temp = self._obj[column_name].to_numpy()[idx]
    temp[-1*(previous):] = np.nan
    return pd.concat([self._obj, pd.DataFrame(temp, columns=[f"{column_name}_B{i}" for i in range(1, previous + 1)])], axis=1)
  
  def apply_on_previous(self, column_name, previous, func, **kwargs):
    idx = np.arange(len(self._obj))[:, None] + np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self._obj) - 1)
    temp = self._obj[column_name].to_numpy()[idx]
    r = np.apply_along_axis(func, 1, temp, **kwargs)
    r[-1*(previous-1):] = np.nan
    return r
    # return pd.concat([self._obj, pd.DataFrame(r, columns=[f"{column_name}_{func.__name__}"])], axis=1)

  def insert_slope(self, column_name, previous=1, x_range=0.0001):
    idx = np.arange(len(self._obj))[:, None] + np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self._obj) - 1)
    y = self._obj[column_name].to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    # x = np.arange(previous, 0, -1)
    x = np.linspace(0, x_range, previous)[::-1]
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    return pd.concat([self._obj, pd.DataFrame(slopes, columns=[f"{column_name}_slope_{previous}"])], axis=1)

  def insert_sine_slope(self, column_name, previous=1, x_range=0.0001):
    idx = np.arange(len(self._obj))[:, None] + np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self._obj) - 1)
    y = self._obj[column_name].to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    # x = np.arange(previous, 0, -1)
    x = np.linspace(0, x_range, previous)[::-1]
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    sines = slopes / np.sqrt(1 + np.power(slopes, 2))
    return pd.concat([self._obj, pd.DataFrame(sines, columns=[f"{column_name}_sine_{previous}"])], axis=1)
  
  def insert_eucledian_distance(self, column_names):
    if isinstance(column_names, str):
      column_names = [column_names]
    if column_names is None:
      column_names = self._obj.columns
    temp = np.linalg.norm(self._obj[column_names], axis=1)
    tempdf = pd.DataFrame(temp)
    tempdf.columns = [f"eucledian_{'_'.join(column_names)}"]
    return pd.concat([self._obj, tempdf], axis=1)
  
  def insert_normalize_rows(self, column_names):
    if isinstance(column_names, str):
      column_names = [column_names]
    if column_names is None:
      column_names = self._obj.columns
    norm = np.linalg.norm(self._obj[column_names], axis=1)
    temp = pd.DataFrame(self._obj[column_names]/norm[:, None])
    temp.columns = [f"{column_name}_norm" for column_name in column_names]
    return pd.concat([self._obj, temp], axis=1)
  
  def resample_candles(self, freq):
    temp = self._obj.resample(freq, on='t').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum', 's': 'last', 'r': 'sum'})
    temp['t'] = temp.index
    temp = temp[::-1]
    temp = temp.reset_index(drop=True)
    return temp
