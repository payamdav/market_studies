import os
import pandas as pd
from sklearn.model_selection import train_test_split
from lib.data.rates.rate_loader import RateLoader
import numpy as np


class Features:
  def __init__(self, name, pair, auto_load=True) -> None:
    self.name = name or ""
    self.pair = pair or ""
    self.df = RateLoader(self.pair).minute_candles_data_frame().df if (len(self.pair) > 0 and auto_load) else pd.DataFrame()
    self.features_to_hide = []
    self.label_column_name = 'label'
    self.file_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "features")

  def setParams(self, **kwargs):
    self.__dict__.update(kwargs)
    return self

  def build(self):
    return self
  
  def finalize(self):
    self.features = self.df.dropna().drop([self.label_column_name, *(self.features_to_hide)], axis=1)
    self.label = (self.df.dropna())[self.label_column_name]
    self.features = self.features.reset_index(drop=True)
    self.label = self.label.reset_index(drop=True)
    return self

  def data_split(self):
    self.X_test, self.X_train, self.y_test, self.y_train = train_test_split(self.features, self.label, test_size=(1 - self.split_test_size), shuffle=False)
    if getattr(self, 'split_validation_size', 0) > 0:
      self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.split_validation_size)
    return self

  def save(self):
    self.df.to_feather(os.path.join(self.file_path, f'{self.__class__.__name__}_{self.name}_{self.pair}.feather'))
    return self
  
  def load(self):
    self.df = pd.read_feather(os.path.join(self.file_path, f'{self.__class__.__name__}_{self.name}_{self.pair}.feather'))
    return self
  
  def insert_previous(self, column_name, previous=1):
    idx = np.arange(len(self.df))[:, None] + np.arange(1, previous + 1)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    temp = self.df[column_name].to_numpy()[idx]
    temp[-1*(previous):] = np.nan
    self.df = pd.concat([self.df, pd.DataFrame(temp, columns=[f"{column_name}_B{i}" for i in range(1, previous + 1)])], axis=1)
    return self
  
  def apply_on_previous(self, column_name, previous, func, **kwargs):
    idx = np.arange(len(self.df))[:, None] + np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    temp = self.df[column_name].to_numpy()[idx]
    r = np.apply_along_axis(func, 1, temp, **kwargs)
    r[-1*(previous-1):] = np.nan
    self.df = pd.concat([self.df, pd.DataFrame(r, columns=[f"{column_name}_{func.__name__}"])], axis=1)
    return self
  
  def insert_slope(self, column_name, previous=1, x_range=0.0001):
    idx = np.arange(len(self.df))[:, None] + np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    y = self.df[column_name].to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    # x = np.arange(previous, 0, -1)
    x = np.linspace(0, x_range, previous)[::-1]
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    self.df = pd.concat([self.df, pd.DataFrame(slopes, columns=[f"{column_name}_slope_{previous}"])], axis=1)
    return self

  def insert_sine_slope(self, column_name, previous=1, x_range=0.0001):
    idx = np.arange(len(self.df))[:, None] + np.arange(0, previous)[None, :]
    idx = np.clip(idx, 0, len(self.df) - 1)
    y = self.df[column_name].to_numpy()[idx]
    Y = y - y.mean(axis=1)[:, None]
    # x = np.arange(previous, 0, -1)
    x = np.linspace(0, x_range, previous)[::-1]
    X = x - x.mean()
    slopes = np.dot(Y, X) / np.dot(X, X )
    sines = slopes / np.sqrt(1 + np.power(slopes, 2))
    self.df = pd.concat([self.df, pd.DataFrame(sines, columns=[f"{column_name}_sine_{previous}"])], axis=1)
    return self
  
  def insert_normalize_rows(self, column_names):
    if isinstance(column_names, str):
      column_names = [column_names]
    if column_names is None:
      column_names = self.df.columns
    norm = np.linalg.norm(self.df[column_names], axis=1)
    temp = pd.DataFrame(self.df[column_names]/norm[:, None])
    temp.columns = [f"{column_name}_norm" for column_name in column_names]
    self.df = pd.concat([self.df, temp], axis=1)
    return self
  
  def insert_eucledian_distance(self, column_names):
    if isinstance(column_names, str):
      column_names = [column_names]
    if column_names is None:
      column_names = self.df.columns
    temp = np.linalg.norm(self.df[column_names], axis=1)
    tempdf = pd.DataFrame(temp)
    tempdf.columns = [f"eucledian_{'_'.join(column_names)}"]
    self.df = pd.concat([self.df, tempdf], axis=1)
    return self
  
  def resample_candles(self, freq):
    self.df = self.df.resample(freq, on='t').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum', 's': 'last', 'r': 'sum'})
    self.df['t'] = self.df.index
    self.df = self.df[::-1]
    self.df = self.df.reset_index(drop=True)
    return self
  


