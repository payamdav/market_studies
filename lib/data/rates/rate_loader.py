import os
import json
import numpy as np
import pandas as pd

class RateLoader:
  def __init__(self, pair):
    self.pair = pair
    self.file_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "files", "rates")
  
  def minute_candles_numpy(self):
    with open(os.path.join(self.file_path, self.pair + '_m.json')) as f:
      data = json.load(f)
      c = data['candles']
      self.t = np.array([s['t'] for s in c])
      self.o = np.array([s['o'] for s in c])
      self.h = np.array([s['h'] for s in c])
      self.l = np.array([s['l'] for s in c])
      self.c = np.array([s['c'] for s in c])
      self.v = np.array([s['v'] for s in c])
      self.s = np.array([s['s'] for s in c])
      self.r = np.array([s['r'] for s in c])
    return self
  
  def minute_candles_data_frame(self):
    with open(os.path.join(self.file_path, self.pair + '_m.json')) as f:
      data = json.load(f)
      self.df = pd.DataFrame(data['candles'])
      self.df.drop(columns=['m'], inplace=True)
      self.df.rename(columns={'t': 'ts'}, inplace=True)
      self.df['t'] = pd.to_datetime(self.df['ts'], unit='s')
    return self


  



