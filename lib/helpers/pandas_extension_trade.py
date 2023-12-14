import numpy as np
import pandas as pd
import os
from lib.trader.trader import Trader


@pd.api.extensions.register_dataframe_accessor("libTrade")
class LibTradeAccessor:
  def __init__(self, pandas_obj):
    self.df = pandas_obj
  
  def trade(self, **kw):
    defaults = {
      'o': self.df.o.to_numpy()[::-1],
      'h': self.df.h.to_numpy()[::-1],
      'l': self.df.l.to_numpy()[::-1],
      'c': self.df.c.to_numpy()[::-1],
      'p': 'c',
    }
    t = Trader(**(defaults | kw))
    return t
