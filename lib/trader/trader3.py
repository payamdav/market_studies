import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace

property_type = SimpleNamespace(
  absolute=0,
  percent=1,
  delta=2,
)

exit_reasons = SimpleNamespace(
  noTrade=-1,
  unknown=0,
  sl=1,
  tp=2,
  limit=3,
  end=4,
)

col = SimpleNamespace(
  p=0,
  o=1,
  h=2,
  l=3,
  c=4,
  d=5,
  sl=6,
  tp=7,
  limit=8,
  entry=9,
  exit=10,
  reason=11,
  eprice=12,
  xprice=13,
  index=14,
)

class Trader:
  def __init__(self, **kw):
    defaults = {
      'o': None,
      'h': None,
      'l': None,
      'c': None,
      'p': None,

      'd': None,
      'tp': None,
      'sl': None,
      'limit': None,

      'sl_type': property_type.absolute,
      'tp_type': property_type.absolute,
      'ohlc': True,
    }

    self.__dict__.update(defaults | kw)
    self.n = 0
  
  def init(self):
    if self.c is None and self.p is None:
      raise Exception('p or c must be provided')
    if self.c is None:
      self.o = self.h = self.l = self.c = self.p
    if self.p is None:
      self.p = self.c.copy()
    if self.p in ['o', 'h', 'l', 'c']:
      self.p = self.__dict__[self.p]
    # if not self.ohlc:
    #   [delattr(self, k) for k in ['o', 'h', 'l', 'c']]
    self.n = len(self.p)

    if self.d is None:
      self.d = np.full(self.n, 0, dtype=int)
    elif isinstance(self.d, (int, float)):
      self.d = np.full(self.n, self.d, dtype=int)
    elif isinstance(self.d, str):
      self.d = np.full(self.n, 1 if self.d == 'long' else -1, dtype=int)

    if self.sl is None:
      self.sl = np.full(self.n, -1 * self.d * sys.float_info.max, dtype=float)
      self.sl_type = property_type.absolute
    elif isinstance(self.sl, (int, float)):
      self.sl = np.full(self.n, self.sl, dtype=float)
    if self.tp is None:
      self.tp = np.full(self.n, self.d * sys.float_info.max, dtype=float)
      self.tp_type = property_type.absolute
    elif isinstance(self.tp, (int, float)):
      self.tp = np.full(self.n, self.tp, dtype=float)
    if self.limit is None:
      self.limit = np.full(self.n, self.n, dtype=int)
    elif isinstance(self.limit, (int, float)):
      self.limit = np.full(self.n, self.limit, dtype=int)

    if self.sl_type == property_type.percent:
      self.sl = self.p - (self.d * (self.p * self.sl))
    elif self.sl_type == property_type.delta:
      self.sl = self.p - (self.d * self.sl)
    if self.tp_type == property_type.percent:
      self.tp = self.p + (self.d * (self.p * self.tp))
    elif self.tp_type == property_type.delta:
      self.tp = self.p + (self.d * self.tp)
    
    self.entry = np.full(self.n, -1, dtype=int)
    self.exit = np.full(self.n, -1, dtype=int)
    self.reason = np.full(self.n, exit_reasons.unknown, dtype=int)
    self.eprice = np.full(self.n, np.nan, dtype=float)
    self.xprice = np.full(self.n, np.nan, dtype=float)

    self.index = np.arange(self.n)

    # check that all shapes must be same
    if not all([len(x) == self.n for x in [self.d, self.sl, self.tp, self.limit, self.p]]):
      raise Exception('d, sl, tp, limit must have same length')
    if self.ohlc and not all([len(x) == self.n for x in [self.o, self.h, self.l, self.c]]):
      raise Exception('o, h, l, c must have same length')
    
    self.nd = np.vstack([self.p, self.o, self.h, self.l, self.c, self.d, self.sl, self.tp, self.limit, self.entry, self.exit, self.reason, self.eprice, self.xprice, self.index], dtype=float).T
    # print(self.nd)
    # print(self.nd.shape)

    return self

  def doTrade(self):
    self.nd[self.nd[:, col.d] == 0 ,col.reason] = exit_reasons.noTrade
    self.nd[self.nd[:, col.d] != 0 ,col.entry] = self.nd[self.nd[:, col.d] != 0 ,col.index]
    self.nd[self.nd[:, col.d] != 0 ,col.eprice] = self.nd[self.nd[:, col.d] != 0 ,col.p]

    for i in range(self.n):
      for j in range(i+1):
        if self.nd[j, col.reason] != exit_reasons.unknown:
          continue

        # check sl
        for refP in ([self.nd[i, col.o], self.nd[i, col.h], self.nd[i, col.l], self.nd[i, col.c]] if self.ohlc else [self.nd[i, col.p]]):
          if (self.nd[j, col.d] * (refP - self.nd[j, col.sl])) <= 0:
            self.nd[j, col.exit] = i
            self.nd[j, col.xprice] = refP
            self.nd[j, col.reason] = exit_reasons.sl
            break
        # check tp
        for refP in ([self.nd[i, col.o], self.nd[i, col.h], self.nd[i, col.l], self.nd[i, col.c]] if self.ohlc else [self.nd[i, col.p]]):
          if (self.nd[j, col.d] * (refP - self.nd[j, col.tp])) >= 0:
            self.nd[j, col.exit] = i
            self.nd[j, col.xprice] = refP
            self.nd[j, col.reason] = exit_reasons.tp
            break
        # check limit
        if self.nd[j, col.entry] + self.nd[j, col.limit] <= i:
          self.nd[j, col.exit] = i
          self.nd[j, col.xprice] = self.nd[i, col.c] if self.ohlc else self.nd[i, col.p]
          self.nd[j, col.reason] = exit_reasons.limit
    # check for unfinished trades
    for i in range(self.n):
      if self.nd[i, col.reason] == exit_reasons.unknown:
        self.nd[i, col.exit] = self.n - 1
        self.nd[i, col.xprice] = self.nd[-1, col.c] if self.ohlc else self.nd[-1, col.p]
        self.nd[i, col.reason] = exit_reasons.end
    return self
  
