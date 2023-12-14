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
    if not self.ohlc:
      [delattr(self, k) for k in ['o', 'h', 'l', 'c']]
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

    return self

  def doTrade(self):
    tmask = self.d != 0
    ntmask = self.d == 0
    lmask = self.d == 1
    smask = self.d == -1

    self.reason[ntmask] = exit_reasons.noTrade
    self.entry[tmask] = self.index[tmask]
    self.eprice[tmask] = self.p[tmask]

    for i in range(self.n - 1, -1, -1):
      slicer = slice(-1, i+1, -1)
      # check sl
      for refP in ([self.o[i], self.h[i], self.l[i], self.c[i]] if self.ohlc else [self.p[i]]):
        indices = (self.index[slicer])[(self.reason[slicer] == exit_reasons.unknown) & ((self.d[slicer] * (refP - self.sl[slicer])) <= 0)]
        if len(indices) > 0:
          self.exit[indices] = i
          self.xprice[indices] = refP
          self.reason[indices] = exit_reasons.sl
      # check tp
      for refP in ([self.o[i], self.h[i], self.l[i], self.c[i]] if self.ohlc else [self.p[i]]):
        indices = (self.index[slicer])[(self.reason[slicer] == exit_reasons.unknown) & ((self.d[slicer] * (refP - self.tp[slicer])) >= 0)]
        if len(indices) > 0:
          self.exit[indices] = i
          self.xprice[indices] = refP
          self.reason[indices] = exit_reasons.tp
      # check limit
      indices = (self.index[slicer])[(self.reason[slicer] == exit_reasons.unknown) & (self.entry[slicer] - self.limit[slicer] >= i)]
      if len(indices) > 0:
        self.exit[indices] = i
        self.xprice[indices] = self.c[i] if self.ohlc else self.p[i]
        self.reason[indices] = exit_reasons.limit
    
    # check for unfinished trades
    indices = self.index[(self.reason == exit_reasons.unknown)]
    if len(indices) > 0:
      self.exit[indices] = 0
      self.xprice[indices] = self.c[0] if self.ohlc else self.p[0]
      self.reason[indices] = exit_reasons.end
    
    return self
  
