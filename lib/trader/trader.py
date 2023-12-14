import numpy as np
import pandas as pd
import os
import sys
from types import SimpleNamespace
from hypertrader import hyperTrader

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

    # check that all shapes must be same
    if not all([len(x) == self.n for x in [self.o, self.h, self.l, self.c, self.p, self.d, self.sl, self.tp, self.limit]]):
      raise Exception('d, sl, tp, limit must have same length')

    return self

  def doTrade(self):
    trade_indices = self.d.nonzero()[0]
    trade_count = len(trade_indices)
    entry = np.array(trade_indices, dtype=int, copy=True)
    d = self.d[entry].copy()
    sl = self.sl[entry].copy()
    tp = self.tp[entry].copy()
    limit = self.limit[entry].copy()

    reason = np.full(trade_count, exit_reasons.unknown, dtype=int)
    ex = np.full(trade_count, -1, dtype=int)
    eprice = np.full(trade_count, 0, dtype=float)
    xprice = np.full(trade_count, 0, dtype=float)

    hyperTrader(self.o.copy().data, self.h.copy().data, self.l.copy().data, self.c.copy().data, self.p.copy().data, d.data, entry.data, ex.data, reason.data, limit.data, tp.data, sl.data, eprice.data, xprice.data)
    self.reason = np.full(self.n, exit_reasons.noTrade, dtype=int)
    self.ex = np.full(self.n, -1, dtype=int)
    self.eprice = np.full(self.n, 0, dtype=float)
    self.xprice = np.full(self.n, 0, dtype=float)
    self.entry = np.full(self.n, -1, dtype=int)
    self.reason[entry] = reason
    self.ex[entry] = ex
    self.eprice[entry] = eprice
    self.xprice[entry] = xprice
    self.entry[entry] = entry

    return self
  
  def make_report(self):
    r = SimpleNamespace()
    r.count = self.d.nonzero()[0].size
    r.count_long = np.count_nonzero(self.d == 1)
    r.count_short = np.count_nonzero(self.d == -1)
    r.count_win = np.count_nonzero(self.reason == exit_reasons.tp)
    r.count_loss = np.count_nonzero(self.reason == exit_reasons.sl)
    r.count_limit = np.count_nonzero(self.reason == exit_reasons.limit)
    r.win_rate = r.count_win / r.count if r.count > 0 else 0
    r.loss_rate = r.count_loss / r.count if r.count > 0 else 0
    r.profit = np.sum(self.xprice - self.eprice)
    r.profit_long = np.sum(self.xprice[self.d == 1] - self.eprice[self.d == 1])
    r.profit_short = np.sum(self.xprice[self.d == -1] - self.eprice[self.d == -1])
    r.profit_win = np.sum(self.xprice[self.reason == exit_reasons.tp] - self.eprice[self.reason == exit_reasons.tp])
    r.profit_loss = np.sum(self.xprice[self.reason == exit_reasons.sl] - self.eprice[self.reason == exit_reasons.sl])
    r.min_duration = np.min(self.ex[self.d != 0] - self.entry[self.d != 0])
    r.max_duration = np.max(self.ex[self.d != 0] - self.entry[self.d != 0])
    r.avg_duration = np.mean(self.ex[self.d != 0] - self.entry[self.d != 0])
    self.report = r
    return self


  
