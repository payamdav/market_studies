import os
from lib.models.model_base import ModelBase
import xgboost as xgb

class XGBoost(ModelBase):
  def training(self):
    self.params.update({
      'objective': getattr(self, 'objective', None),
      'eval_metric': getattr(self, 'eval_metric', None),
      'eta': getattr(self, 'eta', None),
      'max_depth': getattr(self, 'max_depth', None),
      'num_class': getattr(self, 'num_class', None),
    })
    self.params = {k: v for k, v in self.params.items() if v is not None}

    self.dtrain = xgb.DMatrix(self.features.X_train, label=self.features.y_train)
    # self.dtest = xgb.DMatrix(self.features.X_test, label=self.features.y_test)
    if getattr(self.features, 'X_val', None) is not None:
      self.dval = xgb.DMatrix(self.features.X_val, label=self.features.y_val)
      evals = [(self.dtrain, 'train'), (self.dval, 'eval')]
      self.model = xgb.train(
        params = self.params, 
        dtrain = self.dtrain,
        evals=evals,
        num_boost_round = getattr(self, 'num_boost_round', None) or 1000,
        verbose_eval = getattr(self, 'verbose_eval', None) or 100,
        early_stopping_rounds = getattr(self, 'early_stopping_rounds', None) or 100,
        )
    else:
      self.model = xgb.train(
        params = self.params, 
        dtrain = self.dtrain,
        num_boost_round = getattr(self, 'num_boost_round', None) or 1000,
        )
    return self
  
  def predict(self, X_test=None):
    dtest = xgb.DMatrix(X_test) if X_test is not None else xgb.DMatrix(self.features.X_test)
    self.y_pred = self.model.predict(dtest)
    return self
  
  def save(self):
    self.model.save_model(os.path.join(self.file_path, f'{self.__class__.__name__}_{self.name}_{self.pair}.ubj'))
    return self
  
  def load(self):
    self.model = xgb.Booster()  # init model
    self.model.load_model(os.path.join(self.file_path, f'{self.__class__.__name__}_{self.name}_{self.pair}.ubj'))  # load data
    return self
