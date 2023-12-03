import os
from lib.models.model_base import ModelBase
import lightgbm as lgb

class LightGBM(ModelBase):
  def training(self):
    self.params.update({
      'objective': getattr(self, 'objective', None),
      'learning_rate': getattr(self, 'learning_rate', None),
      'max_depth': getattr(self, 'max_depth', None),
      'num_class': getattr(self, 'num_class', None),
    })
    self.params = {k: v for k, v in self.params.items() if v is not None}

    self.dtrain = lgb.Dataset(self.features.X_train, label=self.features.y_train)
    if getattr(self.features, 'X_val', None) is not None:
      self.dval = lgb.Dataset(self.features.X_val, label=self.features.y_val, reference=self.dtrain)
      self.model = lgb.train(
        self.params, 
        self.dtrain, 
        getattr(self, 'num_boost_round', None) or 1000, 
        valid_sets=[self.dval], 
        callbacks=[
          lgb.early_stopping(stopping_rounds=getattr(self, 'stopping_rounds', None) or 1000), 
          lgb.log_evaluation(period=getattr(self, 'log_evaluation', None) or 50)
          ]
          )
    else:
      self.model = lgb.train(
        self.params, 
        self.dtrain, 
        getattr(self, 'num_boost_round', None) or 1000, 
        )
    return self
  
  def predict(self, X_test=None):
    dtest = X_test if X_test is not None else self.features.X_test
    self.y_pred = self.model.predict(dtest)
    return self
  
  def save(self):
    self.model.save_model(os.path.join(self.file_path, f'{self.__class__.__name__}_{self.name}_{self.pair}.txt'))
    return self
  
  def load(self):
    self.model = lgb.Booster(model_file=os.path.join(self.file_path, f'{self.__class__.__name__}_{self.name}_{self.pair}.txt'))
    return self
