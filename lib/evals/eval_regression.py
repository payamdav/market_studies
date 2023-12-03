import numpy as np

class EvalRegression:
  def __init__(self) -> None:
    pass

  def setParams(self, **kwargs):
    self.__dict__.update(kwargs)
    return self
  
  def predict(self):
    self.pred = self.model.predict(self.X_test).y_pred
    return self
  
  def evaluate(self):
    self.mse = np.mean((self.y_test - self.pred)**2)
    self.rmse = np.sqrt(self.mse)
    self.mae = np.mean(np.abs(self.y_test - self.pred))
    self.mape = np.mean(np.abs(self.y_test - self.pred) / self.y_test) * 100
    return self
  
  def print(self):
    print(f'MSE: {self.mse}')
    print(f'RMSE: {self.rmse}')
    print(f'MAE: {self.mae}')
    print(f'MAPE: {self.mape}')
    return self

