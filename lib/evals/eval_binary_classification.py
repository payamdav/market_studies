from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class EvalBinary:
  def __init__(self) -> None:
    pass

  def setParams(self, **kwargs):
    self.__dict__.update(kwargs)
    return self
  
  def predict(self):
    self.pred = self.model.predict(self.X_test).y_pred
    self.predictions = np.where(self.pred > self.binary_prob_threshold, 1, 0)
    return self
  
  def evaluate(self):
    self.accuracy = accuracy_score(self.y_test, self.predictions)
    self.precision = precision_score(self.y_test, self.predictions)
    self.recall = recall_score(self.y_test, self.predictions)
    self.f1 = f1_score(self.y_test, self.predictions)
    self.confusion_matrix = confusion_matrix(self.y_test, self.predictions)
    return self
  
  def print(self):
    print(f'Accuracy: {self.accuracy}')
    print(f'Precision: {self.precision}')
    print(f'Recall: {self.recall}')
    print(f'F1: {self.f1}')
    print(f'Confusion Matrix: {self.confusion_matrix}')
    return self
