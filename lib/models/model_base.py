import os

class ModelBase:
  def __init__(self, name, pair, auto_save=True) -> None:
    self.name = name or ""
    self.pair = pair or ""
    self.auto_save = auto_save
    self.params = {}
    self.file_path = os.path.join(os.path.dirname(__file__), "..", "..", "files", "models")

  def setParams(self, **kwargs):
    self.__dict__.update(kwargs)
    return self

  def train(self):
    self.training()
    if self.auto_save:
      self.save()
    return self
  
  def training(self):
    return self

  def predict(self):
    return self
  
  def save(self):
    return self
  
  def load(self):
    return self
  
  