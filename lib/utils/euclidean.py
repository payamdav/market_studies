import numpy as np

def euclidean_distance2(X, Y):
  sx = np.sum(X**2, axis=1, keepdims=True)
  sy = np.sum(Y**2, axis=1, keepdims=True)
  return np.sqrt(-2 * X.dot(Y.T) + sx + sy.T)

def euclidean_distance(X, Y):
  return np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1))
