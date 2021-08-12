
import numpy as np
from functools import reduce
from tqdm import tqdm
from multiprocessing import Pool

class PointCloudKNeighborRegressor:

  def __init__(self, k = 10, num_proc=8):
    self.k = k
    self.num_proc = num_proc

  def fit(self, X, y):
    self.train_X = X
    self.train_y = y

  def predict(self, X):
    with Pool(self.num_proc) as pool:
      ret = list(tqdm(pool.imap(_Worker(self.train_X, self.train_y, self.k), X), total=len(X)))
      return np.array(ret)

class _Worker:

  def __init__(self, train_X, train_y, k):
    self.train_X = train_X
    self.train_y = train_y
    self.k = k
  
  def __call__(self, x):
    def _comparator(X_y):
      sum_dist = 0
      for x1 in x:
        min_dist = float('inf')
        for x2 in X_y[0]:
          dist = np.linalg.norm(x1 - x2)
          if dist < min_dist:
            min_dist = dist
        sum_dist += min_dist
      return sum_dist

    _sorted = sorted(zip(self.train_X, self.train_y), key=_comparator)

    return reduce(lambda sum, X_y: sum + X_y[1], _sorted[0:self.k], 0) / self.k


