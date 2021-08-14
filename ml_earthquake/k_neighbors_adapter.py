import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNeighborsAdapter:

    def __init__(self, k = 100):
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, X, y):
        self.model.fit(self._reshapeX(X), y)

    def predict(self, X):
        return np.delete(self.model.predict_proba(self._reshapeX(X)), 1, 1).reshape(-1)

    def _reshapeX(self, X):
        return X.reshape([X.shape[0], -1])
