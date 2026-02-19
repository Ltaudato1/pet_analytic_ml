import numpy as np
import pandas as pd

class MyKNNReg():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric if metric in ('euclidean', 'chebyshev', 'manhattan', 'cosine') else 'euclidean'
        self.weight = weight if weight in ('uniform', 'rank', 'distance') else 'uniform'
    
    def __str__(self):
        return f'MyKNNReg class: k={self.k}'
    
    def fit(self, X, y):
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y
        self.train_size = (len(X), len(X.T))
        
    def _calc_dist(self, x_1, x_2):
        if self.metric == 'euclidean':
            residual = x_1 - x_2
            return np.sqrt(residual @ residual.T)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(x_1 - x_2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x_1 - x_2))
        elif self.metric == 'cosine':
            norm1 = np.sqrt(x_1 @ x_1.T)
            norm2 = np.sqrt(x_2 @ x_2.T)
            
            if norm1 * norm2 == 0:
                return 1
            return 1 - (x_1 @ x_2.T) / (norm1 * norm2)
        
    def _calc_predict(self, D):
        if self.weight == 'uniform':
            return np.mean(np.array([y_j for d_j, y_j in D]))
        elif self.weight == 'rank':
            denom = np.sum(np.array([1 / (i + 1) for i, _ in enumerate(D)]))
            weights = np.array([1 / (denom * (i + 1)) for i, _ in enumerate(D)])
            return weights @ np.array(D)[:, 1].T
        elif self.weight == 'distance':
            denom = np.sum(np.array([1 / d for d, _ in D]))
            weights = np.array([1 / (denom * d) for d, _ in D])
            return weights @ np.array(D)[:, 1].T
            
        
    def predict(self, X):
        X_train = self.X
        y_train = self.y
        X_test = X.values if isinstance(X, pd.DataFrame) else X
        y_pred = [0] * len(X_train)
        
        for i, x_i in enumerate(X_test):
            D = [(self._calc_dist(x_i, x_j), y_j) for x_j, y_j in zip(X_train, y_train)]
            D = sorted(D, key=lambda x: x[0])[:self.k]
            y_pred[i] = self._calc_predict(D)
        return np.array(y_pred)