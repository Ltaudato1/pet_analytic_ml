import numpy as np
import pandas as pd

class MyKNNClf():
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric if metric in ('euclidean', 'chebyshev', 'manhattan', 'cosine') else 'euclidean'
        self.weight = weight if weight in ('uniform', 'rank', 'distance') else 'uniform'
    
    def __str__(self):
        return f'MyKNNClf class: k={self.k}'
    
    def _calc_metric(self, x_1, x_2):
        if self.metric == 'euclidean':
            residual = x_1 - x_2
            return np.sqrt(residual @ residual.T)
        elif self.metric == 'chebyshev':
            return np.max(np.abs(x_1 - x_2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x_1 - x_2))
        elif self.metric == 'cosine':
            norm1 = np.sqrt(np.sum(x_1**2))
            norm2 = np.sqrt(np.sum(x_2**2))
            if norm1 == 0 or norm2 == 0:
                return 1
            return 1 - (x_1 @ x_2.T) / (norm1 * norm2)
            
    
    def fit(self, X, y):
        self.train_size = (len(X), len(X.T))
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y
    
    def predict(self, X):
        X_test = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_train = self.X
        y_train = self.y
        
        y_pred = []
        for x_pred in X_test:
            D = [0] * self.train_size[0]
            for i, (x_i, y_i) in enumerate(zip(X_train, y_train)):
                D[i] = (self._calc_metric(x_pred, x_i), y_i)
            D = sorted(D, key=lambda x: x[0])[:self.k]
            if self.weight == 'uniform':
                positives = np.sum(np.array([y_i for x_i, y_i in D]))
                negatives = np.sum(np.array([1 - y_i for x_i, y_i in D]))
                y_pred.append(1 if positives >= negatives else 0)
            elif self.weight == 'rank':
                positives_idx = np.array([i for i, (x_i, y_i) in enumerate(D) if y_i == 1])
                negatives_idx = np.array([i for i, (x_i, y_i) in enumerate(D) if y_i == 0])
                Q_0 = np.sum(np.array([1 / (i + 1) for i in negatives_idx]))
                Q_1 = np.sum(np.array([1 / (i + 1) for i in positives_idx]))
                
                y_pred.append(1 if Q_1 >= Q_0 else 0)
            elif self.weight == 'distance':
                positives_dist = np.array([x_i for x_i, y_i in D if y_i == 1])
                negatives_dist = np.array([x_i for x_i, y_i in D if y_i == 0])
                Q_0 = np.sum(np.array([1 / d for d in negatives_dist]))
                Q_1 = np.sum(np.array([1 / d for d in positives_dist]))
                
                y_pred.append(1 if Q_1 >= Q_0 else 0)
        return np.array(y_pred)
    
    def predict_proba(self, X):
        X_test = X.values if isinstance(X, pd.DataFrame) else X
        X_train = self.X
        y_train = self.y
        
        y_pred = []
        for x_pred in X_test:
            D = [0] * self.train_size[0]
            for i, (x_i, y_i) in enumerate(zip(X_train, y_train)):
                D[i] = (self._calc_metric(x_pred, x_i), y_i)
            D = sorted(D, key=lambda x: x[0])[:self.k]
            if self.weight == 'uniform':
                positives = np.sum(np.array([y_i for x_i, y_i in D]))
                y_pred.append(positives / self.k)
            elif self.weight == 'rank':
                positives_idx = np.array([i for i, (_, y_i) in enumerate(D) if y_i == 1])
                denom = np.sum(np.array([1 / (i + 1) for i, _ in enumerate(D)]))
                Q_1 = np.sum(np.array([1 / (i + 1) for i in positives_idx])) / denom
                
                y_pred.append(Q_1)
            elif self.weight == 'distance':
                positives_dist = np.array([x_i for x_i, y_i in D if y_i == 1])
                denom = np.sum(np.array([1 / x_i for x_i, _ in D]))
                Q_1 = np.sum(np.array([1 / d for d in positives_dist])) / denom
                
                y_pred.append(Q_1)
        return np.array(y_pred)