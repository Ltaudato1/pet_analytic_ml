import pandas as pd
import numpy as np


def sigma_function(X, w):
    M = X @ w.T
    return 1 / (1 + np.exp(-M))

class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
    
    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X, y, verbose):
        features = X.copy()
        features.insert(0, 'x0', 1)
        n = len(features.T)
        m = len(features)
        self.weights = np.ones(n)
        eps = 1e-15
        
        for i in range(self.n_iter):
            predict = sigma_function(features, self.weights)
            logLoss = -np.mean(y * np.log(predict + eps) + (1 - y) * np.log(1 - predict + eps))
            grad = (1 / m) * (features.T @ (predict - y))
            self.weights -= self.learning_rate * grad
            
            if verbose and i % verbose == 0:
                idx = 'start' if i == 0 else str(i)
                print(f'{idx} | loss: {logLoss}', end='')
                if self.metric:
                    print(f' | {self.metric}: {self.metric_value}')
                else:
                    print()       
                    
    def predict_proba(self, X):
        features = X.copy()
        features.insert(0, 'x0', 1)
        return sigma_function(features, self.weights)
    
    def predict(self, X):
        features = X.copy()
        features.insert(0, 'x0', 1)
        trash_hold = 0.5
        return (sigma_function(features, self.weights) > trash_hold)
    
    def get_coef(self):
        return self.weights[1:]
    