import numpy as np
import pandas as pd
import random

class MySVM():
    def __init__(self, n_iter=10, learning_rate=0.001, weights=None, b=None, C=1, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.c = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def __str__(self):
        return f'MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
            
        X_train = X.values if isinstance(X, pd.DataFrame) else X
        y_train = 2 * y - 1
        
        n = len(X_train)
        m = len(X_train.T)
        self.weights = np.ones(m)
        self.b = 1
        
        batch_size = n
        if self.sgd_sample is not None:
            if isinstance(self.sgd_sample, float):
                batch_size = min(int(self.sgd_sample * n), n)
            else:
                batch_size = min(n, self.sgd_sample)
        
        L_w = np.zeros(m)
        L_b = 0
        
        for i in range(self.n_iter):
            X_batch = X_train
            y_batch = y_train
            if self.sgd_sample is not None:
                indices = random.sample(range(n), batch_size)
                X_batch = X[indices]
                y_batch = y[indices]
            for x_train, y_train in zip(X_batch, y_batch):
                if y_train * (self.weights @ x_train.T + self.b) >= 1:
                    L_w = 2 * self.weights
                    L_b = 0
                else:
                    L_w = 2 * self.weights - self.c * y_train * x_train
                    L_b = -self.c * y_train
                
                self.weights -= self.learning_rate * L_w
                self.b -= self.learning_rate * L_b
                
            margins = y * (self.weights @ X.T + self.b)
            hinge_loss = np.maximum(0, 1 - margins)
            loss = self.weights @ self.weights.T + self.c * np.mean(hinge_loss)
                                                          
            if verbose and i % verbose == 0:
                title = 'start' if i == 0 else str(i)
                print(f'{title} | loss: {loss}')
    
    def predict(self, X):
        y_pred = (np.sign(self.weights @ X.T + self.b) + 1) // 2
        return y_pred.astype(int)
        
    def get_coef(self):
        return (self.weights, self.b)