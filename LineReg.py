import numpy as np
import pandas as pd
import random


class MyLineReg():
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate=0.1,
        metric: str = None, reg: str = None,
        l1_coef: float = 0, l2_coef: float = 0,
        sgd_sample=None, random_state=42
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        if metric in ('mae', 'mse', 'rmse', 'mape', 'r2'):
            self.metric = metric
        else:
            self.metric = None
        self.metric_value = 0
        if reg in ('l1', 'l2', 'elasticnet'):
            self.reg = reg
        else:
            self.reg = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f'''
            MyLineReg class:
            n_iter={self.n_iter},
            learning_rate={self.learning_rate}
        '''

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool):
        random.seed(self.random_state)
        X.insert(0, 'x0', 1)
        n = len(X.T)
        m = len(X)
        avg = np.mean(y)
        self.weights = np.ones(n)
        predict = X @ self.weights.T

        for i in range(self.n_iter):
            sample_size = m
            sample_rows_idx = range(m)

            if self.sgd_sample:
                if (isinstance(self.sgd_sample, int)):
                    sample_size = self.sgd_sample
                else:
                    sample_size = int(m * self.sgd_sample)
                sample_rows_idx = random.sample(range(m), sample_size)

            X_batch = X.values[sample_rows_idx, :]
            y_batch = y.values[sample_rows_idx]
            predict_batch = X_batch @ self.weights.T
            grad = (2 / sample_size) * ((predict_batch - y_batch).T @ X_batch)
            if self.reg:
                if self.reg in ('l1', 'elasticnet'):
                    grad += self.l1_coef * np.sign(self.weights.T)
                if self.reg == ('l2', 'elasticnet'):
                    grad += 2 * self.l2_coef * self.weights.T

            lmd = 0
            if callable(self.learning_rate):
                lmd = self.learning_rate(i + 1)
            else:
                lmd = self.learning_rate

            self.weights -= lmd * grad
            predict = X @ self.weights.T
            loss = np.average((predict - y)**2)

            if self.reg:
                if self.reg in ('l1', 'elasticnet'):
                    loss += self.l1_coef * np.sum(abs(self.weights))
                if self.reg == ('l2', 'elasticnet'):
                    loss += self.l2_coef * np.sum(self.weights**2)

            if self.metric:
                match self.metric:
                    case 'mae':
                        self.metric_value = np.mean(abs(y - predict))
                    case 'mse':
                        self.metric_value = loss
                    case 'rmse':
                        self.metric_value = np.sqrt(loss)
                    case 'mape':
                        loss_abs = np.mean(abs(y - predict) / y)
                        self.metric_value = 100 * loss_abs
                    case 'r2':
                        diff_avg = np.sum((y - avg)**2)
                        diff_loss = np.sum((y - predict)**2)
                        self.metric_value = 1 - diff_loss / diff_avg

            if verbose and i % verbose == 0:
                idx = 'start' if i == 0 else str(i)
                print(f'{idx} | loss: {loss}', end='')
                if self.metric:
                    print(f' | {self.metric}: {self.metric_value}')
                else:
                    print()

    def predict(self, X: pd.DataFrame):
        X.insert(0, 'x0', 1)
        predict = X @ self.weights.T
        return predict

    def get_best_score(self):
        return self.metric_value

    def get_coef(self):
        return self.weights[1:]
