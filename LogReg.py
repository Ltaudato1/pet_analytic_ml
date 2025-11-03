import numpy as np
import pandas as pd
import random


def calc_metric(metric: str, classes: list, threshold: float):
    if metric == 'roc_auc':
        s = 0
        s_equals = 0
        classes.sort(key=lambda x: (x[0], x[1]), reverse=True)
        positives = 0
        negatives = 0
        for i in range(len(classes)):
            if classes[i][1] == 1:
                positives += 1
                if i > 0 and classes[i][0] == classes[i-1][0]:
                    s_equals += 1
                else:
                    s_equals = 0
            else:
                negatives += 1
                if classes[i][0] == classes[i-1][0] and classes[i-1][1] == 1:
                    s_equals += 1
                s += positives + s_equals / 2
        return s / (positives * negatives)
    else:
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for item in classes:
            if item[0] > threshold:
                if item[1] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if item[1] == 0:
                    tn += 1
                else:
                    fn += 1
        match metric:
            case 'accuracy':
                return (tp + tn) / (tp + tn + fp + fn)
            case 'precision':
                return tp / (tp + fp)
            case 'recall':
                return tp / (tp + fn)
            case 'f1':
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                return 2 * precision * recall / (precision + recall)
            case _:
                return None


def sigma_function(X, w):
    M = X @ w.T
    return 1 / (1 + np.exp(-M))


class MyLogReg():
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate=0.1,
        metric: str = None, reg: str = None,
        l1_coef: float = 0, l2_coef: float = 0,
        sgd_sample=None, random_state=42
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        if metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
            self.metric = metric
        else:
            self.metric = None
        self.metric_value = 0
        self.threshold = 0.5
        if reg in ('l1', 'l2', 'elasticnet'):
            self.reg = reg
        else:
            self.reg = None
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return f'''
            MyLogReg class:
            n_iter={self.n_iter}
            learning_rate={self.learning_rate}
        '''

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        features = X.copy()
        features.insert(0, 'x0', 1)
        n = len(features.T)
        m = len(features)
        self.weights = np.ones(n)
        eps = 1e-15
        predict = sigma_function(features, self.weights)

        for i in range(self.n_iter):
            sample_size = m
            sample_rows_idx = range(m)

            if self.sgd_sample:
                if (isinstance(self.sgd_sample, int)):
                    sample_size = self.sgd_sample
                else:
                    sample_size = min(m, int(m * self.sgd_sample))
                sample_rows_idx = random.sample(range(m), sample_size)

            X_batch = features.values[sample_rows_idx, :]
            y_batch = y.values[sample_rows_idx]
            predict_batch = sigma_function(X_batch, self.weights)

            logLoss = -np.mean(
                y * np.log(predict + eps)
                + (1 - y) * np.log(1 - predict + eps)
                )
            grad = (1 / sample_size) * (X_batch.T @ (predict_batch - y_batch))
            if self.reg:
                if self.reg in ('l1', 'elasticnet'):
                    logLoss += self.l1_coef * sum(abs(self.weights))
                    grad += self.l1_coef * np.sign(self.weights.T)
                if self.reg in ('l2', 'elasticnet'):
                    logLoss += self.l2_coef * sum(self.weights**2)
                    grad += 2 * self.l2_coef * self.weights.T

            lmd = 0
            if callable(self.learning_rate):
                lmd = self.learning_rate(i + 1)
            else:
                lmd = self.learning_rate
            self.weights -= lmd * grad
            predict = sigma_function(features, self.weights)

            if verbose and i % verbose == 0:
                idx = 'start' if i == 0 else str(i)
                print(f'{idx} | loss: {logLoss}', end='')
                if self.metric:
                    classes = [(x, y) for x, y in zip(predict, y)]
                    self.metric_value = calc_metric(
                        self.metric, classes, self.threshold
                    )
                    print(f' | {self.metric}: {self.metric_value}')
                else:
                    print()

        classes = [(x, y) for x, y in zip(predict, y)]
        self.metric_value = calc_metric(self.metric, classes, self.threshold)

    def predict_proba(self, X):
        features = X.copy()
        features.insert(0, 'x0', 1)
        return sigma_function(features, self.weights)

    def predict(self, X):
        features = X.copy()
        features.insert(0, 'x0', 1)
        return (sigma_function(features, self.weights) > self.threshold)

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.metric_value
