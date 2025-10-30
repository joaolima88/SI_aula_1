import numpy as np

from si.data import dataset
from si.data.dataset import Dataset


class RidgeClassification(Model):

    def __init__(self,l2_term, alpha, max_iter, patience, scale, **kwargs):
        self.l2_term = l2_term
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = None

        def _fit(self, dataset:Dataset):
            if self.scale:
                self.mean = np.nanmean(dataset.X, axis=0)
                self.std = np.nanstd(dataset.X, axis=0)
                x = (dataset.X - self.mean) / self.std

            else:
                x = dataset.X
                m, n = dataset.shape()
                self.theta_zero = 0

                i = 0
                early_stopping = 0

                while i < self.max_iter and early_stopping <= self.patience:

                    y_pred = np.dot(x, self.theta) + self.theta_zero
                    gradiente = (self.alpha / m) * np.dot(y_pred - dataset.y, x)
                    penalization_term = self.theta * (1 - (self.alpha * self.l2_term / m))

                    self.theta = penalization_term - gradiente
                    self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - dataset.y)

                    self.cost_history[i] = self.cost(dataset.y, y_pred)

                    if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                        early_stopping += 1

                    i += 1

                return self

            def cost(y_true, y_pred):
                m = len(y_true)
                return 1 / 2*m * (np.sum(y_pred - y_true)**2 - (self.l2_term * np.sum(self.theta**2)))


