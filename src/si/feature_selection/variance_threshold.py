from si.base.transformer import Transformer
from si.base.transformer import Transformer
from si.data.dataset import Dataset
import numpy as np

class VarianceThreshold(Transformer):

    def __init__(self, threshold:float, **kwargs):
        self.threshold = threshold
        self.variance = None

    def _fit(self, X, y=None) -> 'VarianceThreshold':
        self.variance = np.var(Dataset.X, axis=0) # computa a variança ao longo do eixo das linhas

    def _transform(self, dataset:Dataset) -> Dataset:
        mask = self.variance >= self.threshold
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask]
        return Dataset(X=X, features=features, y=dataset.y, label = dataset.label)

