import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics import f_classification


class SelectPercentile(Transformer):

    def __init__(self, percentile: float, score_func: callable = f_classification, **kwargs):
        """
        Selects features from the given percentile of a score function
        and returns a new Dataset object with the selected features.

        methods:- _fit – estimates the F and p values for each feature using the scoring_func;
        returns itself (self)- _transform – Selects a given percentage of features based on their F-values,
        ensuring the number of selected features adheres to the specified percentile.
        The method handles ties at the threshold to maintain the correct number of
        features. Example: Suppose you have 10 features with F-values [1.2, 3.4, 2.1,
        5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2] and set percentile=40. The threshold is
        calculated as the F-value at the 60th percentile, which is 5.6. Initially, the
        mask selects features with F-values greater than 5.6, resulting in [7.8, 6.5].
        Since we need 4 features (40% of 10), the method identifies ties at the
        threshold (5.6) and includes the first two of these tied features ([5.6, 5.6]) to
        meet the requirement. The final selected features are [5.6, 5.6, 7.8, 6.5],
        ensuring the output adheres to the specified percentile.


        Parameters
        ----------
        percentile : float
            Percentile (0-100) for selecting features.
        score_func : callable, optional
            Variance analysis function. Uses f_classification by default.
        """
        super().__init__(**kwargs)

        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be a float between 0 and 100")
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimate the F and P values for each feature using the scoring function.

        Parameters
        ----------
        dataset : Dataset
            Dataset object where features are to be selected.

        Returns
        -------
        self : SelectPercentile
            The instance with the F and P values calculated.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features with the highest F value up to the specified percentile.

        Parameters
        ----------
        dataset : Dataset
            Dataset object to select features from.

        Returns
        -------
        dataset : Dataset
            A new Dataset object with the selected features.
        """
        threshold = np.percentile(self.F, 100 - self.percentile)

        mask = self.F >= threshold

        if mask.sum() > int(len(self.F) * self.percentile / 100):
            sorted_indices = np.argsort(-self.F)
            num_features = int(len(self.F) * self.percentile / 100)
            selected_indices = sorted_indices[:num_features]
            mask = np.zeros_like(self.F, dtype=bool)
            mask[selected_indices] = True

        selected_features = np.array(dataset.features)[mask]
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=list(selected_features), label=dataset.label)