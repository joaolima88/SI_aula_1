import unittest
import numpy as np
from si.data.dataset import Dataset




class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])


    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())



#----------------------------------- TEST DROPNA -----------------------------------------------------------------------

    def test_dropna_removes_nans_in_x_and_y(self):

        X = np.array([
            [1.0, 2.0, np.nan],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [np.nan, 11.0, 12.0],
            [13.0, 14.0, 15.0]
        ])

        y = np.array([
            100.0,
            200.0,
            np.nan,
            400.0,
            500.0
        ])

        # lines to keep: 1,4

        dataset = Dataset(X, y)

        dataset_cleaned = dataset.dropna()
        expected_X = np.array([[4.0, 5.0, 6.0],[13.0, 14.0, 15.0]])
        expected_y = np.array([200.0,500.0])

        self.assertEqual(dataset_cleaned.X.shape, expected_X.shape)
        self.assertEqual(dataset_cleaned.y.shape, expected_y.shape)
        self.assertTrue(np.allclose(dataset_cleaned.X, expected_X))
        self.assertTrue(np.allclose(dataset_cleaned.y, expected_y))

    def test_dropna_on_clean_data(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([10, 20])

        dataset = Dataset(X.copy(), y.copy())
        dataset_cleaned = dataset.dropna()

        self.assertTrue(np.allclose(dataset_cleaned.X, X))
        self.assertTrue(np.allclose(dataset_cleaned.y, y))


# ----------------------------------- TEST FILLNA ----------------------------------------------------------------------


# FIXED VALUE
    def test_fillna_with_fixed_value(self):

        self.X = np.array([
            [10.0, 1.0],
            [20.0, np.nan],  # NaN na Coluna 1
            [np.nan, 3.0],  # NaN na Coluna 0
            [40.0, 4.0]
        ])
        self.y = np.array([1, 2, 3, 4])
        dataset = Dataset(self.X.copy(), self.y.copy())

        fixed_value = 15.23
        dataset_filled = dataset.fillna(fixed_value)

        expected_X = np.array([
            [10.0, 1.0],
            [20.0, fixed_value],
            [fixed_value, 3.0],
            [40.0, 4.0]
        ])

        self.assertEqual(dataset_filled.X.shape, expected_X.shape)
        self.assertTrue(np.allclose(dataset_filled.X, expected_X))
        self.assertTrue(np.allclose(dataset_filled.y, self.y))


# MEAN
    def test_fillna_with_mean(self):
        self.X = np.array([
            [10.0, 1.0],
            [10.0, np.nan],  # NaN na Coluna 1
            [np.nan, 3.0],  # NaN na Coluna 0
            [10.0, 4.0]
        ])
        self.y = np.array([1, 2, 3, 4])
        dataset = Dataset(self.X.copy(), self.y.copy())

        mean_col_0 = 10.0
        mean_col_1 = 8.0 / 3.0

        calculated_means = dataset.get_mean()
        expected_means = np.array([mean_col_0, mean_col_1])

        self.assertTrue(np.allclose(calculated_means, expected_means))

        dataset_filled = dataset.fillna('mean')

        expected_X = np.array([
            [10.0, 1.0],
            [10.0, mean_col_1],
            [mean_col_0, 3.0],
            [10.0, 4.0]
        ])

        self.assertEqual(dataset_filled.X.shape, expected_X.shape)
        self.assertTrue(np.allclose(dataset_filled.X, expected_X))


# MEDIAN
    def test_fillna_with_median(self):
        self.X = np.array([
            [10.0, 1.0],
            [20.0, np.nan],  # NaN na Coluna 1
            [np.nan, 3.0],  # NaN na Coluna 0
            [40.0, 4.0]
        ])
        self.y = np.array([1, 2, 3, 4])
        dataset = Dataset(self.X.copy(), self.y.copy())

        median_col_0 = 20.0 #10.0, 20.0, 40.0 -> middle value = 20.0
        median_col_1 = 3.0 #1.0, 3.0, 4.0 -> middle value = 3.0

        calculated_medians = dataset.get_median()
        expected_medians = np.array([median_col_0, median_col_1])

        self.assertTrue(np.allclose(calculated_medians, expected_medians))

        dataset_filled = dataset.fillna('median')

        expected_X = np.array([
            [10.0, 1.0],
            [20.0, median_col_1],  # NaN na Coluna 1 preenchido
            [median_col_0, 3.0],  # NaN na Coluna 0 preenchido
            [40.0, 4.0]
        ])

        self.assertEqual(dataset_filled.X.shape, expected_X.shape)
        self.assertTrue(np.allclose(dataset_filled.X, expected_X))


# ----------------------------------- TEST REMOVE_BY_INDEX -------------------------------------------------------------


    def test_remove_by_index(self): #remove a primeira amostra
        self.X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        self.y = np.array([100.0, 200.0, 300.0, 400.0])
        dataset = Dataset(self.X.copy(), self.y.copy())

        ind = 0
        dataset_modified = dataset.remove_by_index(ind)

        expected_X = np.array([
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        expected_y = np.array([200.0, 300.0, 400.0])

        self.assertEqual(len(dataset_modified.X), len(expected_X))
        self.assertTrue(np.allclose(dataset_modified.X, expected_X))
        self.assertTrue(np.allclose(dataset_modified.y, expected_y))


