from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
import os
from si.decomposition.pca import PCA

from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from datasets import DATASETS_PATH

class TestPCA(TestCase):
    """
    Unit tests for the PCA class, verifying its behavior in fitting and transforming data.
    """

    def setUp(self):
        """
        Set up the test environment by loading the dataset and initializing PCA with a specified number of components.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.n_components = 2
        self.pca = PCA(n_components=self.n_components)

    def test_fit(self):
        """
        Testa o processo de ajuste (_fit) do PCA para garantir que a média,
        as componentes e a variância explicada são computadas e armazenadas corretamente.
        """

        # O número de features originais (Iris) é 4
        n_features_original = self.dataset.X.shape[1]

        # 1. Execução do Fit
        self.pca._fit(self.dataset)

        # 2. Verificação de self.mean (Média das Features)
        # O self.pca.mean deve ser o vetor de médias de cada coluna
        expected_mean = np.mean(self.dataset.X, axis=0)
        self.assertTrue(np.allclose(self.pca.mean, expected_mean))
        self.assertEqual(self.pca.mean.shape, (n_features_original,))

        # 3. Verificação de self.components (Componentes Principais)
        # O shape deve ser (n_components, n_features_original)
        expected_components_shape = (self.n_components, n_features_original)
        self.assertEqual(self.pca.components.shape, expected_components_shape)

        # 4. Verificação de self.explained_variance (Variância Explicada)
        # O número de entradas deve ser igual a n_components
        self.assertEqual(len(self.pca.explained_variance), self.n_components)

        # 5. Verificação de Limites da Variância Explicada (Proporção)
        # Os valores individuais devem estar entre 0 e 1, e a soma não pode ser maior que 1
        self.assertTrue(np.all(self.pca.explained_variance >= 0) and np.all(self.pca.explained_variance <= 1))

        # 6. Verificação de Ordem (Confirma a Lógica de Ordenação)
        # O primeiro valor deve ser maior ou igual ao segundo, confirmando a ordenação decrescente
        self.assertTrue(self.pca.explained_variance[0] >= self.pca.explained_variance[1])


    def test_transform(self):
        """
        Testa se o _transform projeta corretamente os dados para o novo espaço dimensional,
        preservando o número de amostras e o tipo do objeto retornado.
        """

        # O número de features originais (Iris) é 4

        # 1. Fit e Transformação
        self.pca.fit(self.dataset)

        # dataset_transformed é o objeto Dataset devolvido por transform()
        dataset_transformed = self.pca.transform(self.dataset)

        # Array NumPy com os dados reduzidos
        X_reduced = dataset_transformed.X

        # 2. Asserções Críticas

        # A) Verificação de Tipo (Deve devolver um objeto Dataset)
        self.assertIsInstance(dataset_transformed, type(self.dataset))

        # B) Verificação do Shape (Dimensão Correta)
        # Número de linhas (amostras) deve ser o mesmo
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])

        # Número de colunas (features) deve ser igual a n_components (2)
        self.assertEqual(X_reduced.shape[1], self.n_components)

        # C) Verificação da Média (Dados Centrados)
        # O resultado X_reduced deve ter média ~0 em cada componente (PC)
        # Note: Esta é uma verificação forte, mas útil para o PCA.
        self.assertTrue(np.allclose(np.mean(X_reduced, axis=0), 0.0))

        # D) Verificação dos Nomes das Features
        expected_features = ['PC1', 'PC2']
        self.assertEqual(dataset_transformed.features, expected_features)