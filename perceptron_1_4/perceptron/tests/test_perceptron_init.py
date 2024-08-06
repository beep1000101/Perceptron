from random import randint
from sklearn.datasets import make_classification

from perceptron_1_4.perceptron import Perceptron


def test_perceptron_init_base():
    n_samples = 64
    n_features = 2
    n_informative = 2
    n_redundant = 0
    n_clusters_per_class = 1
    class_sep = 2
    random_state = randint(0, 10000)
    learning_rate = .02

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state
    )

    perceptron = Perceptron(data_points=X, learning_rate=learning_rate)


test_perceptron_init_base()
