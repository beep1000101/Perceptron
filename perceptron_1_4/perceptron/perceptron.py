import numpy as np
import matplotlib.pyplot as plt

from random import randint
from sklearn.datasets import make_classification
from matplotlib.animation import FuncAnimation


class Perceptron():
    """Class that implements perceptron for 2D data classification."""

    def __init__(self, data_points, learning_rate):
        self.data_points = data_points
        self.data_points_bias = np.hstack(
            (np.ones((self.data_points.shape[0], 1)),
             self.data_points)
        )
        self.learning_rate = learning_rate
        # initialize random vector of ints from 0 to 1
        self.weights = np.random.randint(-1, 1, (data_points.shape[1] + 1, 1))
