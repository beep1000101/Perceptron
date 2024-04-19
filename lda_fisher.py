import numpy as np


class LDAFisherClassifier:
    '''
    LDA Fisher Classifier that uses Fisher's Linear Discriminant Analysis to classify 2D data.
    '''

    def __init__(self, vector1, vector2):
        self.v1 = vector1
        self.v2 = vector2

    def fit(self):
        '''Fit the linear parameters to the data'''
        # Compute means and covariance matrices
        mean1 = np.mean(self.v1, axis=0)
        mean2 = np.mean(self.v2, axis=0)
        covariance1 = np.cov(self.v1.T)
        covariance2 = np.cov(self.v2.T)

        # Calculate the pooled covariance matrix
        w_matrix = covariance1 + covariance2

        # Compute the inverse of the pooled covariance matrix
        w_matrix_inverted = np.linalg.inv(w_matrix)

        # Calculate the canonical vector
        canonical_vector = w_matrix_inverted @ (mean1 - mean2)

        # Calculate the slope
        self.slope = -canonical_vector[0] / canonical_vector[1]

        # Compute the intercept based on projection onto the canonical vector
        midpoint = (mean1 + mean2) / 2
        self.intercept = - self.slope * midpoint[0] + midpoint[1]

    def linear_params(self):
        '''Return the slope and intercept of the linear decision boundary'''
        return self.slope, self.intercept
