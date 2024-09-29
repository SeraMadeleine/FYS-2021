import numpy as np
from scipy.stats import gamma, norm
from sklearn.model_selection import train_test_split


class BayesClassifier:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.c0 = x[y == 0]
        self.c1 = x[y == 1]
        
        # Parameters for distributions
        self.alpha = 2          # Shape parameter for the gamma distribution
        self.beta = None
        self.mu = None
        self.sigma = None

    def estimate_parameters(self, c0_train, c1_train):
        """Estimate parameters based on MLE."""
        self.beta = self.alpha / np.mean(c0_train)
        self.mu = np.mean(c1_train)
        self.sigma = np.sqrt(np.mean((c1_train - self.mu)**2))

    def classify(self, x):
        """Classify data points based on the estimated parameters."""
        p_c0 = gamma.pdf(x, a=self.alpha, scale=1/self.beta)
        p_c1 = norm.pdf(x, loc=self.mu, scale=self.sigma)
        return np.where(p_c0 > p_c1, 0, 1)

    def train_and_test(self, test_size=0.2, random_state=42):
        """Split data into training and test sets, estimate parameters, and test the classifier."""
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=random_state)
        c0_train = x_train[y_train == 0]
        c1_train = x_train[y_train == 1]
        

        # Estimate parameters based on the training data
        self.estimate_parameters(c0_train, c1_train)
        

        # Classify the test data
        y_pred = self.classify(x_test)
        accuracy = np.mean(y_pred == y_test)
        
        return x_test, y_test, y_pred, accuracy

