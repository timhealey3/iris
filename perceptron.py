import numpy as np
from data import IrisDataset
import pandas as pd

class Perceptron:
    def __init__(self, lr=0.01, epochs=50, random_seed=1):
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed

    """X training data, y target values"""
    def fit(self, X, y):
        random_gen = np.random.RandomState(self.random_seed)
        self.weights_ = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = np.float_(0.)
        self.errors = []
        # training loop
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.weights_ += update * xi
                self.bias_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    """calc net input"""
    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_
    
    """predict binary classifier"""
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
"""Sepal length, Pedal Length = Class Setosa/Veriscolor works in a perceptron because the two classes can be seeperated by a linear hyperplane"""
df = IrisDataset()
perceptron = Perceptron()
y = df.iloc[0:100, 4].values
X = df.iloc[0:100, [0, 2]].values
perceptron.fit(X, y)
