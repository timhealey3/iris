import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LogisticRegressionGD:
    """Gradient descent based logistic regression"""
    def __init__(self, lr=0.01, epochs=50, random_seed=1):
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed

    """training or fitting"""
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias_ = np.float(0.)
        self.losses = []
        for i in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weights_ += self.lr * 2.0 * X.T.dot(errors) / X.shape[0]
            self.bias_ += self.lr * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output))- ((1 - y).dot(np.log(1 - output)))/ X.shape[0])
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_
    
    """Logistic Sigmoid Activation"""
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# standardize the data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
lgrd = LogisticRegressionGD(lr=0.3, epochs=1000, random_seed=1)
# or can do the scikit-learns implementation
lr = LogisticRegression(C=100.0, solver='lbfgs')
lr.fit(X_train_std, y_train)
prediction = lr.predict(X_test_std[:3, :])
print(prediction)