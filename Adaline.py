import numpy as np

class AdalineGD:
    """Implemnentation of ADApative LInear NEuron classifier"""
    def __init__(self, lr=0.01, epochs=10, shuffle=True, random_seed=1):
        self.lr = lr
        self.epochs = epochs
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.w_initialized = False

    """fitting function"""
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self
    
    """Shuffle data through making random permutations"""
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    """init weights to small random numbers"""
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_seed)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    """update the weights"""
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.lr * 2.0 * xi * error
        self.b_ += self.lr * 2.0 * error
        loss = error**2
        return loss

    """neuron calc layer"""
    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_
    
    """activation layer"""
    def activation(self, X):
        return X
    
    """prediction layer"""
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)