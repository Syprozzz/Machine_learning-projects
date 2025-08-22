import math 
import numpy as np
class LogisticRegression:

    def __init__(self, n_iters=1000, lr= 0.001):
        self.weights = None
        self.bias = None
        self.n_iters = n_iters
        self.lr = lr

    def sigmoid(self,z):
        #clipping to avoid overflow on runtime
        z = np.clip(z, -500, 500) 
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        #initializing both weights and bias as zeros

        for _ in range(self.n_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            #basically just shuffling through dataset for SGD implementation

            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                #looping over shuffled data(X)
                z = np.dot(x_i, self.weights) + self.bias
                a_prediction = self.sigmoid(z)

                dw = (a_prediction - y_i)*x_i
                db = (a_prediction - y_i)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict_probability(self,X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    #now for binary classification
    def predict(self, X):
        probs = self.predict_probability(X)
        return (probs>=0.5).astype(int)
    #returns true to false as 1,0

    def log_loss(self, X, y):
        eps = 1e-15
        probs = self.predict_probability(X)
        probs = np.clip(probs, eps, 1 - eps)
        loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        return loss

    def accuracy(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

                

       

