import math
import numpy as np

class LinearRegression:
    
    def __init__(self):
        self.weight=None
        self.bias=None
        self.loss_history=[]
        

    def fit(self, X, y, epochs=2000, learning_rate=0.01):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(epochs):
            y_predicted = np.dot(X, self.weight) + self.bias
            
            dw = 1/n_samples * np.dot(X.T, (y_predicted - y))
            db = 1/n_samples * np.sum(y_predicted - y)

            self.weight = self.weight - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            # Track loss
            loss = np.mean((y - y_predicted) ** 2)
            self.loss_history.append(loss)

            #print loss every 100 epochs
            #if epoch % 100 == 0:
             #   print(f"Epoch {epoch}, Loss: {loss}")
    

    def predict(self, X):
        y_predicted = np.dot(X, self.weight) + self.bias
        return y_predicted
