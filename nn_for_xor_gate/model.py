import math
import numpy as np

#Dataset for XOR Gate
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 1

        #initializing weights and biases for layers
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def fit(self, X, y, epochs=10000, lr=0.01):
        for epoch in range(epochs):
            #forward pass
            z1 = np.dot(X, self.w1) + self.b1
            a1 = self.sigmoid(z1)

            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self.sigmoid(z2)
            #loss
            loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))
            #"1e-8" coz if sometimes value nearer to 0 so it prevents infinite[log(0)-->inf]

            #backprop
            dz2 = a2 - y
            dw2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.w2.T)
            dz1 = da1 * self.sigmoid_derivative(z1)
            dw1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

             #update weights and biases
            self.w1 -= lr * dw1
            self.b1 -= lr * db1
            self.w2 -= lr * dw2
            self.b2 -= lr * db2

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, loss: {loss:.4f}")

    def predict(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)

        
        return (a2 > 0.45).astype(int)

# Train the model
model = NeuralNetwork()
model.fit(X, y)

# Test predictions
predictions = model.predict(X)
print("\nPredictions:")
print(predictions)