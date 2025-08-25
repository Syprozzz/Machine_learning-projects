import math
import numpy as np

class Neural_Network:

    def __init__(self, input_layer, hidden1_layer, hidden2_layer, output_layer):
        #initiliazing weights with random values
        np.random.seed(1)
        self.w1 = np.random.randn(input_layer, hidden1_layer)
        self.b1 = np.zeros((1, hidden1_layer))

        self.w2 = np.random.randn(hidden1_layer, hidden2_layer)
        self.b2 = np.zeros((1, hidden2_layer))

        self.w3 = np.random.randn(hidden2_layer, output_layer)
        self.b3 = np.zeros((1, output_layer))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def loss_func(self, y_actual, y_pred):
           #clip predictions to avoid log(0) --> infinite
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))
        return loss

    def forward_pass(self, X):
    #now lets pass our forward predictions

        #for hidden layer 1
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        #for hidden layer 2
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)

        #for output layer 
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3).reshape(-1, 1)
        #here reshaping to (-1, 1) is actually crucial which i found out after hours of debugging
        #It ensures a3 is a 2d array which if not can hinder backprop calculations
        return self.a3
    
    def back_prop(self, X, y):
    #calculating gradients
        #of output layer
        #the derivatives actually quite simple taking loss wrt z3 w3 and b3 using chain rule just solve bruv
        self.dz3 = self.a3 - y.reshape(-1, 1)
        self.dw3 = np.dot(self.a2.T, self.dz3)
        self.db3 = np.sum(self.dz3, axis=0, keepdims=True)
        #keepdims=True for shape (1, no_of_neurons) rather than (no_of_neurons, )
        #of hidden layer 2
        #the derivaties here is quite confusing but quite easy you just use chain rule here too
        #eg da2--> dL/da3 . da3/dz3 . dz3/da2 --> solve and --> dz3 . w3
        self.da2 = np.dot(self.dz3, self.w3.T)   
        self.dz2 = self.da2 * self.relu_derivative(self.z2)   
        self.dw2 = np.dot(self.a1.T, self.dz2)   
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        #of hidden layer 1
        self.da1 = np.dot(self.dz2, self.w2.T)
        self.dz1 = self.da1 * self.relu_derivative(self.z1)
        self.dw1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)

    def update_params(self, lr=0.01):
        #updating all the weights and biases
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2
        self.w3 -= lr * self.dw3
        self.b3 -= lr * self.db3

    def train(self, X, y, batch_size=32, epochs=2000):
        n_samples, n_features = X.shape
        
        for epoch in range(epochs):
            #shuffling data at every epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            #lets loop from 0 to n_samples over our batch size increment per loop
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                #forward pass
                self.forward_pass(X_batch)
                #backpropagation
                self.back_prop(X_batch, y_batch)
                #updating parameters
                self.update_params()
            
            # Compute loss on the entire training set
            self.forward_pass(X)          # make a3 = predictions for all X
            y_pred = self.a3
            if epoch % 200 == 0:
                acc = self.accuracy(X, y)
                print(f"Epoch: {epoch}, loss: {self.loss_func(y, y_pred):.4f}, accuracy: {acc:.4f}")

    def predict(self,X):

        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3).reshape(-1, 1)

        return(self.a3>0.5).astype(int)

    def accuracy(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)



