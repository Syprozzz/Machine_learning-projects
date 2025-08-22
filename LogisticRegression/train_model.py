import numpy as np
from model import LogisticRegression
from data import get_data
from sklearn.model_selection import train_test_split

X, y = get_data()

def standardize(X):
    mean = np.mean(X, axis=0)   
    std = np.std(X, axis=0)
    return (X - mean)/std, mean, std

#splitting data into two parts for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#now standardize those train and test sets
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled, _, _ = standardize(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)

probability = model.predict_probability(X_test_scaled)
prediction = model.predict(X_test_scaled)

train_loss = model.log_loss(X_train_scaled, y_train)
test_loss = model.log_loss(X_test_scaled, y_test)

print(f"Train Log Loss: {train_loss:.4f}")
print(f"Test Log Loss: {test_loss:.4f}")


accuracy = model.accuracy(X_test_scaled, y_test)
print(f"The accuracy of the model is {accuracy:.4f}")

