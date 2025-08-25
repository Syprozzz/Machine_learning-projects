import numpy as np
from model import Neural_Network
from data import get_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = get_data()

y = y.reshape(-1, 1)
#splitting data into two parts for training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


#rescaling features so mean is 0 and S.D is 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = Neural_Network(X_train_scaled.shape[1], 16, 8, 1)
model.train(X_train_scaled, y_train)

prediction = model.predict(X_test_scaled)

accuracy = model.accuracy(X_test_scaled, y_test)
print(f"The accuracy of the model is {accuracy*100:.4f}%")