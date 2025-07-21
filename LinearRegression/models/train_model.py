from load_data import get_data
from main import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#features_names = fetch_california_housing().feature_names

X, y = get_data()

#mean of features
#X_mean = X.mean(axis=0)
#Standarization
def standardize(X):
    mean = np.mean(X, axis=0)   
    std = np.std(X, axis=0)
    return (X-mean)/std, mean, std

X_scaled, mean, std = standardize(X)

#fitting model 
model = LinearRegression()
model.fit(X_scaled, y)

#predicting value
y_pred = model.predict(X_scaled)

#model loss history
loss_history=model.loss_history

#mean error value
def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(y, y_pred)
print(mse_value)


#plotting loss curve

plt.figure(figsize=(8,6))
plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Loss Curve - MSE vs Epochs")
plt.grid(True)
plt.show()

#plotting r square score

def r_sqr_score(y_true, y_pred,):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res/ss_total)

r2score = r_sqr_score(y,y_pred)
print(f"r square error is {r2score}")