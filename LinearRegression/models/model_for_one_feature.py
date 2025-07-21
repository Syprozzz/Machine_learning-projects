from load_data import get_data
from main import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
#features_names = fetch_california_housing().feature_names

X, y = get_data()
#features
features=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
total_features = len(features)
#feature_index = 2  
#Standarization
def standardize(X):
    mean = np.mean(X, axis=0)   
    std = np.std(X, axis=0)
    return (X-mean)/std, mean, std

X_scaled, mean, std = standardize(X)

#rscore
def r_sqr_score(y_true, y_pred,):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res/ss_total)

#our model
model_single = LinearRegression()
#looping through each feature
for feature_index in range(total_features):
    X_feature = X_scaled[:, feature_index].reshape(-1, 1)


    model_single.fit(X_feature, y)
    y_pred_single = model_single.predict(X_feature)



    r2score = r_sqr_score(y,y_pred_single)
    print(f"r square error is {r2score} for {features[feature_index]}")

# Plotting the best fit line for the first feature

#plt.figure(figsize=(8,6))
#plt.scatter(X_feature, y, color='green', label='Data points')
#plt.plot(X_feature, y_pred_single, color='red', label='Best fit line')
#plt.xlabel(features[feature_index])
#plt.ylabel("Target (y)")
#plt.title("Linear Regression - One Feature")
#plt.legend()
#plt.grid(True)
#plt.show()
