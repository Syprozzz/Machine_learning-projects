import numpy as np
from sklearn.datasets import fetch_california_housing

def get_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target
    return X,y
