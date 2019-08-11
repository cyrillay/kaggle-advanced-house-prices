#!/usr/bin/python3

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

def initialize_model(model_type, parameters = {}) :
    if (model_type.startswith("linear")) :
        return linear_model.LinearRegression(normalize = True)
    elif (model_type.startswith("random")) : 
        return RandomForestRegressor(max_depth=parameters["max_depth"], n_estimators=parameters["n_estimators"], random_state=False, verbose=False)