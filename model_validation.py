#!/usr/bin/python3

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error

DEFAULT_CROSS_VALIDATION_ITERATIONS = 5

def get_RMSLE(y_true, y_pred) :
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def get_cross_validation_r2_score(model, x, y, cv = DEFAULT_CROSS_VALIDATION_ITERATIONS) : 
    return np.absolute(cross_val_score(model, x, y, cv=cv, scoring="r2"))

def get_cross_validation_RMSLE(model, x, y, cv = DEFAULT_CROSS_VALIDATION_ITERATIONS) : 
    return np.absolute(cross_val_score(model, x, y, cv=cv, scoring="neg_mean_squared_log_error"))


