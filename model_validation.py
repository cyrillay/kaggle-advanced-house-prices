#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

DEFAULT_CROSS_VALIDATION_ITERATIONS = 5

def get_RMSLE(y_true, y_pred) :
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def get_mean_absolute_error(y_true, y_pred) : 
    return mean_absolute_error(y_true, y_pred)

def get_cross_validation_r2_score(model, x, y, cv = DEFAULT_CROSS_VALIDATION_ITERATIONS) : 
    return np.absolute(cross_val_score(model, x, y, cv=cv, scoring="r2"))

def get_cross_validation_RMSLE(model, x, y, cv = DEFAULT_CROSS_VALIDATION_ITERATIONS) : 
    # if neg --> throw error
    return np.sqrt(np.absolute(cross_val_score(model, x, y, cv=cv, scoring="neg_mean_squared_log_error")))



#Validation function
n_folds = 5

def rmsle_cv(model, x, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x)
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def pretty_print_scores(models, x, y) :
    scores = {}
    df = pd.DataFrame(scores, columns= ['Model', 'RMSLE', 'R2', 'custom', 'Variance'])
    for i in range(0, len(models)):
        model = models[i]
        model_name = type(model).__name__
        rmsle = get_cross_validation_RMSLE(model = model, x = x, y = y)
        r2_scores = get_cross_validation_r2_score(model = model, x = x, y = y)
        custom = rmsle_cv(model, x, y)
        new_line = [model_name] + list((rmsle.mean(), r2_scores.mean(), custom.mean(), custom.std()))
        print("new line = ", new_line)
        df.loc[i] = new_line
    
    print("All scores = \n", df)
    return df
            

