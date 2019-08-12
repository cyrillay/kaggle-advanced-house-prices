#!/usr/bin/python3

import xgboost as xgb
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

def initialize_model(model_type, parameters = {}) :
    if (model_type.startswith("linear")) :
        return linear_model.LinearRegression(normalize = True)
    elif (model_type.startswith("random")) : 
        return RandomForestRegressor(max_depth=parameters["max_depth"], n_estimators=parameters["n_estimators"], random_state=False, verbose=False)
    elif (model_type.startswith("lasso")) :
        return make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    elif (model_type.startswith("xgboost")) :
        return xgb.XGBRegressor(
                    colsample_bytree=0.4603, gamma=0.0468, 
                    learning_rate=0.05, max_depth=3, 
                    min_child_weight=1.7817, n_estimators=2200,
                    reg_alpha=0.4640, reg_lambda=0.8571,
                    subsample=0.5213, silent=1,
                    random_state =7, nthread = -1
                )
    elif (model_type.startswith("stack")) :
        return AveragingModels(models = (initialize_model("random", parameters = {"max_depth" : 45, "n_estimators" : 100}), initialize_model("lasso"), initialize_model("xgboost")))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)