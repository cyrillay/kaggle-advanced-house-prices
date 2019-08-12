#!/usr/bin/python3

# Author : Cyril Lay
# Kaggle competition - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard
# The goal of this machine learning script is train and evaluate different models on the Kaggle housing dataset to get the best score.
# The dataset is composed of various features about houses and their sale price

import numpy as np
import pandas as pd
import json
from sklearn import metrics 
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from data_preparation import clean_data, unskew_target_variable, encode_to_categorical, feature_engineering
from model_validation import get_cross_validation_r2_score, get_cross_validation_RMSLE, get_RMSLE, rmsle_cv, pretty_print_scores, get_mean_absolute_error
from models_preparation import initialize_model
from data_visualization import plot, plot_distribution, missing_data_ratio, correlation_heatmap, pretty_print
from scipy.stats import norm, skew

 # Limit floats output to 2 decimal points
pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


# ------------------- Pre processing ------------------- #

def load_dataset(path):
    input = pd.read_csv(path, header = 0)
    return pd.DataFrame(input)

# --------------------- Model --------------------- #

def get_random_forest_best_parameters_by_grid_search(X, Y):
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': (25, 30, 40, 45),
            'n_estimators': (50, 60, 75, 100),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
    )
    grid_result = gsc.fit(X, Y)
    best_params = grid_result.best_params_
    print(json.dumps(best_params, indent = 2))
    return best_params

# ------------------- Submission ------------------- #
def create_submission_file(model, x_test, x_test_ids, x_train, y_train) :
    model.fit(x_train, y_train)
    predicted_price = model.predict(x_test)
    submission = pd.DataFrame()
    submission['Id'] = x_test_ids
    submission['SalePrice'] = np.expm1(predicted_price)
    submission.to_csv("submission.csv", index=False)

def main() :
    data_raw = load_dataset(path = "train.csv")
    # test.csv doesn't have the labels, can't be used for training, but can be for filling missing values
    x_submission = load_dataset(path = "test.csv")
    test_ids = x_submission['Id']
    train_ids = data_raw['Id']
    ntrain = data_raw.shape[0]

    all_data = pd.concat((data_raw, x_submission)).reset_index(drop=True)
    print("all_data size is : {}".format(all_data.shape))

    print("missing data before cleaning : ", missing_data_ratio(all_data))
    data_cleaned = clean_data(all_data)
    print("missing data after cleaning : ", missing_data_ratio(data_cleaned))

    data_cleaned.drop('Id', axis=1, inplace = True)
    data_cleaned_encoded = encode_to_categorical(data_cleaned)
    data_cleaned_encoded_extended = feature_engineering(data_cleaned_encoded)
    print("Final dataset : ", data_cleaned_encoded_extended)

    # plot_distribution(data_cleaned_encoded_extended)
    # ==> Skew is corrected

    # Separate the two datasets
    train_data_cleaned_encoded_extended = data_cleaned_encoded_extended[:ntrain]
    x_submission_cleaned_encoded_extended = data_cleaned_encoded_extended[ntrain:]
    x_submission_cleaned_encoded_extended.drop('SalePrice', axis=1, inplace = True)

    # Split train and test data
    df_train, df_test = train_test_split(train_data_cleaned_encoded_extended, test_size=0.4)
    unskew_target_variable(df_train)
    
    cp = train_data_cleaned_encoded_extended.copy() 
    unskew_target_variable(cp)
    y = cp.SalePrice
    y_train = df_train.SalePrice
    y_test = df_test.SalePrice

    x = train_data_cleaned_encoded_extended.drop("SalePrice", axis=1)
    x_train = df_train.drop("SalePrice", axis=1)
    x_test = df_test.drop("SalePrice", axis=1)

    #  -------------- Model Comparison --------------

    # print("------- Linear Regression -------")
    # regression_model = initialize_model("linear_regression")
    # regression_model.fit(x_train, y_train)
    # y_pred = regression_model.predict(x_test)
    # print(y_test[:5])
    # print(y_pred[:5])
    # print("Cross validation R2 : ", get_cross_validation_r2_score(regression_model, x, y))
    # print("Cross validation neg mean squared log error : ", get_cross_validation_RMSLE(regression_model, x, y))
    # print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred)))

    print("------- Random Forest -------")
    # random_forest_model = initialize_model("random_forest", parameters = get_random_forest_best_parameters_by_grid_search(x, y))
    random_forest_model = initialize_model("random_forest", parameters = {"max_depth" : 45, "n_estimators" : 100})
    random_forest_model.fit(x_train, y_train)
    y_pred_RF = random_forest_model.predict(x_test)
    print(y_test.T[:5])
    print(np.expm1(y_pred_RF[:5]))
    print("Cross validation R2 : ", get_cross_validation_r2_score(random_forest_model, x, y))
    print("Cross validation RMSLE : ", get_cross_validation_RMSLE(random_forest_model, x, y))
    # np.expm1 to apply the reverse transformation applied to y_train to remove skew
    print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred_RF)))

    print("------- Lasso -------")
    lasso_model = initialize_model("lasso")
    lasso_model.fit(x_train, y_train)
    y_pred_lasso = lasso_model.predict(x_test)
    print("y_test = ", y_test.head(5))
    print("y_pred = ", np.expm1(y_pred_lasso[:5]))
    print("Mean Absolute Error : ", get_mean_absolute_error(y_true = y_test, y_pred = np.expm1(y_pred_lasso)))
    # print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred_lasso)))
    # print("Cross validation mean R2 : ", get_cross_validation_r2_score(lasso_model, x, y).mean())
    # print("Cross validation mean RMSLE : ", get_cross_validation_RMSLE(lasso_model, x, y).mean())
    # print("Custom CV mean error : ", rmsle_cv(lasso_model, x, y).mean())

    print("------- XGBoost -------")
    xgboost_model = initialize_model("xgboost")
    xgboost_model.fit(x_train, y_train)
    y_pred_xgboost = xgboost_model.predict(x_test)
    print("y_test = ", y_test.head(5))
    print("y_pred = ", np.expm1(y_pred_xgboost[:5]))
    print("Mean Absolute Error : ", get_mean_absolute_error(y_true = y_test, y_pred = np.expm1(y_pred_xgboost)))
    # print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred_xgboost)))
    # print("Cross validation mean R2 : ", get_cross_validation_r2_score(xgboost_model, x, y).mean())
    # print("Cross validation mean RMSLE : ", get_cross_validation_RMSLE(xgboost_model, x, y).mean())
    # print("Custom CV mean error : ", rmsle_cv(xgboost_model, x, y).mean())

    print("------- Model Stacking -------")
    stacking_model = initialize_model("stack")
    stacking_model.fit(x_train, y_train)
    y_pred_stacking_model = stacking_model.predict(x_test)
    print("Mean Absolute Error : ", get_mean_absolute_error(y_true = y_test, y_pred = np.expm1(y_pred_stacking_model)))
    # print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred_stacking_model)))
    # print("Cross validation RMSLE : ", get_cross_validation_RMSLE(stacking_model, x, y))
    # print("Cross validation mean RMSLE : ", get_cross_validation_RMSLE(stacking_model, x, y).mean())

    pretty_print_scores(models = (random_forest_model, lasso_model, xgboost_model, stacking_model), x = x, y = y)
    print("End")

    # create_submission_file(model = stacking_model, x_test = x_submission_cleaned_encoded_extended, x_test_ids = test_ids, x_train = x, y_train = y)

if __name__== "__main__":
  main()

#   pretty_print(df)




# ---- TODO :
# - add other models,
# grid search + cross validate for hyper parameters
# Analyze XGBoost and try default parameters
# Generate other features
# Create a helper function to pretty print model various scores, also print variance of each model
# Use downsampling to handle imbalanced data.
