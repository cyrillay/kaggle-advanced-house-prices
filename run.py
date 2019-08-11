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

from data_preparation import clean_data, clean_data_train, encode_to_categorical
from model_validation import get_cross_validation_r2_score, get_cross_validation_RMSLE, get_RMSLE
from models_preparation import initialize_model
from data_visualization import plot, plot_distribution, missing_data_ratio, correlation_heatmap

 # Limit floats output to 2 decimal points
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))


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
def create_submission_file(model, x_test, x_test_ids) :
    predicted_price = model.predict(x_test)
    submission = pd.DataFrame()
    submission['Id'] = x_test_ids
    submission['SalePrice'] = np.expm1(predicted_price)
    submission.to_csv("submission.csv", index=False)

def main() :
    data_raw = load_dataset(path = "train.csv")
    # test.csv doesn't have the labels, can't be used for training, but can be for filling missing values
    data_test_raw = load_dataset(path = "test.csv")
    test_ids = data_test_raw['Id']
    ntrain = data_raw.shape[0]

    all_data = pd.concat((data_raw, data_test_raw)).reset_index(drop=True)
    print("all_data size is : {}".format(all_data.shape))

    print("missing data before cleaning : ", missing_data_ratio(all_data))
    data_cleaned = clean_data(all_data)
    print("missing data after cleaning : ", missing_data_ratio(data_cleaned))

    data_cleaned_encoded = encode_to_categorical(data_cleaned)
    data_cleaned_encoded.drop('Id', axis=1, inplace = True)

    # Separate back the two datasets to get the one with the target variable
    train_data_cleaned_encoded = data_cleaned_encoded[:ntrain]
    test_data_cleaned_encoded = data_cleaned_encoded[ntrain:]
    test_data_cleaned_encoded.drop('SalePrice', axis=1, inplace = True)

    # Split train and test data
    df_train, df_test = train_test_split(train_data_cleaned_encoded, test_size=0.2)
    clean_data_train(df_train)

    x = train_data_cleaned_encoded.drop("SalePrice", axis=1)
    x_train = df_train.drop("SalePrice", axis=1)
    x_test = df_test.drop("SalePrice", axis=1)

    y = train_data_cleaned_encoded.SalePrice
    y_train = df_train.SalePrice
    y_test = df_test.SalePrice

    # plot_distribution(df_train)
    # ==> Skew is corrected
    
    #  -------- Model Comparison --------
    print("------- Linear Regression -------")
    regression_model = initialize_model("linear_regression")
    regression_model.fit(x_train, y_train)
    y_pred = regression_model.predict(x_test)
    print(y_test[:5])
    print(y_pred[:5])
    print("Cross validation R2 : ", get_cross_validation_r2_score(regression_model, x, y))
    print("Cross validation neg mean squared log error : ", get_cross_validation_RMSLE(regression_model, x, y))
    print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred)))

    print("------- Random Forest -------")
    # random_forest_model = initialize_model("random_forest", parameters = get_random_forest_best_parameters_by_grid_search(x, y))
    random_forest_model = initialize_model("random_forest", parameters = {"max_depth" : 45, "n_estimators" : 100})
    random_forest_model.fit(x_train, y_train)
    y_pred = random_forest_model.predict(x_test)
    print(y_test.T[:5])
    print(np.expm1(y_pred[:5]))
    
    print("Cross validation R2 : ", get_cross_validation_r2_score(random_forest_model, x, y))
    print("Cross validation RMSLE : ", get_cross_validation_RMSLE(random_forest_model, x, y))
    # np.expm1 to apply the reverse transformation applied to y_train to remove skew
    print("RMSLE : ", get_RMSLE(y_true = y_test, y_pred = np.expm1(y_pred)))
    print("End")

    create_submission_file(model = random_forest_model, x_test = test_data_cleaned_encoded, x_test_ids = test_ids)

if __name__== "__main__":
  main()