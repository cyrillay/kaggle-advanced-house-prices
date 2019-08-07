#!/usr/bin/python3
# Author : Cyril Lay
# Kaggle competition - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard
# The goal of machine learning script is train and evaluate different models on the Kaggle dataset.
# The dataset is composed of various features on housd

import numpy as np
import pandas as pd
from sklearn import metrics 
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
import json


# ------------------- Pre processing ------------------- #

def load_dataset(path):
    input = pd.read_csv(path, header = 0)
    return pd.DataFrame(input)

def clean_data(house_dataset) :
    # Replace Nan with None String
    df1 = house_dataset.replace(np.nan, 'None', regex=True)
    # Replace None with zero in integer columns
    df1['LotFrontage'].replace('None', 0, regex=True, inplace = True)
    df1['MasVnrArea'].replace('None', 0, regex=True, inplace = True)
    df1['BsmtFinSF1'].replace('None', 0, regex=True, inplace = True)
    df1['BsmtFinSF2'].replace('None', 0, regex=True, inplace = True)
    df1['BsmtUnfSF'].replace('None', 0, regex=True, inplace = True)
    df1['TotalBsmtSF'].replace('None', 0, regex=True, inplace = True)
    df1['BsmtFullBath'].replace('None', 0, regex=True, inplace = True)
    df1['BsmtHalfBath'].replace('None', 0, regex=True, inplace = True)
    df1['GarageCars'].replace('None', 0, regex=True, inplace = True)
    df1['GarageArea'].replace('None', 0, regex=True, inplace = True)
    # TODO : Find better encoding - Shouldn't do on GarageYearBuilt because 0 as a year doesn't make sense
    df1['GarageYrBlt'].replace('None', 0, regex=True, inplace = True)
    return df1

def encode_to_categorical(house_dataset) :
    # Encode the strings with numerical value
    # TODO : don't encode everything : encode based on colum type ?
    encoder = preprocessing.LabelEncoder()
    # labels = house_dataset['SalePrice']
    X = house_dataset.loc[:, ~house_dataset.columns.isin(['Id', 'SalePrice'])]
    X_encoded = X.apply(encoder.fit_transform)
    X_encoded['Id'] = house_dataset.Id
    house_dataset.loc[:, ~house_dataset.columns.isin(['Id', 'SalePrice'])] = X_encoded
    return house_dataset

# --------------------- Model --------------------- #
def initialize_model(model_type, parameters = {}) :
    if (model_type.startswith("linear")) :
        return linear_model.LinearRegression(normalize = True)
    elif (model_type.startswith("random")) : 
        return RandomForestRegressor(max_depth=parameters["max_depth"], n_estimators=parameters["n_estimators"], random_state=False, verbose=False)

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

# ------------------- Visualization ------------------- #

# plt.scatter(X.GrLivArea, Y,color='g')
# plt.plot(X.GrLivArea, regression_model.predict(X),color='k')
# plt.show()

def plot_predictions(X, Y, Y_PRED) :
    fig, ax = plt.subplots()
    ax.scatter(x = X, y = Y)
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('X', fontsize=13)
    plt.plot(X, Y_PRED,color='k')
    plt.show()

# ------------------- Submission ------------------- #
# df_test_raw = load_dataset(path = "test.csv")
# cleaned_test_data = clean_data(house_dataset = df_test_raw)
# encoded_test_data = encode_to_categorical(cleaned_test_data)

# predictedPrice = regression_model.predict(encoded_test_data)

# submission = pd.DataFrame()
# submission['Id'] = encoded_test_data.Id
# submission['SalePrice'] = predictedPrice

# submission.to_csv("submission.csv", index=False)

# print(submission)


def main() :
    data_raw = load_dataset(path = "train.csv")
    data_cleaned = clean_data(data_raw)
    data_cleaned_encoded = encode_to_categorical(data_cleaned)

    # -------- Data exploration --------
    df_train['SalePrice'].describe()

    df_train, df_test = train_test_split(data_cleaned_encoded, test_size=0.2)

    x = data_cleaned_encoded.drop("SalePrice", axis=1)
    x_train = df_train.drop("SalePrice", axis=1)
    x_test = df_test.drop("SalePrice", axis=1)

    y = data_cleaned_encoded.SalePrice
    y_train = df_train.SalePrice
    y_test = df_test.SalePrice
    
    #  -------- Model Comparison --------
    print("------- Linear Regression -------")
    regression_model = initialize_model("linear_regression")
    regression_model.fit(x_train, y_train)
    y_pred = regression_model.predict(x_test)
    print(y_test[:5])
    print("pred = ")
    print(y_pred[:5])
    print("Cross validation MSE : ", np.absolute(cross_val_score(regression_model, x, y, cv=10, scoring="neg_mean_squared_error")), " (5 repetitions) ")
    print("Cross validation R2 : ", np.absolute(cross_val_score(regression_model, x, y, cv=10, scoring="r2")), " (5 repetitions) ")
    error = np.sqrt(mean_squared_log_error(y_true = y_test, y_pred = y_pred))
    print("Root Mean Squared Logarithmic error = ", error, "\nR2 = ", r2_score(y_true = y_test, y_pred = y_pred))

    print("------- Random Forest -------")
    random_forest_model = initialize_model("random_forest", parameters = get_random_forest_best_parameters_by_grid_search(x, y))
    random_forest_model.fit(x_train, y_train)
    y_pred = random_forest_model.predict(x_test)
    print(y_test[:5])
    print(y_pred[:5])
    print("Cross validation MSE : ", np.absolute(cross_val_score(random_forest_model, x, y, cv=10, scoring="neg_mean_squared_error")), " (5 repetitions) ")
    print("Cross validation R2 : ", np.absolute(cross_val_score(random_forest_model, x, y, cv=10, scoring="r2")), " (5 repetitions) ")
    error = np.sqrt(mean_squared_log_error(y_true = y_test, y_pred = y_pred))
    print("Root Mean Squared Logarithmic error = ", error, ",\nR2 = ", r2_score(y_true = y_test, y_pred = y_pred))

    print("End")

if __name__== "__main__":
  main()

# TODO : add CV for Root Mean Squared Logarithmic Error (not built-in in sklearn)
