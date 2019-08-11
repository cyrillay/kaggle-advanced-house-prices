#!/usr/bin/python3
from sklearn import preprocessing
import numpy as np

def clean_data(house_dataset) :
    # Deleting outliers
    # No outliers
    # house_dataset = house_dataset.drop(house_dataset[(house_dataset['GrLivArea']>4000) & (house_dataset['SalePrice']<300000)].index)

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

# Training specific processing
def clean_data_train(house_dataset_train) : 
    # Apply log(1+x) to the target variable to remove the skew
    house_dataset_train["SalePrice"] = np.log1p(house_dataset_train["SalePrice"])
    # Note : Must apply reverse transformation to prediction : exp(y_pred) - 1 / np.expm1(y_pred)