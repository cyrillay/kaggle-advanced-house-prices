#!/usr/bin/python3
from sklearn import preprocessing
import numpy as np

# Handles missing values, or irrelevant features
def clean_data(house_dataset) :
    # No outliers to delete
    df = house_dataset
    df["PoolQC"].fillna("None", inplace = True) # NA : no pool
    df["MiscFeature"].fillna("None", inplace = True) # NA : no miscellaneous feature
    df["Alley"].fillna("None", inplace = True) # NA : no Alley in the house
    df["Fence"].fillna("None", inplace = True) # NA : no fence
    df["FireplaceQu"].fillna("None", inplace = True) # NA : no fireplace

    # For the number columns, we will try to replace the missing values in different ways.
    # We assume that the area of each street connected to the house property has a similar area to other houses in its neighborhood,
    # so we fill the missing values by the median LotFrontage of the neighborhood.
    # TODO : add randomness ?
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    # NA : no garage
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        df[col] = df[col].fillna('None')
    # NA : no garage : no cars
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0) # TODO : find a better way to replace the GarageYrBlt
    # NA : no basement => 0 bathrooms, 0 square feet
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        df[col] = df[col].fillna(0)
    # NA : no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')
    # NA : no masonry veneer (wall type) => 0 square feet
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # 'RL' is by far the most common value (>99%). So we can fill in missing values with 'RL'
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

    # For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA.
    # Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We remove it.
    # TODO : or remove the row ?
    df = df.drop(['Utilities'], axis=1)

    # Data description says NA means typical
    df["Functional"] = df["Functional"].fillna("Typ")

    # It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
    # TODO : or remove the row ?
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    # Same
    df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])
    # Same
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

    # Fill with most frequent
    # TODO
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

    # We assume Na means no building class. We replace missing values with None
    df['MSSubClass'] = df['MSSubClass'].fillna("None")

    return df

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
    updated_column = house_dataset_train.loc[:, 'SalePrice'].copy().apply(lambda x: np.log1p(x))
    house_dataset_train.loc[:, 'SalePrice'] = updated_column
    # Note : Must apply reverse transformation to prediction : exp(y_pred) - 1 OR np.expm1(y_pred)

# It's better to call this method with the full dataset, both train and test
# Even if test doesn't have the target variable and can't be used for training, 
# its features can help deduce the missing values
# def fill_missing_data(house_dataset) : 
