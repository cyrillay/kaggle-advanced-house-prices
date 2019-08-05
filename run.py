
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns

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

df_train_raw = load_dataset(path = "train.csv")
cleaned_train_data = clean_data(house_dataset = df_train_raw)
encoded_train_data = encode_to_categorical(cleaned_train_data)

# Convert categorical variable into dummy/indicator variables.
# pd.get_dummies(df)

# --------------------- Model --------------------- #
# Linear regression
regression_model = linear_model.LinearRegression(normalize = True)
df = encoded_train_data = encode_to_categorical(cleaned_train_data)
X = df.drop("SalePrice", axis=1)
Y = df.SalePrice
regression_model.fit(X, Y)

# Random Forest
# TODO


# ------------------- Validation ------------------- #

# TODO : Cross validation

# ------------------- Visualization ------------------- #

# plt.scatter(X.Id, Y,color='g')
# plt.plot(X.Id, regression_model.predict(X),color='k')
# plt.show()

# ------------------- Submission ------------------- #
df_test_raw = load_dataset(path = "test.csv")
cleaned_test_data = clean_data(house_dataset = df_test_raw)
encoded_test_data = encode_to_categorical(cleaned_test_data)

predictedPrice = regression_model.predict(encoded_test_data)

submission = pd.DataFrame()
submission['Id'] = encoded_test_data.Id
submission['SalePrice'] = predictedPrice

submission.to_csv("submission.csv", index=False)

print(submission)
