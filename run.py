from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier 
from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns

input_file_test = "test.csv"
input = pd.read_csv(input_file_test, header = 0)
dfMat = input.values
df_test = pd.DataFrame(input)

input_file = "train.csv"
input = pd.read_csv(input_file, header = 0)
dfMat = input.values
df = pd.DataFrame(input)

# ------------------- Pre processing ------------------- #

# Replace Nan with None String
df1 = df.replace(np.nan, 'None', regex=True)
# Replace None with zero in integer columns
df1['LotFrontage'].replace('None', 0, regex=True, inplace = True)
df1['MasVnrArea'].replace('None', 0, regex=True, inplace = True)
# TODO : Find better encoding - Shouldn't do on GarageYearBuilt because 0 as a year doesn't make sense
df1['GarageYrBlt'].replace('None', 0, regex=True, inplace = True)

# Encode the strings with numerical value
# TODO : don't encode everything : encode based on colum type ?
le = preprocessing.LabelEncoder()
df2 = df1.drop("SalePrice", axis = 1).apply(le.fit_transform)
df2['SalePrice'] = df1['SalePrice']

# Convert categorical variable into dummy/indicator variables.
# pd.get_dummies(df)

# Linear regression
regression_model = linear_model.LinearRegression()
X = df2.drop("SalePrice", axis=1)
Y = df2.SalePrice
regression_model.fit(X, Y)

plt.scatter(X, Y,color='g')
plt.plot(X, regression_model.predict(X),color='k')

plt.show()