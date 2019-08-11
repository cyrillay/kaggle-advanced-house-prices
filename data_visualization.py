#!/usr/bin/python3

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import norm, skew

def plot(x, y) :
    fig, ax = plt.subplots()
    ax.scatter(x = x, y = y)
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()


def plot_distribution(house_dataset) : 
    sns.distplot(house_dataset['SalePrice'] , fit=norm);

    # Get the fitted parameters used by the function
    (mean, variance) = norm.fit(house_dataset['SalePrice'])
    print("Distribution mean =", mean)
    print("Distribution variance =", variance)

    # Plot the distribution
    plt.legend(['Distribution : mean={:.2f} and variance={:.2f} )'.format(mean, variance)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    # QQ-plot
    fig = plt.figure()
    res = stats.probplot(house_dataset['SalePrice'], plot=plt)
    plt.show()
    # ==> skewed on the right

def missing_data_ratio(house_dataset) : 
    na_count_per_column = house_dataset.isnull().sum()
    percentage_missing_per_column = (na_count_per_column / len(house_dataset)) * 100
    df = percentage_missing_per_column.loc[(percentage_missing_per_column!=0)].sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :df})
    return missing_data.head(20)

def correlation_heatmap(house_dataset) :
    corrmat = house_dataset.corr()
    sns.heatmap(corrmat, 
        xticklabels=corrmat.columns,
        yticklabels=corrmat.columns)
    plt.show()