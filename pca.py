from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes._subplots import Axes
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler


def clean_import(file: str) -> Tuple[pd.Index, list, pd.DataFrame]:
    """Minimally processes data from file"""
    data = pd.read_csv(file)
    data.dropna(inplace=True)
    data.columns = data.columns.str.lower()
    data.set_index('country', inplace=True)
    data.columns = [x.replace(' ', '_') for x in data.columns]
    return data.index, data.columns, data


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    """Scale the data"""
    row_idx = data.index
    col_idx = data.columns
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data))
    data.index, data.columns = row_idx, col_idx
    return data


def annotate_points(data: pd.DataFrame):
    """Annotate plotted points"""
    for point in data.index:
        label = point
        plt.annotate(
            label,
            (data.loc[point][0], data.loc[point][1]),
            textcoords="offset points",  # how to position the text
            xytext=(-10, -12),  # distance from text to points (x,y)
            ha='left'  # horizontal alignment can be left, right or center
        )


def plot_most_correlated(data: pd.DataFrame) -> None:
    """Plot the two features with highest correlation"""
    correlation = data.corr()
    mask = correlation < 1.
    sorted_correlation = correlation[mask].abs().unstack().sort_values(ascending=False)
    x_name = sorted_correlation.index[0][0]
    y_name = sorted_correlation.index[0][1]
    x = data[x_name]
    y = data[y_name]
    # fig, ax = plt.subplots()
    plt.scatter(x, y)
    plt.title('Highest Correlated Features')
    plt.xlabel(f'{x_name} Consumption')
    plt.ylabel(f'{y_name} Consumption')
    annotate_points(pd.concat([x, y], axis=1))
    plt.show()


if __name__ == '__main__':
    file = 'homework1/data/food-consumption.csv'
    country_idx, column_idx, data = clean_import(file)
    sc_data = scale_data(data)

    plot_most_correlated(data)

    pca = PCA()
    pca.fit(data)

    pca_2= PCA(n_components=2)
    projected = pca_2.fit_transform(data)

    plt.scatter(projected[:, 0], projected[:, 1])
    plt.title('Projected Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    pc_1 = pca.components_[0, :]
    pc_2 = pca.components_[1, :]

    projected = pd.DataFrame(projected, index=country_idx, columns=['PC1', 'PC2'])
    sns.scatterplot(x='PC1', y='PC2', data=projected, markers=data.index)
    annotate_points(projected)
    plt.show()

    plt.bar(column_idx, pca.components_[0, :])
    plt.xticks(rotation=70)
    plt.show()




