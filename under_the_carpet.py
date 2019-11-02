import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn import cluster, datasets
from tqdm import tqdm
from umap import UMAP
from pynndescent import NNDescent
from fastcluster import single
from scipy.cluster.hierarchy import cut_tree, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn import metrics
from sklearn_pandas.dataframe_mapper import DataFrameMapper
from sklearn.decomposition import PCA
import random
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')
sns.set_style('white')


def plot_data(data, labels=None, title=None, legend=True, centroids=None):
    plt.figure(figsize=(7,5), dpi=100)
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels)

        unique_labels = list(sorted(set(x for x in labels if str(x) != '-1')))
        labels = np.array(list(map(str, labels)))
        if '-1' in labels:
            plt.scatter(data[labels=='-1', 0], data[labels=='-1', 1], s=5, c='black', alpha=0.5)


        for i, label in enumerate(unique_labels):
            label = str(label)
            plt.scatter(data[labels==label, 0], data[labels==label, 1], s=5, c='C{}'.format(i),
                        cmap='viridis', label=label, alpha=0.5)

            if centroids is not None:
                plt.scatter(centroids[i, 0], centroids[i, 1], s=250, c='white', marker='o')
                plt.scatter(centroids[i, 0], centroids[i, 1], s=100, marker='X')

        if legend:
            plt.legend(fontsize=10, markerscale=3, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.scatter(data[:, 0], data[:, 1], s=5, color='grey', alpha=0.5)

    if title:
        plt.title(title)
    plt.axis('off')


def load_mnist(size):
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
    sample_index = np.random.choice(list(range(len(X))), size=size, replace=False)
    digits_X = X[sample_index]
    digits_y = y[sample_index]
    return digits_X, digits_y


def load_dd(size):
    characters = pd.read_csv('characters.csv')
    characters = characters.sample(size, random_state=42)
    return characters

