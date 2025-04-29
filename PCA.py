import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import seaborn as sns


def run_pca(df, target_columns, target=None, variance_threshold=0.99):
    """
    Performs PCA on the selected features from the dataframe.

    Parameters:
    - df: Pandas DataFrame with your data
    - target_columns: List of column names to include in PCA
    - target: Optional array or Series for coloring in 2D plot
    - variance_threshold: How much total variance to retain (default 99%)

    Returns:
    - pca_model: The fitted PCA object
    - X_pca: Transformed data
    - components: PCA loadings
    """
    X = df[target_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca, pca.components_


def plot_pca_2d(X_pca, target=None):
    plt.figure(figsize=(8, 6))
    if target is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Target')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_3d(X_pca):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c='blue', alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA')
    plt.tight_layout()
    plt.show()


def plot_variance(pca):
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
    plt.axhline(y=99, color='r', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_variance_bar(pca):
    variance_explained = pca.explained_variance_ratio_ * 100
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(variance_explained) + 1), variance_explained)
    plt.axhline(y=1, color='r', linestyle='--', label='1% Threshold')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Component')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loadings(components, feature_names):
    plt.figure(figsize=(10, 7))
    for i in range(len(feature_names)):
        plt.arrow(0, 0, components[0, i], components[1, i],
                  head_width=0.02, head_length=0.05, fc='blue', ec='black')
        plt.text(components[0, i] * 1.15, components[1, i] * 1.15, feature_names[i],
                 color='black', ha='center', va='center')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Loadings')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


from pca import run_pca, plot_pca_2d, plot_pca_3d, plot_variance, plot_variance_bar, plot_loadings

# Suppose df is your uploaded DataFrame
target_columns = ['SiO2', 'Al2O3', 'MgO', 'TiO2', 'Cr2O3', 'CaO', 'MnO', 'Na2O', 'K2O', 'FeO']
target = df['SomeClassColumn'] if 'SomeClassColumn' in df else None

pca_model, X_pca, components = run_pca(df, target_columns, target=target)

# Then call the plot functions
plot_pca_2d(X_pca, target=target)
plot_pca_3d(X_pca)
plot_variance(pca_model)
plot_variance_bar(pca_model)
plot_loadings(components, target_columns)
