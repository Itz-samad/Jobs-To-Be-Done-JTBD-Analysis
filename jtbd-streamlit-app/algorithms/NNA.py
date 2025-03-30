import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

def run(df):
    st.header("Nearest Neighbours Analysis")
    # Drop the Story_ID column (non-numeric data)
    X = df.drop(columns=["Story_ID"])

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    
    # Applying nearest neighbors analysis
    n_neighbors = 3  # Number of nearest neighbors
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nearest_neighbors.fit(X_pca)

    # Compute the neighbor connectivity
    distances, indices = nearest_neighbors.kneighbors(X_pca)

    # Apply clustering using connectivity information
    # using DBSCAN to form clusters based on neighborhood density
    dbscan = DBSCAN(eps=distances.mean(), min_samples=n_neighbors, metric="euclidean")
    labels = dbscan.fit_predict(X_pca)
    
    # Add cluster labels to the dataframe
    df["Cluster"] = labels
    st.write("Clustered Successfully")    
    return X_pca, labels

# def dendogram_plotting(X, df, indices):
#     num_points = len(X)
#     connectivity_matrix = np.zeros((num_points, num_points))

#     for i in range(num_points):
#         for j in indices[i]:
#             connectivity_matrix[i, j] = 1
#             connectivity_matrix[j, i] = 1  # Symmetric matrix

#     # Convert Connectivity to Dissimilarity
#     dissimilarity_matrix = 1 - connectivity_matrix

#     # Generate Pseudo-Hierarchical Linkage
#     linkage_matrix = linkage(squareform(dissimilarity_matrix), method="average")

#     fig, ax = plt.subplots(figsize=(15, 8))
#     dendrogram(linkage_matrix, labels=df["Story_ID"].values, leaf_rotation=90, leaf_font_size=10)
#     ax.set_title("Dendrogram using NNA")
#     ax.set_xlabel("Stories")
#     ax.set_ylabel("Distance")
#     st.pyplot(fig)




def plot_knn_dbscan_scatter(X_pca, labels):

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a scatter plot
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 2], c=labels, cmap='viridis', s=50, 
                         alpha=0.7, edgecolors='k')
    
    # Add a colorbar to the figure
    fig.colorbar(scatter, ax=ax)

    
    
    ax.set_title("KNN-DBSCAN Clustering Scatter Plot")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2") 
    
    # Display the plot in Streamlit
    st.pyplot(fig)


