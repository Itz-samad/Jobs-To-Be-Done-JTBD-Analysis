import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
def run(df):
    st.header("Nearest Neighbours Analysis")
    # Drop the Story_ID column (non-numeric data)
    X = df.drop(columns=["Story_ID"])
    
    # Applying nearest neighbors analysis
    n_neighbors = 3  # Number of nearest neighbors
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nearest_neighbors.fit(X)

    # Compute the neighbor connectivity
    distances, indices = nearest_neighbors.kneighbors(X)

    # Apply clustering using connectivity information
    # using DBSCAN to form clusters based on neighborhood density
    dbscan = DBSCAN(eps=distances.mean(), min_samples=n_neighbors, metric="euclidean")
    labels = dbscan.fit_predict(X)
    
    # Add cluster labels to the dataframe
    df["Cluster"] = labels
    st.write("Clustered Successfuly")    
    return X, df, indices

def dendogram_plotting(X, df, indices):
    num_points = len(X)
    connectivity_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in indices[i]:
            connectivity_matrix[i, j] = 1
            connectivity_matrix[j, i] = 1  # Symmetric matrix

    # Convert Connectivity to Dissimilarity
    dissimilarity_matrix = 1 - connectivity_matrix

    # Generate Pseudo-Hierarchical Linkage
    linkage_matrix = linkage(squareform(dissimilarity_matrix), method="average")

    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(linkage_matrix, labels=df["Story_ID"].values, leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Dendrogram using NNA")
    ax.set_xlabel("Stories")
    ax.set_ylabel("Distance")
    st.pyplot(fig)




