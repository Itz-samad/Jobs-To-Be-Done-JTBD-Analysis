import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from scipy.spatial.distance import jensenshannon
import numpy as np

def run(df):
    st.header("HDBSCAN Clustering")
    # Drop the Story_ID column (non-numeric data)
    X = df.drop(columns=["Story_ID"])

    # Initialize and fit HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean")
    labels = clusterer.fit_predict(X)
    
    # Optionally add cluster labels to your original DataFrame if desired
    df["Cluster"] = labels
    st.write("Clustered Successfuly")    
    return X, df, labels, clusterer

def dendogram_plotting(X, df, labels, clusterer):
    probabilities = clusterer.probabilities_

    # Construct Similarity Matrix Using HDBSCAN Clusters
    num_stories = len(X)
    similarity_matrix = np.zeros((num_stories, num_stories))

    for i in range(num_stories):
        for j in range(num_stories):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Perfect similarity with itself
            elif labels[i] == labels[j] and labels[i] != -1:  # Same cluster, exclude noise
                # Use product of membership probabilities as similarity
                similarity_matrix[i, j] = probabilities[i] * probabilities[j]
            else:
                similarity_matrix[i, j] = 0.0  # Different clusters or noise

    # Convert similarity to dissimilarity (1 - similarity)
    dissimilarity_matrix = 1 - similarity_matrix

    # Generate Pseudo-Hierarchical Linkage
    linkage_matrix = linkage(dissimilarity_matrix, method="average")

    fig, ax = plt.subplots(figsize=(20, 13))
    dendrogram(linkage_matrix, labels=df["Story_ID"].values, leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Dendrogram using Ward's Method")
    ax.set_xlabel("Stories")
    ax.set_ylabel("Distance")
    st.pyplot(fig)




