import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon
import numpy as np

def run(df):
    st.header("K-means Clustering")
    # Drop the Story_ID column (non-numeric data)
    X = df.drop(columns=["Story_ID"])
    
    k = 4  # Define the number of clusters
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Add cluster labels to the dataframe
    df["Cluster"] = labels
    st.write("Clustered Successfuly")    
    return X, df, kmeans

def dendogram_plotting(X, df, kmeans):
    # Compute distances from each story to centroids
    distances = kmeans.transform(X)

    # Convert distances to probabilities (softmax)
    probabilities = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)

    # Step 2: Compute pairwise Jensen-Shannon divergence matrix
    n = len(X)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
         for j in range(n):
            distance_matrix[i, j] = jensenshannon(probabilities[i], probabilities[j])

    linkage_matrix = linkage(distance_matrix)




    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(linkage_matrix, labels=df["Story_ID"].values, leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Dendrogram using Ward's Method")
    ax.set_xlabel("Stories")
    ax.set_ylabel("Distance")
    st.pyplot(fig)




