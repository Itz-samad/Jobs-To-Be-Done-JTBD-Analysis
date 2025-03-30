import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

def run(df):
    st.header("K-means Clustering")
    # Drop the Story_ID column (non-numeric data)
    X = df.drop(columns=["Story_ID"])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    k = 4  # Define the number of clusters
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    
    # Add cluster labels to the dataframe
    df["Cluster"] = labels
    st.write("Clustered Successfully")    
    return X_pca, labels, kmeans




def plot_kmeans_scatter(X_pca, labels, kmeans):

   
    fig, ax = plt.subplots(figsize=(10, 8))
    # Scatter plot of data points colored by their cluster labels
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=100, ax=ax)
    
    # Plot the centroids
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='x', s=200, label='Centroids')
    
    ax.set_title("Scatter Plot After Clustering (KMeans)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    
    # Render the figure in the Streamlit app
    st.pyplot(fig)