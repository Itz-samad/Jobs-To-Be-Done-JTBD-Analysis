import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from scipy.spatial.distance import jensenshannon
import numpy as np
from sklearn.decomposition import PCA

def run(df):
    st.header("HDBSCAN Clustering")
    # Drop the Story_ID column (non-numeric data)   
    X = df.drop(columns=["Story_ID"])

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Initialize and fit HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean")
    
    labels = clusterer.fit_predict(X_pca)

    
    # Optionally add cluster labels to your original DataFrame if desired
    df["Cluster"] = labels
    st.write("Clustered Successfully")    
    return X_pca, labels


def plot_hdbscan_scatter(X_pca, labels):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the scatter plot on the axes
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 2], c= labels,
                        cmap='viridis', s=50, alpha=0.7, edgecolors='k')

    # Add a colorbar to the figure (linked to the scatter plot)
    fig.colorbar(scatter, ax=ax)

    # Set the title and axis labels
    ax.set_title('HDBSCAN Clustering')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Display the figure in your Streamlit app
    st.pyplot(fig)