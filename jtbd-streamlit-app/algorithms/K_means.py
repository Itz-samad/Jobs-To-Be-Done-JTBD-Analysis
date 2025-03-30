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

# def dendogram_plotting(X, df, kmeans):
#     # Compute distances from each story to centroids
#     distances = kmeans.transform(X)

#     # Convert distances to probabilities (softmax)
#     probabilities = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)

#     # Step 2: Compute pairwise Jensen-Shannon divergence matrix
#     n = len(X)
#     distance_matrix = np.zeros((n, n))
#     for i in range(n):
#          for j in range(n):
#             distance_matrix[i, j] = jensenshannon(probabilities[i], probabilities[j])

#     linkage_matrix = linkage(distance_matrix)




#     fig, ax = plt.subplots(figsize=(15, 8))
#     dendrogram(linkage_matrix, labels=df["Story_ID"].values, leaf_rotation=90, leaf_font_size=10)
#     ax.set_title("Dendrogram using K-means Clustering")
#     ax.set_xlabel("Stories")
#     ax.set_ylabel("Distance")
#     st.pyplot(fig)




# def plot_kmeans_clusters(X, df, kmeans):
#     # Get cluster labels and centroids
#     labels = kmeans.labels_
#     centroids = kmeans.cluster_centers_
    
#     # Compute distances from each story to its cluster centroid
#     distances = np.linalg.norm(X - centroids[labels], axis=1)
    
#     # Sort stories by cluster and then by distance within cluster
#     sorted_indices = np.lexsort((distances, labels))
#     sorted_stories = df["Story_ID"].iloc[sorted_indices].values
#     sorted_distances = distances[sorted_indices]
#     sorted_labels = labels[sorted_indices]
    
#     # Create plot
#     fig, ax = plt.subplots(figsize=(15, 8))
#     bars = ax.bar(
#         x=range(len(sorted_stories)),
#         height=sorted_distances,
#         color=plt.cm.tab10(sorted_labels),
#         alpha=0.7
#     )
    
#     # Formatting
#     ax.set_title("K-means Cluster Cohesion")
#     ax.set_xlabel("Stories (Sorted by Cluster and Distance to Centroid)")
#     ax.set_ylabel("Distance to Cluster Centroid")
#     ax.set_xticks(range(len(sorted_stories)))
#     ax.set_xticklabels(sorted_stories, rotation=90, fontsize=8)
    
#     # Add cluster separation lines
#     cluster_changes = np.where(np.diff(sorted_labels))[0]
#     for change in cluster_changes:
#         ax.axvline(change + 0.5, color='black', linestyle='--', linewidth=0.8)
    
#     # Add legend
#     unique_labels = np.unique(sorted_labels)
#     handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(l)) for l in unique_labels]
#     ax.legend(handles, [f"Cluster {l}" for l in unique_labels])
    
#     st.pyplot(fig)
#     plt.close()




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