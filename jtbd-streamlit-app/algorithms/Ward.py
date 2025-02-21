import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def run(df):
    st.header("Ward Method Clustering")
    # Drop the Story_ID column (non-numeric data)
    X = df.drop(columns=["Story_ID"])
    
    # Perform hierarchical clustering using Ward's method
    linkage_matrix = linkage(X, method="ward")
    num_clusters = 4  # Adjust as needed
    labels = fcluster(linkage_matrix, num_clusters, criterion="maxclust")
    
    # Add cluster labels to the dataframe
    df["Cluster"] = labels
    st.write("Clustered Successfuly")    
    return df, linkage_matrix

def dendogram_plotting(df, linkage_matrix):
   

    fig, ax = plt.subplots(figsize=(15, 8))
    dendrogram(linkage_matrix, labels=df["Story_ID"].values, leaf_rotation=90, leaf_font_size=10)
    ax.set_title("Dendrogram using Ward's Method")
    ax.set_xlabel("Stories")
    ax.set_ylabel("Distance")
    st.pyplot(fig)




