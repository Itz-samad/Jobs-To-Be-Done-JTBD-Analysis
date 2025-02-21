import streamlit as st
import pandas as pd

# Import your algorithms (adjust paths as needed)
from algorithms import Ward, HDBSCAN, K_means, NNA

def main():
    st.title("Jobs To Be Done (JTBD) Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Algorithm selection with a default option
    algorithm = st.selectbox(
        "Choose Algorithm",
        options=[
            '--Select an algorithm--',
            'Ward Method',
            'HDBSCAN',
            'K-means Clustering',
            'NNA Analysis'
        ]
    )
    
    # Submit button to process the file and algorithm selection
    if st.button("Submit"):
        # Check if the user has selected a valid algorithm
        if algorithm == '--Select an algorithm--':
            st.error("Please select an algorithm before submitting.")
        elif uploaded_file is not None:
            # Read the dataset
            df = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded!")
            
            st.subheader("Raw Data Preview")
            st.dataframe(df)
            
            # Process based on algorithm selection
            if algorithm == "Ward Method":
                data, link_mat = Ward.run(df)
                Ward.dendogram_plotting(data, link_mat)
            elif algorithm == "HDBSCAN":
                X, data, labels, CLusterer = HDBSCAN.run(df)
                HDBSCAN.dendogram_plotting(X, data, labels, CLusterer)
            elif algorithm == "K-means Clustering":
                X, data, kmesns = K_means.run(df)
                K_means.dendogram_plotting(X, data, kmesns)
            elif algorithm == "NNA Analysis":
                X, data, indices = NNA.run(df)
                NNA.dendogram_plotting(X, data, indices)
        else:
            st.error("Please upload a CSV file before submitting.")

if __name__ == "__main__":
    main()
