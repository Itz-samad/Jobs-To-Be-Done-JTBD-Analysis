import streamlit as st
import pandas as pd

# Import your algorithms (adjust paths as needed)
from algorithms import Ward, HDBSCAN, K_means, NNA

def main():
    st.title("Jobs To Be Done (JTBD) Analysis")
    
    # File upload: allow CSV, Excel, and TXT files
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "txt"])
    
    # Delimiter selection using radio buttons with a placeholder (only used for CSV/TXT files)
    delim_options = {
        "comma": ",",
        "semicolon": ";",
        "tab": "\t",
        "space": " "
    }
    delim_choice = st.radio(
        "Select Seperator (for CSV/TXT files)", 
        options=["--None--"] + list(delim_options.keys())
    )
    
    # Checkbox: Does the dataset have a header row?
    header_option = st.checkbox("Dataset has header?", value=True)
    
    # Checkbox: Is the first column the ID column?
    id_column_option = st.checkbox("First column is ID column?", value=True)
    
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
    
    # Submit button to process the file and options
    if st.button("Submit"):
        # Validate algorithm selection
        if algorithm == '--Select an algorithm--':
            st.error("Please select an algorithm before submitting.")
            return
        
        # Validate file upload
        if uploaded_file is None:
            st.error("Please upload a dataset before submitting.")
            return
        
        # Determine the file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # For CSV/TXT, validate that a delimiter has been selected
        if file_extension in ['csv', 'txt'] and delim_choice == "--None--":
            st.error("Please select a Seperator for CSV/TXT files.")
            return
        
        try:
            # Read the dataset based on file type and header option
            if file_extension in ['csv', 'txt']:
                delimiter = delim_options[delim_choice]
                if header_option:
                    df = pd.read_csv(uploaded_file, delimiter=delimiter)
                else:
                    df = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)
            elif file_extension in ['xlsx', 'xls']:
                if header_option:
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, header=None)
            else:
                st.error("Unsupported file format")
                return
            
            # Handle ID column logic:
            if header_option:
                if id_column_option:
                    # Rename the first column to "Story_ID"
                    df = df.rename(columns={df.columns[0]: "Story_ID"})
                else:
                    # Validate that "Story_ID" exists in the header
                    if "Story_ID" not in df.columns:
                        st.error("The uploaded file does not contain a 'Story_ID' column. "
                                 "Please verify your file or check 'First column is ID column'.")
                        return
            else:
                # No headers: assign generic names
                num_cols = df.shape[1]
                if id_column_option:
                    # First column becomes "Story_ID"
                    columns = ["Story_ID"] + [f"Column_{i}" for i in range(1, num_cols)]
                    df.columns = columns
                else:
                    # Generic column names
                    df.columns = [f"Column_{i}" for i in range(num_cols)]
            
            st.success("File successfully uploaded!")
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            # Process based on algorithm selection
            if algorithm == "Ward Method":
                data, link_mat = Ward.run(df)
                Ward.dendogram_plotting(data, link_mat)
            elif algorithm == "HDBSCAN":
                X, data, labels, clusterer = HDBSCAN.run(df)
                HDBSCAN.dendogram_plotting(X, data, labels, clusterer)
            elif algorithm == "K-means Clustering":
                X, data, kmesns = K_means.run(df)
                K_means.dendogram_plotting(X, data, kmesns)
            elif algorithm == "NNA Analysis":
                X, data, indices = NNA.run(df)
                NNA.dendogram_plotting(X, data, indices)
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    main()
