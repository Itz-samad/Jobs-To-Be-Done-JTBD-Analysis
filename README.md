# Jobs-To-Be-Done-(JTBD)-Analysis

This project is dedicated to analyzing and comparing different hierarchical clustering algorithms on the same dataset derived from Jobs-To-Be-Done (JTBD) interviews. Each branch of the repository implements a specific clustering algorithm, allowing for a detailed exploration of their behavior and performance. Except the jtbd-streamlit-app branch, which holds the streamlit app 

## Overview
The following clustering algorithms are implemented in separate branches:

1. **Ward Method (Hierarchical Clustering)**: Uses the Ward linkage method to minimize intra-cluster variance.
2. **K-Means Clustering**: Partitions data into \(k\) clusters by minimizing within-cluster variance.
3. **HDBSCAN Algorithm**: A density-based clustering approach for handling clusters of varying densities.
4. **Nearest Neighbors Analysis (NNA)**: Groups data points based on proximity using nearest neighbors.

Each branch contains the implementation of the respective algorithm, as well as preprocessing steps and visualizations.

## Purpose
The goal of this project is to compare these clustering techniques to identify the most suitable algorithm for analyzing JTBD interview data. By clustering responses, we aim to uncover patterns and insights that can help inform decision-making and strategy development.

## Repository Structure
- **Main Branch**: Contains the README and general information about the project.
- **Ward-Method-(Algorithm)**: Implements hierarchical clustering using the Ward Method. Two Dendogram Ploted using the Algorithm (one Ploted from dataset that has resistance included in it's dataset, and the other doesn't) and two scatter plot for Visualization 

- **K-Means-Clustering**: Focuses on K-Means Clustering. Two Dendogram Ploted using the Algorithm (one Ploted from dataset that has resistance included in it's dataset, and the other doesn't) and two scatter plot for Visualization 

- **HDBSCAN-Algorithm**: Implements the HDBSCAN algorithm. Four Dendogram Ploted using the Algorithm (Two Ploted from dataset that has resistance included in it's dataset, and the others doesn't) and two scatter plot for Visualization 

- **Nearest-Neighbors-Analysis-(NNA)**: Uses a nearest neighbors approach. Two Dendogram Ploted using the Algorithm (one Ploted from dataset that has resistance included in it's dataset, and the other doesn't) and two scatter plot for Visualization 

## Features
- **Dataset**: The dataset used in this project comes from Jobs-To-Be-Done interviews, formatted as a CSV file.
- **Customizable Analysis**: Each branch allows for tweaking parameters specific to the algorithm.
- **Visualization**: Each implementation generates visualizations such as dendrograms, cluster scatter plots, or heatmaps to interpret results.

## Requirements
- Python 3.x
- Required Python libraries (install via pip):
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy` (for Ward Method and other hierarchical clustering)
  - `hdbscan` (for HDBSCAN Algorithm)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Jobs-To-Be-Done-JTBD-Analysis.git
   cd Jobs-To-Be-Done-JTBD-Analysis
   ```

2. Checkout the branch of the algorithm you wish to explore:
   ```bash
   git checkout <branch-name>
   ```
   Replace `<branch-name>` with one of:
   - `Ward-Method-(Algorithm)`
   - `K-Means-Clustering`
   - `HDBSCAN-Algorithm`
   - `Nearest-Neighbors-Analysis-(NNA)`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook and follow the instructions:
   ```bash
   jupyter notebook
   ```

## Comparison Metrics
The following metrics are used to compare the algorithms:
- **Intra-Cluster Variance**: Measures compactness within clusters.
- **Silhouette Score**: Evaluates how well clusters are separated.
- **Cluster Interpretability**: Assesses the clarity of patterns uncovered by the algorithm.

## Results
Comparative results for each algorithm will be summarized in the main branch as the analysis progresses.

## Contributing
Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch.
- Submit a pull request with your improvements or new analysis.

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments


