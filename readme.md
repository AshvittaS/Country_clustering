# Country Development Clustering
Deployed link : [link] (https://countryclustering-jbqk8xjy2dsezwbqloyp4m.streamlit.app/)

This project builds an interactive Streamlit app for clustering countries based on economic and social indicators using a custom unsupervised learning pipeline.

## What makes this project unique

- **DBSCAN-based clustering with PCA preprocessing**: the app uses a trained `StandardScaler` + `PCA` pipeline, then applies `DBSCAN` to cluster countries in reduced-dimensional space.
- **Dynamic country prediction**: users can enter a new country's features and the app appends this sample to the dataset, scales and projects it with the saved pipeline, then re-runs DBSCAN to determine cluster membership.
- **Outlier detection support**: DBSCAN labels noise points as `-1`, and the app explicitly reports when a new country is classified as an outlier.
- **Similar-country lookup**: once a cluster is assigned, the app lists original countries from the dataset that belong to the same cluster.
- **Clustering evaluation and comparison**: the notebook compares multiple clustering approaches (KMeans, Agglomerative, DBSCAN) and uses silhouette analysis to validate the pipeline.
- **Reusable saved preprocessing objects**: the project persists `scaler.pkl` and `pca.pkl` so the Streamlit app can reproduce the exact transformation used during model development.

## Core pipeline

1. Load raw country indicator data from `Country-data.csv`.
2. Scale features with `StandardScaler`.
3. Reduce dimensionality using `PCA` while preserving ~95% variance.
4. Cluster the PCA-transformed data with `DBSCAN`.
5. Expose a Streamlit front end where users enter values and receive:
   - assigned cluster label
   - outlier warning if applicable
   - number of countries in the cluster
   - similar countries from the training data

## Key project files

- `app.py`: Streamlit app that accepts input indicators, applies the saved scaler/PCA, reruns DBSCAN, and displays results.
- `task.ipynb`: exploratory notebook with data loading, preprocessing, PCA analysis, clustering comparisons, and clustering interpretation.
- `Country-data.csv`: original country indicators dataset.
- `scaler.pkl` / `pca.pkl`: serialized preprocessing objects used by the app.
- `best_clustering_model.pkl`: placeholder for the selected best clustering algorithm from the notebook workflow.

## Why this approach is valuable

- DBSCAN is well-suited for country development data because it can identify natural clusters without requiring a fixed number of groups.
- PCA improves clustering quality by removing noise and redundant features before clustering.
- Saving preprocessing objects ensures the deployed app uses the same transformation pipeline as the analysis.
- The interactive UI makes the project practical for probing how new country metrics map to learned development clusters.

## Notes

- The app is intentionally built for unsupervised cluster assignment rather than supervised country classification.
- The notebook includes manual interpretation of cluster groups and comparisons across clustering algorithms.
