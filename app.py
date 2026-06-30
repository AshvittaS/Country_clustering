import streamlit as st
import pandas as pd
import joblib
from sklearn.cluster import DBSCAN

# ----------------------------
# Load scaler and PCA
# ----------------------------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ----------------------------
# Load original dataset
# ----------------------------
df = pd.read_csv("Country-data.csv")

st.set_page_config(page_title="Country Development Clustering", layout="wide")

st.title("🌍 Country Development Clustering (DBSCAN + PCA)")

st.write("Enter the country's economic and social indicators.")

col1, col2 = st.columns(2)

with col1:
    child_mort = st.number_input("Child Mortality", value=20.0)
    exports = st.number_input("Exports (% GDP)", value=40.0)
    health = st.number_input("Health (% GDP)", value=6.0)
    imports = st.number_input("Imports (% GDP)", value=45.0)
    income = st.number_input("Income", value=10000.0)

with col2:
    inflation = st.number_input("Inflation", value=5.0)
    life_expec = st.number_input("Life Expectancy", value=70.0)
    total_fer = st.number_input("Total Fertility Rate", value=2.5)
    gdpp = st.number_input("GDP per Capita", value=5000.0)

if st.button("Predict Cluster"):

    # New sample
    new_country = pd.DataFrame({
        "child_mort": [child_mort],
        "exports": [exports],
        "health": [health],
        "imports": [imports],
        "income": [income],
        "inflation": [inflation],
        "life_expec": [life_expec],
        "total_fer": [total_fer],
        "gdpp": [gdpp]
    })

    # Combine with existing data
    X = df.drop("country", axis=1)
    X = pd.concat([X, new_country], ignore_index=True)

    # Scale
    X_scaled = scaler.transform(X)

    # PCA
    X_pca = pca.transform(X_scaled)

    # Run DBSCAN again
    dbscan = DBSCAN(eps=1.2, min_samples=5)
    labels = dbscan.fit_predict(X_pca)

    # Cluster of the new country
    cluster = labels[-1]

    st.success(f"Assigned Cluster: {cluster}")

    if cluster == -1:
        st.error("⚠️ This country is classified as an Outlier (Noise) by DBSCAN.")
    else:
        st.info(f"This country belongs to Cluster {cluster}.")

    # Show cluster size
    cluster_size = (labels == cluster).sum() if cluster != -1 else 1
    st.write(f"Countries in this cluster: **{cluster_size}**")

    # Show countries in the same cluster
    if cluster != -1:
        original_labels = labels[:-1]
        countries = df.loc[original_labels == cluster, "country"].tolist()

        st.subheader("Similar Countries")
        st.write(countries)