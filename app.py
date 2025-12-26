import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ------------------------------------
# Streamlit Page Config
# ------------------------------------
st.set_page_config(
    page_title="Customer Personality Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Personality Segmentation App")
st.write("Upload customer data to predict customer clusters using K-Means.")

# ------------------------------------
# Load Saved Models
# ------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca_model.joblib")
    model = joblib.load("K-means is best_clustering_model.joblib")
    return scaler, pca, model

scaler, pca, model = load_models()

# ------------------------------------
# File Upload
# ------------------------------------
uploaded_file = st.file_uploader(
    "Upload Excel or CSV file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # ------------------------------------
    # Read Data
    # ------------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # ------------------------------------
    # Preprocessing (MATCHES NOTEBOOK)
    # ------------------------------------
    st.subheader("⚙️ Data Preprocessing")

    df_processed = df.copy()

    # Drop non-numerical / ID columns (from marketing campaign dataset)
    drop_cols = [
        'ID','Dt_Customer','Z_CostContact','Z_Revenue'
    ]
    df_processed.drop(
        columns=[col for col in drop_cols if col in df_processed.columns],
        inplace=True,
        errors='ignore'
    )

    # Select numerical columns only
    numerical_cols = df_processed.select_dtypes(include=np.number).columns
    df_processed = df_processed[numerical_cols]

    # Handle missing values
    df_processed.fillna(df_processed.median(), inplace=True)

    st.success("Preprocessing completed successfully.")

    # ------------------------------------
    # Scaling + PCA
    # ------------------------------------
    scaled_data = scaler.transform(df_processed)
    pca_data = pca.transform(scaled_data)

    # ------------------------------------
    # Prediction
    # ------------------------------------
    if st.button("🚀 Predict Customer Segments"):

        clusters = model.predict(pca_data)
        df["Customer_Segment"] = clusters

        st.subheader("✅ Clustered Output")
        st.dataframe(df.head(20))

        # ------------------------------------
        # Cluster Distribution
        # ------------------------------------
        st.subheader("📊 Cluster Distribution")
        st.bar_chart(df["Customer_Segment"].value_counts())

        # ------------------------------------
        # Download Results
        # ------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Clustered Data",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )

