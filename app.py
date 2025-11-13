

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from model import HybridPSOGA

st.title("🧠 Hybrid PSO-GA Feature Selection App")
st.write("Upload your dataset (CSV) and select the target column.")

# Upload CSV
file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.write("### Preview of Data")
    st.dataframe(df.head())

    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Run Optimization"):
        try:
            st.write("### Preprocessing Data...")

            # Copy dataframe to avoid modifying original
            df_proc = df.copy() 

            # Convert all categorical features to numeric
            for col in df_proc.columns:
                if df_proc[col].dtype == "object" or isinstance(df_proc[col].iloc[0], str):
                    df_proc[col] = LabelEncoder().fit_transform(df_proc[col].astype(str))

            # Separate features and target
            X = df_proc.drop(columns=[target_col]).values.astype(float)
            y = df_proc[target_col].values

            st.write("### Running Hybrid PSO-GA Optimization... ⏳")

            optimizer = HybridPSOGA(
                n_particles=20,
                n_iterations=30,
                classifier_type='rf'
            )

            best_features_mask = optimizer.optimize(X, y)
            selected_features = df_proc.drop(columns=[target_col]).columns[best_features_mask.astype(bool)]

            st.success(f"Optimization Complete ✅ — {len(selected_features)} features selected.")
            st.write("### Selected Features:")
            st.write(selected_features.tolist())

            # Optionally: Train final model with selected features
            st.write("### Training final RandomForest model with selected features...")
            final_model = RandomForestClassifier(n_estimators=100, random_state=42)
            final_model.fit(X[:, best_features_mask.astype(bool)], y)
            st.success("Final model trained successfully!")

        except Exception as e:
            st.error(f"Error: {e}")



