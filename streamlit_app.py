import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Load model
model = load('assets/iris_model.joblib')

# Page config
st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")

st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements below to predict the species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Predict
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
predicted_species = species_map[prediction]

# Display result
st.subheader("🌼 Predicted Species:")
st.success(predicted_species)

# Optional: Show probabilities
if st.checkbox("Show prediction confidence"):
    probs = model.predict_proba(input_data)[0]
    prob_df = pd.DataFrame({
        "Species": [species_map[i] for i in range(3)],
        "Confidence": [f"{p*100:.2f}%" for p in probs]
    })
    st.table(prob_df)
