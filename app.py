import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = tf.keras.models.load_model('grid_stability_model.h5')

# Define the Streamlit app
st.title("Electrical Grid Stability Prediction")
st.write("Enter the required features to predict grid stability (stable or unstable).")

# Input fields for user data
tau1 = st.number_input("Reaction Time (tau1)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
tau2 = st.number_input("Reaction Time (tau2)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
tau3 = st.number_input("Reaction Time (tau3)", min_value=0.5, max_value=10.0, value=8.0, step=0.1)
tau4 = st.number_input("Reaction Time (tau4)", min_value=0.5, max_value=10.0, value=9.0, step=0.1)
p2 = st.number_input("Nominal Power Consumed/Produced (p2)", min_value=-2.0, max_value=-0.5, value=-1.0, step=0.1)
p3 = st.number_input("Nominal Power Consumed/Produced (p3)", min_value=-2.0, max_value=-0.5, value=-1.5, step=0.1)
p4 = st.number_input("Nominal Power Consumed/Produced (p4)", min_value=-2.0, max_value=-0.5, value=-1.3, step=0.1)
g1 = st.number_input("Elasticity Coefficient (g1)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
g2 = st.number_input("Elasticity Coefficient (g2)", min_value=0.05, max_value=1.0, value=0.6, step=0.05)
g3 = st.number_input("Elasticity Coefficient (g3)", min_value=0.05, max_value=1.0, value=0.7, step=0.05)
g4 = st.number_input("Elasticity Coefficient (g4)", min_value=0.05, max_value=1.0, value=0.8, step=0.05)

# Collect input features
input_data = np.array([[tau1, tau2, tau3, tau4, p2, p3, p4, g1, g2, g3, g4]])

# Perform prediction
if st.button("Predict Stability"):
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)  # Scale the input data
    prediction = (model.predict(input_scaled) > 0.5).astype(int)
    
    # Map prediction to label
    result = "Stable" if prediction[0][0] == 0 else "Unstable"
    st.subheader(f"Prediction: {result}")

