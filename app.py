import streamlit as st
import pickle
import pandas as pd
import os
from recommend import get_diet_plan

# Set Page Config
st.set_page_config(page_title="AI Malnutrition Detector", page_icon="🥗")

# Load the model using a robust path for Cloud Deployment
@st.cache_resource
def load_model():
    # This finds the directory where app.py is located on the server
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'malnutrition_model.pkl')
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- UI Header ---
st.title("🥗 AI Child Malnutrition Detection & Diet System")
st.markdown("This system uses Machine Learning to predict malnutrition levels and suggest Indian dietary improvements.")
st.divider()

# --- Sidebar Inputs ---
st.sidebar.header("Child's Health Parameters")
age = st.sidebar.number_input("Age (Months)", min_value=1, max_value=60, value=24)
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, max_value=30.0, value=12.0, step=0.1)
height = st.sidebar.number_input("Height (cm)", min_value=40.0, max_value=120.0, value=85.0, step=0.1)

# --- Calculation ---
bmi = weight / ((height / 100) ** 2)

if st.sidebar.button("Predict Status"):
    # Prepare Data
    input_data = pd.DataFrame([[age, weight, height, bmi]], 
                             columns=['age_months', 'weight_kg', 'height_cm', 'bmi'])
    
    # Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Calculated BMI", f"{bmi:.2f}")
    with col2:
        status_color = "green" if prediction.lower() == "normal" else "red"
        st.markdown(f"### Status: :{status_color}[{prediction.upper()}]")
    
    st.divider()
    
    # Display Diet
    st.subheader("📋 Recommended Indian Diet Plan")
    diet = get_diet_plan(prediction)
    
    if diet is not None:
        st.table(diet)
    else:
        st.error("Nutrition dataset not found!")
    
    # Community Advice
    st.info("**Community Note:** Please consult a healthcare professional for clinical diagnosis.")