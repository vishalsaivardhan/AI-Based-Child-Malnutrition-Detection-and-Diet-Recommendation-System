import pickle
import pandas as pd
from recommend import get_diet_plan

# Load the trained model
try:
    with open('malnutrition_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file not found. Please run 'train_model.py' first.")
    exit()

print("="*40)
print(" AI CHILD MALNUTRITION DETECTION SYSTEM ")
print("="*40)

# User Inputs
age = float(input("Enter Age (months): "))
weight = float(input("Enter Weight (kg): "))
height = float(input("Enter Height (cm): "))

# Calculate BMI (weight / (height in meters)^2)
bmi = weight / ((height / 100) ** 2)

# Prepare data for prediction (must match training features)
input_data = pd.DataFrame([[age, weight, height, bmi]], 
                         columns=['age_months', 'weight_kg', 'height_cm', 'bmi'])

# Predict Status
prediction = model.predict(input_data)[0]

print(f"\n[RESULTS]")
print(f"Calculated BMI: {bmi:.2f}")
print(f"Predicted Nutrition Status: {prediction.upper()}")
# Get Diet Plan
print("\n[RECOMMENDED DIET PLAN]")
diet = get_diet_plan(prediction)

if diet is not None:
    print(diet.to_string(index=False))
else:
    print("⚠️ Error: Nutrition dataset not found. Please ensure 'Indian_Food_Nutrition_Processed.csv' is in the CSP folder.")

print("="*40)