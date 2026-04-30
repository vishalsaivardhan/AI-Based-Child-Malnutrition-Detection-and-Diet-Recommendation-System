import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load the Malnutrition Dataset
# Ensure the file is in the same folder as this script
df = pd.read_csv('C:\\Users\\visha\\OneDrive\\Desktop\\CSP\\data\\children-malnutrition-dataset.csv')

# 2. Define Features and Target based on your file columns
# Features: age_months, weight_kg, height_cm, bmi
X = df[['age_months', 'weight_kg', 'height_cm', 'bmi']]
# Target: nutrition_status
y = df['nutrition_status'] 

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the trained model
with open('malnutrition_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model training complete! Saved as 'malnutrition_model.pkl'.")
print(f"Accuracy on test set: {model.score(X_test, y_test):.2%}")