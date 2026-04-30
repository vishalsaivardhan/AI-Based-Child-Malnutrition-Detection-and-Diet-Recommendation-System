import pandas as pd
import os

def get_diet_plan(status):
    base_path = os.path.dirname(os.path.abspath(__file__))
    # DOUBLE CHECK THIS FILENAME matches your sidebar exactly
    file_path = os.path.join(base_path, 'Indian_Food_Nutrition_Processed.csv')
    
    if not os.path.exists(file_path):
        return None # Return None if file is missing
    
    nutrition_df = pd.read_csv(file_path)
    
    # Recommendation Logic
    if status.lower() in ['moderate', 'severe']:
        recommendations = nutrition_df.sort_values(by=['Protein (g)', 'Calories (kcal)'], ascending=False).head(5)
    else:
        recommendations = nutrition_df.sample(5)
        
    return recommendations[['Dish Name', 'Protein (g)', 'Calories (kcal)']]
