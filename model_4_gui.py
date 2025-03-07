import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the trained model (replace with your model path)
model = tf.keras.models.load_model('student_performance_model.h5')

# Define the columns and their possible values
numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
ordinal_cols = {
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Far', 'Moderate', 'Near']
}
binary_cols = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'Gender']
nominal_cols = ['School_Type', 'Peer_Influence']

# Create a dictionary to store user inputs
input_data = {col: None for col in numerical_cols + list(ordinal_cols.keys()) + binary_cols + nominal_cols}

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert ordinal columns to numerical values
    for col, levels in ordinal_cols.items():
        input_data[col] = levels.index(input_data[col])
    
    # Convert binary columns to 0/1
    input_data['Extracurricular_Activities'] = 1 if input_data['Extracurricular_Activities'] == 'Yes' else 0
    input_data['Internet_Access'] = 1 if input_data['Internet_Access'] == 'Yes' else 0
    input_data['Learning_Disabilities'] = 1 if input_data['Learning_Disabilities'] == 'Yes' else 0
    input_data['Gender'] = 1 if input_data['Gender'] == 'Male' else 0
    
    # One-hot encode nominal columns
    input_data['School_Type_Public'] = 1 if input_data['School_Type'] == 'Public' else 0
    input_data['School_Type_Private'] = 1 if input_data['School_Type'] == 'Private' else 0
    input_data['Peer_Influence_Positive'] = 1 if input_data['Peer_Influence'] == 'Positive' else 0
    input_data['Peer_Influence_Negative'] = 1 if input_data['Peer_Influence'] == 'Negative' else 0
    input_data['Peer_Influence_Neutral'] = 1 if input_data['Peer_Influence'] == 'Neutral' else 0
    
    # Remove the original nominal columns
    del input_data['School_Type']
    del input_data['Peer_Influence']
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Normalize numerical columns
    scaler = StandardScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    return input_df

# Function to predict the score
def predict_score():
    try:
        # Get user inputs
        for col in numerical_cols:
            input_data[col] = float(entries[col].get())
        for col in ordinal_cols.keys():
            input_data[col] = entries[col].get()
        for col in binary_cols:
            input_data[col] = entries[col].get()
        for col in nominal_cols:
            input_data[col] = entries[col].get()
        
        # Preprocess the input data
        input_df = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Show the result
        messagebox.showinfo("Prediction", f"Predicted Exam Score: {prediction[0][0]:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the GUI
root = tk.Tk()
root.title("Student Exam Score Predictor")

# Create input fields
entries = {}
row = 0

# Numerical columns
for col in numerical_cols:
    ttk.Label(root, text=col).grid(row=row, column=0, padx=10, pady=5)
    entries[col] = ttk.Entry(root)
    entries[col].grid(row=row, column=1, padx=10, pady=5)
    row += 1

# Ordinal columns
for col, levels in ordinal_cols.items():
    ttk.Label(root, text=col).grid(row=row, column=0, padx=10, pady=5)
    entries[col] = ttk.Combobox(root, values=levels)
    entries[col].grid(row=row, column=1, padx=10, pady=5)
    row += 1

# Binary columns
for col in binary_cols:
    ttk.Label(root, text=col).grid(row=row, column=0, padx=10, pady=5)
    entries[col] = ttk.Combobox(root, values=['Yes', 'No'])
    entries[col].grid(row=row, column=1, padx=10, pady=5)
    row += 1

# Nominal columns
for col in nominal_cols:
    ttk.Label(root, text=col).grid(row=row, column=0, padx=10, pady=5)
    if col == 'School_Type':
        entries[col] = ttk.Combobox(root, values=['Public', 'Private'])
    elif col == 'Peer_Influence':
        entries[col] = ttk.Combobox(root, values=['Positive', 'Negative', 'Neutral'])
    entries[col].grid(row=row, column=1, padx=10, pady=5)
    row += 1

# Predict button
ttk.Button(root, text="Predict Score", command=predict_score).grid(row=row, column=0, columnspan=2, pady=10)

# Run the GUI
root.mainloop()