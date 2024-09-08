import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv")
    return data

@st.cache_resource
def train_model(data):
    X = data[['age', 'bmi', 'children', 'smoker', 'region']]
    y = data['charges']
    X = pd.get_dummies(X, drop_first=True)

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X.columns

# App title
st.title('Insurance Cost Predictor')

# Load data and train model
data = load_data()
model, feature_names = train_model(data)

# User input
st.header('Enter Your Information')
age = st.number_input('Age', min_value=18, max_value=100, value=30)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['northeast', 'northwest', 'southeast', 'southwest'])

# Predict button
if st.button('Predict Insurance Cost'):
    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Ensure all columns from training are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Make prediction
    prediction = model.predict(input_data[feature_names])

    # Display result
    st.success(f'Predicted Insurance Cost: ${prediction[0]:,.2f}')

# Optional: Display the dataset
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(data)